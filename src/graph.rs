//! Graph partitioning utilities for structured multi-block grids.
//!
//! The `partition_from_face_matches` function requires the `metis-partition` Cargo feature.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::connectivity::FaceMatch;

/// Strategy for combining weights when multiple faces connect the same
/// pair of blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WeightAggregate {
    Sum,
    Max,
    Min,
}

/// Adjacency and edge-weight data for a block graph.
#[derive(Clone, Debug)]
pub struct BlockGraph {
    /// Adjacency list: `adj_list[u]` is a sorted, deduplicated list of
    /// neighbors for block `u`.
    pub adj_list: Vec<Vec<usize>>,
    /// Edge weights: `edge_weights[u][v]` = communication cost between `u` and `v`.
    pub edge_weights: Vec<HashMap<usize, i64>>,
}

/// CSR (Compressed Sparse Row) representation of a block graph.
#[derive(Clone, Debug)]
pub struct CsrGraph {
    pub xadj: Vec<i32>,
    pub adjncy: Vec<i32>,
    pub eweights: Vec<i32>,
}

/// Build a weighted graph from face-match data.
///
/// Each face match contributes a weight equal to the product of the face
/// dimension ranges (`dI * dJ * dK`), representing the communication cost
/// (number of shared nodes).
pub fn build_weighted_graph_from_face_matches(
    face_matches: &[FaceMatch],
    n_blocks: usize,
    aggregate: WeightAggregate,
    ignore_self_matches: bool,
) -> BlockGraph {
    let mut pair_weight: HashMap<(usize, usize), i64> = HashMap::new();

    for m in face_matches {
        let i = m.block1.block_index;
        let j = m.block2.block_index;
        if ignore_self_matches && i == j {
            continue;
        }

        let di = (m.block1.imax as i64 - m.block1.imin as i64).unsigned_abs().max(1);
        let dj = (m.block1.jmax as i64 - m.block1.jmin as i64).unsigned_abs().max(1);
        let dk = (m.block1.kmax as i64 - m.block1.kmin as i64).unsigned_abs().max(1);
        let w = (di * dj * dk) as i64;

        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let entry = pair_weight.entry((a, b)).or_insert(0);
        match aggregate {
            WeightAggregate::Sum => *entry += w,
            WeightAggregate::Max => *entry = (*entry).max(w),
            WeightAggregate::Min => {
                if *entry == 0 {
                    *entry = w;
                } else {
                    *entry = (*entry).min(w);
                }
            }
        }
    }

    let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    let mut edge_weights: Vec<HashMap<usize, i64>> = vec![HashMap::new(); n_blocks];

    for (&(a, b), &w) in &pair_weight {
        adj_list[a].push(b);
        adj_list[b].push(a);
        edge_weights[a].insert(b, w);
        edge_weights[b].insert(a, w);
    }

    // Sort and deduplicate adjacency lists
    for list in &mut adj_list {
        list.sort_unstable();
        list.dedup();
    }

    BlockGraph {
        adj_list,
        edge_weights,
    }
}

/// Convert a [`BlockGraph`] to CSR arrays suitable for METIS.
pub fn csr_from_block_graph(graph: &BlockGraph) -> CsrGraph {
    let mut xadj: Vec<i32> = vec![0];
    let mut adjncy: Vec<i32> = Vec::new();
    let mut eweights: Vec<i32> = Vec::new();

    let mut count: i32 = 0;
    for u in 0..graph.adj_list.len() {
        for &v in &graph.adj_list[u] {
            adjncy.push(v as i32);
            let w = graph.edge_weights[u].get(&v).copied().unwrap_or(1);
            eweights.push(w as i32);
            count += 1;
        }
        xadj.push(count);
    }

    CsrGraph {
        xadj,
        adjncy,
        eweights,
    }
}

/// Partition blocks using METIS graph partitioning.
///
/// Requires the `metis-partition` Cargo feature to be enabled.
///
/// Returns `(parts, graph)` where `parts[i]` is the 0-based partition ID
/// for block `i`.
#[cfg(feature = "metis-partition")]
pub fn partition_from_face_matches(
    face_matches: &[FaceMatch],
    block_sizes: &[usize],
    nparts: usize,
    favor_blocksize: bool,
    aggregate: WeightAggregate,
    ignore_self_matches: bool,
) -> Result<(Vec<i32>, BlockGraph), String> {
    let n_blocks = block_sizes.len();
    let graph = build_weighted_graph_from_face_matches(
        face_matches,
        n_blocks,
        aggregate,
        ignore_self_matches,
    );
    let csr = csr_from_block_graph(&graph);

    let vwgt: Option<Vec<metis::Idx>> = if favor_blocksize {
        Some(block_sizes.iter().map(|&s| s as metis::Idx).collect())
    } else {
        None
    };

    let mut part = vec![0 as metis::Idx; n_blocks];

    // Build METIS graph and partition
    let ncon: metis::Idx = 1;
    let nparts_idx = nparts as metis::Idx;

    let result = metis::Graph::new(ncon, nparts_idx, &csr.xadj, &csr.adjncy);
    match result {
        Ok(mut g) => {
            if let Some(ref vw) = vwgt {
                g = g.set_vwgt(vw);
            }
            g = g.set_adjwgt(&csr.eweights);
            g.part_kway(&mut part)
                .map_err(|e| format!("METIS partitioning failed: {:?}", e))?;
        }
        Err(e) => return Err(format!("Failed to create METIS graph: {:?}", e)),
    }

    Ok((part, graph))
}

/// Write ddcmp.dat and ddcmp_info.txt files for domain decomposition.
///
/// `parts` are 0-based internally, written 1-based in the file.
pub fn write_ddcmp(
    parts: &[i32],
    block_sizes: &[usize],
    graph: &BlockGraph,
    filename: &str,
) -> std::io::Result<()> {
    let n_blocks = parts.len();
    let n_proc = parts.iter().copied().max().unwrap_or(-1) + 1;
    let n_isp = n_proc;

    // Write ddcmp.dat
    let path = Path::new(filename);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{}", n_proc)?;
    writeln!(f, "{}", n_isp)?;
    writeln!(f, "{}", n_blocks)?;
    for b_idx in 0..n_blocks {
        writeln!(f, "{} {}", b_idx + 1, parts[b_idx] + 1)?;
    }
    for isp in 0..n_isp {
        writeln!(f, "{} {}", isp + 1, isp)?;
    }

    // Compute statistics
    let np = n_proc as usize;
    let mut communication_work = vec![0i64; np];
    let mut partition_edge_weights = vec![0i64; np];
    let mut volume_nodes = vec![0usize; np];

    for (b, &bsize) in block_sizes.iter().enumerate() {
        let pid = parts[b] as usize;
        volume_nodes[pid] += bsize;
    }

    for b in 0..block_sizes.len() {
        let pid = parts[b] as usize;
        for &nbr in &graph.adj_list[b] {
            let nbr_pid = parts[nbr] as usize;
            if nbr_pid != pid {
                communication_work[pid] += 1;
                partition_edge_weights[pid] +=
                    graph.edge_weights[b].get(&nbr).copied().unwrap_or(1);
            }
        }
    }

    // Write ddcmp_info.txt
    let info_path = path.with_file_name("ddcmp_info.txt");
    let mut fi = std::fs::File::create(&info_path)?;
    for i in 0..np {
        let block_count = parts.iter().filter(|&&p| p == i as i32).count();
        writeln!(fi, "Parition {} has {} blocks", i, block_count)?;
    }
    writeln!(fi, "Number of partitions/processors {}", n_proc)?;
    for i in 0..np {
        writeln!(
            fi,
            "Parition or processor {} has communication work {} edge_work {} volume_nodes {}",
            i, communication_work[i], partition_edge_weights[i], volume_nodes[i]
        )?;
    }
    let total_comm: i64 = communication_work.iter().sum();
    let total_edge: i64 = partition_edge_weights.iter().sum();
    writeln!(
        fi,
        "Total communication work {} Total edge_work {}",
        total_comm, total_edge
    )?;

    Ok(())
}
