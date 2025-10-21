use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    block::Block,
    block_face_functions::{
        build_connectivity_graph, find_matching_faces, standardize_block_orientation,
    },
    connectivity::FaceMatch,
};

/// Result type for `combine_blocks_mixed_pairs`.
pub type CombinedBlocks = (Vec<Block>, Vec<usize>);

/// Merge two compatible blocks by aligning and stacking matched faces.
///
/// This is a fairly literal translation of the Python logic. Given a face pair
/// reported by `find_matching_faces`, we permute/flip the second block so the
/// matching faces line up, ensure both blocks increase in the same physical
/// direction along the stacking axis, trim the overlapping slice, then
/// concatenate before re-standardising orientation.
pub fn combine_2_blocks_mixed_pairing(block1: &Block, block2: &Block, tol: f64) -> Block {
    let Some((face1, face2, (flip_ud, flip_lr))) = find_matching_faces(block1, block2, tol) else {
        return block1.clone();
    };

    let (axis1, _dir1) = face_axis_info(face1);
    let (axis2, _dir2) = face_axis_info(face2);

    let mut base = block1.clone();
    let mut other = block2.clone();

    if axis1 != axis2 {
        let mut perm = [0usize, 1, 2];
        perm.swap(axis1, axis2);
        other = permute_block_axes(&other, perm);
    }

    other = apply_face_flips(&other, face2, flip_ud, flip_lr);

    let stack_axis = axis1;
    let step1 = dominant_step(&base, stack_axis);
    let step2 = dominant_step(&other, stack_axis);

    if step1.signum() != 0.0 && step2.signum() != 0.0 && step1.signum() != step2.signum() {
        base = flip_block_axis(&base, stack_axis);
    }

    let drop_first = face2.ends_with("min");
    let trimmed_other = trim_block_along_axis(&other, stack_axis, drop_first);

    let merged = if drop_first {
        concat_blocks_along_axis(&base, &trimmed_other, stack_axis)
    } else {
        concat_blocks_along_axis(&trimmed_other, &base, stack_axis)
    };

    standardize_block_orientation(&merged)
}

/// Attempt to merge as many blocks as possible from an initial set.
///
/// Starting from the provided block list (typically ≤ 8 items), this makes
/// repeated passes trying to merge any pair whose faces match. The search is
/// greedy, mirroring the original Python helper: once a pair is merged it is
/// replaced by the new block and the pass restarts until no further reductions
/// occur or `max_tries` is hit.
pub fn combine_blocks_mixed_pairs(
    blocks: &[Block],
    tol: f64,
    max_tries: usize,
) -> CombinedBlocks {
    let mut merged_blocks: Vec<Block> = blocks.to_vec();
    let mut tries = 0usize;

    while merged_blocks.len() > 1 && tries < max_tries {
        let mut consumed = vec![false; merged_blocks.len()];
        let mut next_blocks = Vec::new();
        let mut any_merge = false;

        for i in 0..merged_blocks.len() {
            if consumed[i] {
                continue;
            }
            consumed[i] = true;
            let mut merged: Option<Block> = None;
            for j in (i + 1)..merged_blocks.len() {
                if consumed[j] {
                    continue;
                }
                if find_matching_faces(&merged_blocks[i], &merged_blocks[j], tol).is_some() {
                    let candidate =
                        combine_2_blocks_mixed_pairing(&merged_blocks[i], &merged_blocks[j], tol);
                    consumed[j] = true;
                    merged = Some(candidate);
                    any_merge = true;
                    break;
                }
            }

            if let Some(m) = merged {
                next_blocks.push(m);
            } else {
                next_blocks.push(merged_blocks[i].clone());
            }
        }

        if !any_merge {
            break;
        }

        merged_blocks = next_blocks;
        tries += 1;
    }

    let used_indices = (0..blocks.len()).collect();
    (merged_blocks, used_indices)
}

/// Merge all discoverable n×n×n cube groupings using connectivity data.
///
/// `connectivities` should be the face matches returned by `connectivity`/
/// `connectivity_fast`. The routine builds a graph, performs BFS to locate
/// candidate cube groupings (`cube_size^3` nodes), merges their blocks using
/// `combine_blocks_mixed_pairs`, and keeps track of which original indices fed
/// each merged component.
pub fn combine_nxnxn_cubes_mixed_pairs(
    blocks: &[Block],
    connectivities: &[FaceMatch],
    cube_size: usize,
    tol: f64,
) -> Vec<(Block, HashSet<usize>)> {
    if cube_size == 0 {
        return Vec::new();
    }
    let target_size = cube_size.pow(3);

    let graph = build_connectivity_graph(connectivities);
    let mut used: HashSet<usize> = HashSet::new();
    let mut remaining: Vec<usize> = (0..blocks.len()).collect();
    let mut merged_groups = Vec::new();

    loop {
        let before_len = remaining.len();
        let mut merged_this_round = false;
        let mut new_used: HashSet<usize> = HashSet::new();

        let mut idx = 0usize;
        while idx < remaining.len() {
            let seed = remaining[idx];
            if used.contains(&seed) {
                idx += 1;
                continue;
            }

            let group_opt = find_nxnxn_group(seed, &graph, &used, target_size);
            let Some(group) = group_opt else {
                idx += 1;
                continue;
            };

            if !group.is_disjoint(&new_used) {
                idx += 1;
                continue;
            }

            let mut sorted_group: Vec<usize> = group.iter().copied().collect();
            sorted_group.sort_unstable();

            let group_blocks: Vec<Block> = sorted_group.iter().map(|&i| blocks[i].clone()).collect();
            let (partial_merges, local_indices) =
                combine_blocks_mixed_pairs(&group_blocks, tol, cube_size);

            let index_mapping: HashMap<usize, usize> = sorted_group
                .iter()
                .enumerate()
                .map(|(local, &global)| (local, global))
                .collect();

            for merged_block in partial_merges {
                let mut merged_ids = HashSet::new();
                for &local in &local_indices {
                    if let Some(global) = index_mapping.get(&local) {
                        merged_ids.insert(*global);
                    }
                }
                if merged_ids.is_empty() {
                    continue;
                }
                new_used.extend(&merged_ids);
                merged_groups.push((merged_block, merged_ids));
            }

            merged_this_round = true;
            remaining.retain(|idx| !new_used.contains(idx));
            idx = 0;
        }

        used.extend(&new_used);

        if !merged_this_round || remaining.len() == before_len {
            for idx in remaining {
                if used.contains(&idx) {
                    continue;
                }
                let mut set = HashSet::new();
                set.insert(idx);
                merged_groups.push((blocks[idx].clone(), set));
            }
            break;
        }
    }

    merged_groups
}

/// Perform a breadth-first search from `seed` to find a cube-sized group that
/// has not yet been merged or marked as used.
fn find_nxnxn_group(
    seed: usize,
    graph: &HashMap<usize, HashSet<usize>>,
    used: &HashSet<usize>,
    target_size: usize,
) -> Option<HashSet<usize>> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(seed);

    while let Some(idx) = queue.pop_front() {
        if visited.contains(&idx) || used.contains(&idx) {
            continue;
        }
        visited.insert(idx);
        if visited.len() == target_size {
            break;
        }
        if let Some(neighbors) = graph.get(&idx) {
            for &nbr in neighbors {
                if !visited.contains(&nbr) && !used.contains(&nbr) {
                    queue.push_back(nbr);
                }
            }
        }
    }

    if visited.len() == target_size {
        Some(visited)
    } else {
        None
    }
}

/// Map a face name to its corresponding axis (0/1/2) and sign direction.
fn face_axis_info(face: &str) -> (usize, i32) {
    match face {
        "imin" => (0, -1),
        "imax" => (0, 1),
        "jmin" => (1, -1),
        "jmax" => (1, 1),
        "kmin" => (2, -1),
        "kmax" => (2, 1),
        _ => (0, 0),
    }
}

/// Reorder the block axes according to `perm`, returning a new block.
fn permute_block_axes(block: &Block, perm: [usize; 3]) -> Block {
    let dims = [block.imax, block.jmax, block.kmax];
    let new_dims = [
        dims[perm[0]],
        dims[perm[1]],
        dims[perm[2]],
    ];
    let mut x = vec![0.0; new_dims[0] * new_dims[1] * new_dims[2]];
    let mut y = x.clone();
    let mut z = x.clone();

    for i_new in 0..new_dims[0] {
        for j_new in 0..new_dims[1] {
            for k_new in 0..new_dims[2] {
                let mut old = [0usize; 3];
                old[perm[0]] = i_new;
                old[perm[1]] = j_new;
                old[perm[2]] = k_new;
                let (vx, vy, vz) = block.xyz(old[0], old[1], old[2]);
                let idx = linear_index(new_dims, [i_new, j_new, k_new]);
                x[idx] = vx;
                y[idx] = vy;
                z[idx] = vz;
            }
        }
    }

    Block::new(new_dims[0], new_dims[1], new_dims[2], x, y, z)
}

/// Apply the up/down and left/right flips implied by `flip_ud`/`flip_lr`.
fn apply_face_flips(block: &Block, face: &str, flip_ud: bool, flip_lr: bool) -> Block {
    let mut result = block.clone();
    match face {
        "imin" | "imax" => {
            if flip_ud {
                result = flip_block_axis(&result, 1);
            }
            if flip_lr {
                result = flip_block_axis(&result, 2);
            }
        }
        "jmin" | "jmax" => {
            if flip_ud {
                result = flip_block_axis(&result, 0);
            }
            if flip_lr {
                result = flip_block_axis(&result, 2);
            }
        }
        "kmin" | "kmax" => {
            if flip_ud {
                result = flip_block_axis(&result, 0);
            }
            if flip_lr {
                result = flip_block_axis(&result, 1);
            }
        }
        _ => {}
    }
    result
}

/// Determine which coordinate changes the most along `axis`.
fn dominant_step(block: &Block, axis: usize) -> f64 {
    let steps = [
        coordinate_step(block, axis, 0),
        coordinate_step(block, axis, 1),
        coordinate_step(block, axis, 2),
    ];
    steps
        .iter()
        .copied()
        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap_or(0.0)
}

/// Compute the signed step between the first and last plane along `axis`
/// for the requested coordinate component (0 → X, 1 → Y, 2 → Z).
fn coordinate_step(block: &Block, axis: usize, component: usize) -> f64 {
    let dims = [block.imax, block.jmax, block.kmax];
    if dims[axis] <= 1 {
        return 0.0;
    }
    let mut start = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
    let mut end = start;
    start[axis] = 0;
    end[axis] = dims[axis] - 1;
    let start_val = component_value(block, start, component);
    let end_val = component_value(block, end, component);
    end_val - start_val
}

fn component_value(block: &Block, idx: [usize; 3], component: usize) -> f64 {
    match component {
        0 => block.x[linear_index([block.imax, block.jmax, block.kmax], idx)],
        1 => block.y[linear_index([block.imax, block.jmax, block.kmax], idx)],
        _ => block.z[linear_index([block.imax, block.jmax, block.kmax], idx)],
    }
}

/// Create a copy of the block with the specified axis reversed.
fn flip_block_axis(block: &Block, axis: usize) -> Block {
    let dims = [block.imax, block.jmax, block.kmax];
    let mut x = vec![0.0; block.npoints()];
    let mut y = x.clone();
    let mut z = x.clone();

    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let mut src = [i, j, k];
                src[axis] = dims[axis] - 1 - src[axis];
                let (vx, vy, vz) = block.xyz(src[0], src[1], src[2]);
                let idx = linear_index(dims, [i, j, k]);
                x[idx] = vx;
                y[idx] = vy;
                z[idx] = vz;
            }
        }
    }

    Block::new(dims[0], dims[1], dims[2], x, y, z)
}

/// Drop the overlapping slice along `axis` (front or back) before concatenation.
fn trim_block_along_axis(block: &Block, axis: usize, drop_first: bool) -> Block {
    let dims = [block.imax, block.jmax, block.kmax];
    if dims[axis] <= 1 {
        return block.clone();
    }
    let mut new_dims = dims;
    new_dims[axis] -= 1;
    let mut x = vec![0.0; new_dims[0] * new_dims[1] * new_dims[2]];
    let mut y = x.clone();
    let mut z = x.clone();

    for i in 0..new_dims[0] {
        for j in 0..new_dims[1] {
            for k in 0..new_dims[2] {
                let mut src = [i, j, k];
                if drop_first {
                    src[axis] += 1;
                }
                let (vx, vy, vz) = block.xyz(src[0], src[1], src[2]);
                let idx = linear_index(new_dims, [i, j, k]);
                x[idx] = vx;
                y[idx] = vy;
                z[idx] = vz;
            }
        }
    }

    Block::new(new_dims[0], new_dims[1], new_dims[2], x, y, z)
}

/// Concatenate two blocks along `axis`, assuming matching cross-sections.
fn concat_blocks_along_axis(a: &Block, b: &Block, axis: usize) -> Block {
    let dims_a = [a.imax, a.jmax, a.kmax];
    let dims_b = [b.imax, b.jmax, b.kmax];
    let mut new_dims = dims_a;
    new_dims[axis] += dims_b[axis];

    for idx in 0..3 {
        if idx != axis && dims_a[idx] != dims_b[idx] {
            panic!("Block dimensions do not match for concatenation");
        }
    }

    let total = new_dims[0] * new_dims[1] * new_dims[2];
    let mut x = vec![0.0; total];
    let mut y = x.clone();
    let mut z = x.clone();

    for i in 0..new_dims[0] {
        for j in 0..new_dims[1] {
            for k in 0..new_dims[2] {
                let idx_new = linear_index(new_dims, [i, j, k]);
                let coord = if coordinate_from_block(dims_a, axis, [i, j, k]) {
                    let src = [i, j, k];
                    a.xyz(src[0], src[1], src[2])
                } else {
                    let mut src = [i, j, k];
                    src[axis] -= dims_a[axis];
                    b.xyz(src[0], src[1], src[2])
                };
                x[idx_new] = coord.0;
                y[idx_new] = coord.1;
                z[idx_new] = coord.2;
            }
        }
    }

    Block::new(new_dims[0], new_dims[1], new_dims[2], x, y, z)
}

fn coordinate_from_block(dims_a: [usize; 3], axis: usize, idx: [usize; 3]) -> bool {
    idx[axis] < dims_a[axis]
}

fn linear_index(dims: [usize; 3], idx: [usize; 3]) -> usize {
    (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0]
}
