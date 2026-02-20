//! Fast diagnostic test using the subset mesh exported by test_connectivity_debug.
//!
//! This test loads the subset (~2,629 blocks, 9.4GB) instead of the full 20GB mesh,
//! enabling quick iteration when debugging connectivity failures.
//!
//! Prerequisites:
//!   1. Run the full test first to export the subset:
//!      cargo test --release --test test_connectivity_debug connectivity_debug -- --nocapture
//!   2. Then iterate quickly with:
//!      cargo test --release --test test_connectivity_subset_diagnostics -- --nocapture

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use plot3d::{
    connectivity_fast, full_face_match, get_face_intersection, get_outer_faces, read_plot3d_binary,
    reduce_blocks, BinaryFormat, Endian, FaceMatch, FaceRecord, FloatPrecision,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SUBSET_FILE: &str = "/tmp/connectivity_debug_subset.p3d";
const SUBSET_INDICES_FILE: &str = "/tmp/connectivity_debug_subset_indices.json";
const JSON_FILE: &str =
    "/Users/pjuangph/Documents/GitHub/plot3d_troubleshoot/gridpro_connectivity.json";

// ---------------------------------------------------------------------------
// JSON deserialization (same as main test)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct JsonBlockRecord {
    block_index: usize,
    #[serde(rename = "IMIN")]
    imin: usize,
    #[serde(rename = "JMIN")]
    jmin: usize,
    #[serde(rename = "KMIN")]
    kmin: usize,
    #[serde(rename = "IMAX")]
    imax: usize,
    #[serde(rename = "JMAX")]
    jmax: usize,
    #[serde(rename = "KMAX")]
    kmax: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonFaceMatch {
    block1: JsonBlockRecord,
    block2: JsonBlockRecord,
}

#[derive(Debug, Deserialize)]
struct GridProConnectivity {
    face_matches: Vec<JsonFaceMatch>,
}

// ---------------------------------------------------------------------------
// Normalized face record (orientation-agnostic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NormalizedFaceRecord {
    block_index: usize,
    imin: usize,
    jmin: usize,
    kmin: usize,
    imax: usize,
    jmax: usize,
    kmax: usize,
}

impl NormalizedFaceRecord {
    fn from_json_block(b: &JsonBlockRecord) -> Self {
        Self {
            block_index: b.block_index,
            imin: b.imin.min(b.imax),
            jmin: b.jmin.min(b.jmax),
            kmin: b.kmin.min(b.kmax),
            imax: b.imin.max(b.imax),
            jmax: b.jmin.max(b.jmax),
            kmax: b.kmin.max(b.kmax),
        }
    }

    fn from_face_record(r: &FaceRecord) -> Self {
        Self {
            block_index: r.block_index,
            imin: r.i_lo(),
            jmin: r.j_lo(),
            kmin: r.k_lo(),
            imax: r.i_hi(),
            jmax: r.j_hi(),
            kmax: r.k_hi(),
        }
    }

    fn face_tuple(&self) -> (usize, usize, usize, usize, usize, usize) {
        (self.imin, self.jmin, self.kmin, self.imax, self.jmax, self.kmax)
    }
}

// ---------------------------------------------------------------------------
// MatchKey: canonical unordered pair of faces
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct MatchKey {
    face_a: NormalizedFaceRecord,
    face_b: NormalizedFaceRecord,
}

impl MatchKey {
    fn new(a: NormalizedFaceRecord, b: NormalizedFaceRecord) -> Self {
        if a.block_index < b.block_index
            || (a.block_index == b.block_index && a.face_tuple() <= b.face_tuple())
        {
            MatchKey { face_a: a, face_b: b }
        } else {
            MatchKey { face_a: b, face_b: a }
        }
    }

    fn block_pair(&self) -> (usize, usize) {
        let lo = self.face_a.block_index.min(self.face_b.block_index);
        let hi = self.face_a.block_index.max(self.face_b.block_index);
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_json_keys(json_matches: &[JsonFaceMatch]) -> HashSet<MatchKey> {
    json_matches
        .iter()
        .map(|jm| {
            let a = NormalizedFaceRecord::from_json_block(&jm.block1);
            let b = NormalizedFaceRecord::from_json_block(&jm.block2);
            MatchKey::new(a, b)
        })
        .collect()
}

fn build_computed_keys(computed_matches: &[FaceMatch]) -> HashSet<MatchKey> {
    computed_matches
        .iter()
        .map(|fm| {
            let a = NormalizedFaceRecord::from_face_record(&fm.block1);
            let b = NormalizedFaceRecord::from_face_record(&fm.block2);
            MatchKey::new(a, b)
        })
        .collect()
}

fn group_by_block_pair(keys: &HashSet<MatchKey>) -> HashMap<(usize, usize), Vec<MatchKey>> {
    let mut map: HashMap<(usize, usize), Vec<MatchKey>> = HashMap::new();
    for k in keys {
        map.entry(k.block_pair()).or_default().push(k.clone());
    }
    map
}

/// Compute minimum corner distance between two JSON face records.
fn min_corner_distance(
    jm: &JsonFaceMatch,
    blocks: &[plot3d::Block],
) -> Option<(f64, bool)> {
    let b1_idx = jm.block1.block_index;
    let b2_idx = jm.block2.block_index;
    if b1_idx >= blocks.len() || b2_idx >= blocks.len() {
        return None;
    }
    let block1 = &blocks[b1_idx];
    let block2 = &blocks[b2_idx];

    let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
    let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);

    if n1.imax >= block1.imax || n1.jmax >= block1.jmax || n1.kmax >= block1.kmax {
        return None;
    }
    if n2.imax >= block2.imax || n2.jmax >= block2.jmax || n2.kmax >= block2.kmax {
        return None;
    }

    let (x1_lo, y1_lo, z1_lo) = block1.xyz(n1.imin, n1.jmin, n1.kmin);
    let (x1_hi, y1_hi, z1_hi) = block1.xyz(n1.imax, n1.jmax, n1.kmax);
    let (x2_lo, y2_lo, z2_lo) = block2.xyz(n2.imin, n2.jmin, n2.kmin);
    let (x2_hi, y2_hi, z2_hi) = block2.xyz(n2.imax, n2.jmax, n2.kmax);

    let d_direct = ((x1_lo - x2_lo).powi(2) + (y1_lo - y2_lo).powi(2) + (z1_lo - z2_lo).powi(2))
        .sqrt()
        .max(
            ((x1_hi - x2_hi).powi(2) + (y1_hi - y2_hi).powi(2) + (z1_hi - z2_hi).powi(2)).sqrt(),
        );
    let d_swapped = ((x1_lo - x2_hi).powi(2) + (y1_lo - y2_hi).powi(2) + (z1_lo - z2_hi).powi(2))
        .sqrt()
        .max(
            ((x1_hi - x2_lo).powi(2) + (y1_hi - y2_lo).powi(2) + (z1_hi - z2_lo).powi(2)).sqrt(),
        );

    if d_direct <= d_swapped {
        Some((d_direct, false))
    } else {
        Some((d_swapped, true))
    }
}

/// Check if two blocks have overlapping AABBs.
fn aabb_overlap(b1: &plot3d::Block, b2: &plot3d::Block, tol: f64) -> bool {
    let aabb = |b: &plot3d::Block| -> [f64; 6] {
        let mut mn = [f64::INFINITY; 3];
        let mut mx = [f64::NEG_INFINITY; 3];
        for &x in &b.x { mn[0] = mn[0].min(x); mx[0] = mx[0].max(x); }
        for &y in &b.y { mn[1] = mn[1].min(y); mx[1] = mx[1].max(y); }
        for &z in &b.z { mn[2] = mn[2].min(z); mx[2] = mx[2].max(z); }
        [mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]
    };
    let a = aabb(b1);
    let b = aabb(b2);
    a[1] + tol >= b[0] && b[1] + tol >= a[0]
        && a[3] + tol >= b[2] && b[3] + tol >= a[2]
        && a[5] + tol >= b[4] && b[5] + tol >= a[4]
}

// ---------------------------------------------------------------------------
// Main diagnostic test
// ---------------------------------------------------------------------------

#[test]
fn subset_diagnostics() {
    println!("\n=== SUBSET DIAGNOSTICS ===\n");

    // ── Step 1: Load index remap ──
    println!("Loading index remap from {}...", SUBSET_INDICES_FILE);
    let indices_str = std::fs::read_to_string(SUBSET_INDICES_FILE)
        .expect("Indices file not found. Run test_connectivity_debug first.");
    let original_indices: Vec<usize> =
        serde_json::from_str(&indices_str).expect("Failed to parse indices JSON");
    println!("  Original block indices: {}", original_indices.len());

    // Build remap: original_index -> subset_index
    let remap: HashMap<usize, usize> = original_indices
        .iter()
        .enumerate()
        .map(|(new, &orig)| (orig, new))
        .collect();

    // ── Step 2: Load JSON ──
    println!("Loading JSON...");
    let json_str = std::fs::read_to_string(JSON_FILE).expect("Failed to read JSON file");
    let json_data: GridProConnectivity =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");
    println!("  JSON face_matches: {}", json_data.face_matches.len());

    // Remap JSON entries to subset indices
    let remapped_json: Vec<JsonFaceMatch> = json_data
        .face_matches
        .iter()
        .filter(|jm| {
            remap.contains_key(&jm.block1.block_index)
                && remap.contains_key(&jm.block2.block_index)
        })
        .map(|jm| {
            let mut r = jm.clone();
            r.block1.block_index = remap[&jm.block1.block_index];
            r.block2.block_index = remap[&jm.block2.block_index];
            r
        })
        .collect();
    println!("  Remapped JSON entries: {}", remapped_json.len());

    // ── Step 3: Load subset mesh ──
    println!("Loading subset mesh from {}...", SUBSET_FILE);
    let blocks =
        read_plot3d_binary(SUBSET_FILE, BinaryFormat::Raw, FloatPrecision::F64, Endian::Little)
            .expect("Failed to read subset mesh");
    println!("  Subset blocks: {}", blocks.len());

    let gcd = plot3d::utils::compute_min_gcd(&blocks);
    println!("  GCD: {}", gcd);

    // ── Step 4: Run connectivity_fast ──
    println!("Running connectivity_fast on subset...");
    let (computed_matches, _outer_faces) = connectivity_fast(&blocks);
    println!("  Computed face_matches: {}", computed_matches.len());

    // ── Step 5: Compare ──
    let json_keys = build_json_keys(&remapped_json);
    let computed_keys = build_computed_keys(&computed_matches);

    let missing: HashSet<_> = json_keys.difference(&computed_keys).cloned().collect();
    let extra: HashSet<_> = computed_keys.difference(&json_keys).cloned().collect();
    let common = json_keys.intersection(&computed_keys).count();

    println!("\n=== COMPARISON ===");
    println!("  JSON unique keys:    {}", json_keys.len());
    println!("  Computed unique keys: {}", computed_keys.len());
    println!("  Exact matches:       {}", common);
    println!("  Missing from computed: {}", missing.len());
    println!("  Extra in computed:    {}", extra.len());

    // ── Step 6: Classify missing matches by corner distance ──
    let missing_json: Vec<&JsonFaceMatch> = remapped_json
        .iter()
        .filter(|jm| {
            let a = NormalizedFaceRecord::from_json_block(&jm.block1);
            let b = NormalizedFaceRecord::from_json_block(&jm.block2);
            missing.contains(&MatchKey::new(a, b))
        })
        .collect();

    let mut cat_exact = Vec::new(); // d < 1e-6
    let mut cat_near = Vec::new();  // 1e-6 <= d < 1e-2
    let mut cat_far = Vec::new();   // d >= 1e-2
    let mut cat_oob = 0usize;       // out of bounds

    for jm in &missing_json {
        match min_corner_distance(jm, &blocks) {
            Some((d, swapped)) => {
                if d < 1e-6 {
                    cat_exact.push((*jm, d, swapped));
                } else if d < 1e-2 {
                    cat_near.push((*jm, d, swapped));
                } else {
                    cat_far.push((*jm, d, swapped));
                }
            }
            None => cat_oob += 1,
        }
    }

    println!("\n=== CLASSIFICATION ===");
    println!("  Exact (d < 1e-6):   {} -- faces coincide, algorithm bug", cat_exact.len());
    println!("  Near (1e-6..1e-2):  {} -- tolerance issue", cat_near.len());
    println!("  Far (d >= 1e-2):    {} -- likely periodic", cat_far.len());
    println!("  Out of bounds:      {}", cat_oob);

    // ── Step 7: Comprehensive exact-match analysis ──
    let reduced = reduce_blocks(&blocks, gcd);

    // 7a: Build lookup structures
    let computed_block_pairs: HashSet<(usize, usize)> = computed_matches
        .iter()
        .map(|fm| {
            let a = fm.block1.block_index.min(fm.block2.block_index);
            let b = fm.block1.block_index.max(fm.block2.block_index);
            (a, b)
        })
        .collect();

    // Track which face sides of each block are involved in computed matches
    // Key: (block_index, axis 0=I/1=J/2=K, constant_index_value)
    // Value: list of partner block indices
    let mut matched_sides: HashMap<(usize, u8, usize), Vec<usize>> = HashMap::new();
    for fm in &computed_matches {
        let n1 = NormalizedFaceRecord::from_face_record(&fm.block1);
        let n2 = NormalizedFaceRecord::from_face_record(&fm.block2);

        // Determine face side for block1
        if n1.imin == n1.imax {
            matched_sides
                .entry((n1.block_index, 0, n1.imin))
                .or_default()
                .push(n2.block_index);
        } else if n1.jmin == n1.jmax {
            matched_sides
                .entry((n1.block_index, 1, n1.jmin))
                .or_default()
                .push(n2.block_index);
        } else if n1.kmin == n1.kmax {
            matched_sides
                .entry((n1.block_index, 2, n1.kmin))
                .or_default()
                .push(n2.block_index);
        }
        // Same for block2
        if n2.imin == n2.imax {
            matched_sides
                .entry((n2.block_index, 0, n2.imin))
                .or_default()
                .push(n1.block_index);
        } else if n2.jmin == n2.jmax {
            matched_sides
                .entry((n2.block_index, 1, n2.jmin))
                .or_default()
                .push(n1.block_index);
        } else if n2.kmin == n2.kmax {
            matched_sides
                .entry((n2.block_index, 2, n2.kmin))
                .or_default()
                .push(n1.block_index);
        }
    }

    // Check degenerate faces (blocks with < 6 outer faces)
    let mut degenerate_blocks: HashSet<usize> = HashSet::new();
    let mut degenerate_block_face_counts: HashMap<usize, (usize, usize)> = HashMap::new();
    {
        let exact_block_indices: HashSet<usize> = cat_exact
            .iter()
            .flat_map(|(jm, _, _)| {
                vec![jm.block1.block_index, jm.block2.block_index]
            })
            .collect();
        for &bi in &exact_block_indices {
            if bi < reduced.len() {
                let (faces, internal) = get_outer_faces(&reduced[bi]);
                if !internal.is_empty() {
                    degenerate_blocks.insert(bi);
                }
                degenerate_block_face_counts.insert(bi, (faces.len(), internal.len()));
            }
        }
    }

    // 7b: Fast pass over ALL exact failures
    println!("\n=== EXACT-MATCH ANALYSIS (all {}) ===", cat_exact.len());

    let mut pair_has_match = 0usize;
    let mut pair_no_match = 0usize;
    let mut pair_no_match_aabb_fail = 0usize;
    let mut pair_no_match_aabb_ok = 0usize;
    let mut face_consumed_by_other = 0usize;
    let mut involves_degenerate = 0usize;
    let mut gcd_misaligned = 0usize;

    // For "pair has match" cases: track how many have different face ranges
    let mut pair_match_diff_ranges = 0usize;

    for (jm, _d, _) in &cat_exact {
        let b1 = jm.block1.block_index;
        let b2 = jm.block2.block_index;
        let pair = (b1.min(b2), b1.max(b2));

        // Check GCD alignment
        let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
        let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);
        let gcd_ok = n1.imin % gcd == 0
            && n1.jmin % gcd == 0
            && n1.kmin % gcd == 0
            && n1.imax % gcd == 0
            && n1.jmax % gcd == 0
            && n1.kmax % gcd == 0
            && n2.imin % gcd == 0
            && n2.jmin % gcd == 0
            && n2.kmin % gcd == 0
            && n2.imax % gcd == 0
            && n2.jmax % gcd == 0
            && n2.kmax % gcd == 0;
        if !gcd_ok {
            gcd_misaligned += 1;
        }

        // Check degenerate faces
        if degenerate_blocks.contains(&b1) || degenerate_blocks.contains(&b2) {
            involves_degenerate += 1;
        }

        if computed_block_pairs.contains(&pair) {
            pair_has_match += 1;
            // Check if the computed match has different face ranges
            let json_key = MatchKey::new(n1.clone(), n2.clone());
            if !computed_keys.contains(&json_key) {
                pair_match_diff_ranges += 1;
            }
        } else {
            pair_no_match += 1;
            if b1 < reduced.len() && b2 < reduced.len() {
                if !aabb_overlap(&reduced[b1], &reduced[b2], 1e-6) {
                    pair_no_match_aabb_fail += 1;
                } else {
                    pair_no_match_aabb_ok += 1;

                    // Check face consumption: is either block's face side matched with another block?
                    let side1 = if n1.imin == n1.imax {
                        Some((0u8, n1.imin))
                    } else if n1.jmin == n1.jmax {
                        Some((1u8, n1.jmin))
                    } else if n1.kmin == n1.kmax {
                        Some((2u8, n1.kmin))
                    } else {
                        None
                    };
                    let side2 = if n2.imin == n2.imax {
                        Some((0u8, n2.imin))
                    } else if n2.jmin == n2.jmax {
                        Some((1u8, n2.jmin))
                    } else if n2.kmin == n2.kmax {
                        Some((2u8, n2.kmin))
                    } else {
                        None
                    };

                    let consumed1 = side1.map_or(false, |(axis, val)| {
                        matched_sides.contains_key(&(b1, axis, val))
                    });
                    let consumed2 = side2.map_or(false, |(axis, val)| {
                        matched_sides.contains_key(&(b2, axis, val))
                    });

                    if consumed1 || consumed2 {
                        face_consumed_by_other += 1;
                    }
                }
            }
        }
    }

    println!("  Block pair HAS other computed match: {}", pair_has_match);
    println!(
        "    - But different face ranges:       {}",
        pair_match_diff_ranges
    );
    println!(
        "  Block pair has NO computed match:    {}",
        pair_no_match
    );
    println!(
        "    - AABB fail (reduced):             {}",
        pair_no_match_aabb_fail
    );
    println!(
        "    - AABB ok, face side consumed:     {}",
        face_consumed_by_other
    );
    println!(
        "    - AABB ok, unknown cause:          {}",
        pair_no_match_aabb_ok.saturating_sub(face_consumed_by_other)
    );
    println!("  GCD-misaligned face ranges:         {}", gcd_misaligned);
    println!(
        "  Involves degenerate block:          {}",
        involves_degenerate
    );
    println!(
        "  Degenerate blocks total:            {} / {}",
        degenerate_blocks.len(),
        cat_exact
            .iter()
            .flat_map(|(jm, _, _)| vec![jm.block1.block_index, jm.block2.block_index])
            .collect::<HashSet<_>>()
            .len()
    );

    // 7c: Print sample of face consumption details
    println!("\n=== FACE CONSUMPTION SAMPLES ===");
    let mut consumption_shown = 0;
    for (jm, d, _) in &cat_exact {
        if consumption_shown >= 10 {
            break;
        }
        let b1 = jm.block1.block_index;
        let b2 = jm.block2.block_index;
        let pair = (b1.min(b2), b1.max(b2));
        if computed_block_pairs.contains(&pair) {
            continue;
        }
        if b1 >= reduced.len() || b2 >= reduced.len() {
            continue;
        }
        if !aabb_overlap(&reduced[b1], &reduced[b2], 1e-6) {
            continue;
        }

        let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
        let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);
        let side1 = if n1.imin == n1.imax {
            Some((0u8, n1.imin))
        } else if n1.jmin == n1.jmax {
            Some((1u8, n1.jmin))
        } else if n1.kmin == n1.kmax {
            Some((2u8, n1.kmin))
        } else {
            None
        };
        let side2 = if n2.imin == n2.imax {
            Some((0u8, n2.imin))
        } else if n2.jmin == n2.jmax {
            Some((1u8, n2.jmin))
        } else if n2.kmin == n2.kmax {
            Some((2u8, n2.kmin))
        } else {
            None
        };

        let axis_name = |a: u8| match a {
            0 => "I",
            1 => "J",
            2 => "K",
            _ => "?",
        };
        let partners1 = side1.and_then(|(axis, val)| {
            matched_sides.get(&(b1, axis, val)).map(|p| (axis, val, p.clone()))
        });
        let partners2 = side2.and_then(|(axis, val)| {
            matched_sides.get(&(b2, axis, val)).map(|p| (axis, val, p.clone()))
        });

        println!(
            "  block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}] d={:.2e}",
            b1,
            n1.imin,
            n1.jmin,
            n1.kmin,
            n1.imax,
            n1.jmax,
            n1.kmax,
            b2,
            n2.imin,
            n2.jmin,
            n2.kmin,
            n2.imax,
            n2.jmax,
            n2.kmax,
            d
        );
        if let Some((axis, val, partners)) = &partners1 {
            println!(
                "    block{} {}={} matched with blocks: {:?}",
                b1,
                axis_name(*axis),
                val,
                &partners[..partners.len().min(5)]
            );
        } else {
            println!("    block{} face side: NOT matched with any block", b1);
        }
        if let Some((axis, val, partners)) = &partners2 {
            println!(
                "    block{} {}={} matched with blocks: {:?}",
                b2,
                axis_name(*axis),
                val,
                &partners[..partners.len().min(5)]
            );
        } else {
            println!("    block{} face side: NOT matched with any block", b2);
        }

        // Check face counts
        if let Some(&(nfaces, ninternal)) = degenerate_block_face_counts.get(&b1) {
            if ninternal > 0 {
                println!(
                    "    block{}: {} outer faces, {} internal pairs (DEGENERATE)",
                    b1, nfaces, ninternal
                );
            }
        }
        if let Some(&(nfaces, ninternal)) = degenerate_block_face_counts.get(&b2) {
            if ninternal > 0 {
                println!(
                    "    block{}: {} outer faces, {} internal pairs (DEGENERATE)",
                    b2, nfaces, ninternal
                );
            }
        }

        consumption_shown += 1;
    }

    // 7d: Sampled face-match check on reduced blocks (first 50)
    println!("\n=== FACE-MATCH CHECK (sampled 50) ===");
    let mut aabb_fail = 0usize;
    let mut full_match_exists = 0usize;
    let mut partial_match_exists = 0usize;
    let mut no_match_at_all = 0usize;

    for (jm, d, _swapped) in cat_exact.iter().take(50) {
        let b1 = jm.block1.block_index;
        let b2 = jm.block2.block_index;

        if b1 >= reduced.len() || b2 >= reduced.len() {
            continue;
        }
        if !aabb_overlap(&reduced[b1], &reduced[b2], 1e-6) {
            aabb_fail += 1;
            continue;
        }

        let (faces1, _) = get_outer_faces(&reduced[b1]);
        let (faces2, _) = get_outer_faces(&reduced[b2]);

        let mut found_full = false;
        for f1 in &faces1 {
            for f2 in &faces2 {
                if full_face_match(f1, f2, 1e-6).is_some() {
                    found_full = true;
                    break;
                }
            }
            if found_full {
                break;
            }
        }
        if found_full {
            full_match_exists += 1;
            continue;
        }

        let mut found_partial = false;
        for f1 in &faces1 {
            for f2 in &faces2 {
                let (pts, _, _) =
                    get_face_intersection(f1, f2, &reduced[b1], &reduced[b2], 1e-6);
                if !pts.is_empty() {
                    found_partial = true;
                    break;
                }
            }
            if found_partial {
                break;
            }
        }
        if found_partial {
            partial_match_exists += 1;
        } else {
            no_match_at_all += 1;
        }
    }

    println!("  AABB fail:           {}", aabb_fail);
    println!("  Full match (lost):   {}", full_match_exists);
    println!("  Partial match found: {}", partial_match_exists);
    println!("  No match at all:     {}", no_match_at_all);

    // 7e: Investigate "different face ranges" — are computed matches sub-faces of JSON?
    println!("\n=== DIFFERENT FACE RANGES ANALYSIS ===");
    {
        let mut total_covered = 0usize;
        let mut total_over_split = 0usize;
        let mut shown = 0;

        for (jm, _d, _) in &cat_exact {
            let b1 = jm.block1.block_index;
            let b2 = jm.block2.block_index;
            let pair = (b1.min(b2), b1.max(b2));
            let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
            let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);

            if !computed_block_pairs.contains(&pair) {
                continue;
            }
            // Check if JSON match is already found
            let json_key = MatchKey::new(n1.clone(), n2.clone());
            if computed_keys.contains(&json_key) {
                continue;
            }

            // Get all computed matches for this block pair
            let pair_matches: Vec<_> = computed_matches
                .iter()
                .filter(|fm| {
                    let a = fm.block1.block_index.min(fm.block2.block_index);
                    let b = fm.block1.block_index.max(fm.block2.block_index);
                    (a, b) == pair
                })
                .collect();

            // Check if any computed match is a sub-face of the JSON match
            // (same constant axis, range contained within JSON range)
            let mut contains_sub = false;
            for fm in &pair_matches {
                let cn1 = NormalizedFaceRecord::from_face_record(&fm.block1);
                let cn2 = NormalizedFaceRecord::from_face_record(&fm.block2);

                // Match block1 side
                let (jn, cn) = if cn1.block_index == b1 {
                    (&n1, &cn1)
                } else {
                    (&n2, &cn2)
                };

                // Check containment: computed range within JSON range
                if cn.imin >= jn.imin
                    && cn.imax <= jn.imax
                    && cn.jmin >= jn.jmin
                    && cn.jmax <= jn.jmax
                    && cn.kmin >= jn.kmin
                    && cn.kmax <= jn.kmax
                {
                    contains_sub = true;
                }
            }

            if contains_sub {
                total_over_split += 1;
            }
            // Count if pair has multiple computed matches
            if pair_matches.len() > 1 {
                total_covered += 1;
            }

            if shown < 5 {
                println!(
                    "  block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                    b1, n1.imin, n1.jmin, n1.kmin, n1.imax, n1.jmax, n1.kmax,
                    b2, n2.imin, n2.jmin, n2.kmin, n2.imax, n2.jmax, n2.kmax,
                );
                println!(
                    "    Computed matches for pair: {} total",
                    pair_matches.len()
                );
                for (midx, fm) in pair_matches.iter().enumerate().take(5) {
                    let cn1 = NormalizedFaceRecord::from_face_record(&fm.block1);
                    let cn2 = NormalizedFaceRecord::from_face_record(&fm.block2);
                    println!(
                        "    [{}] block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                        midx,
                        cn1.block_index, cn1.imin, cn1.jmin, cn1.kmin, cn1.imax, cn1.jmax, cn1.kmax,
                        cn2.block_index, cn2.imin, cn2.jmin, cn2.kmin, cn2.imax, cn2.jmax, cn2.kmax,
                    );
                }
                shown += 1;
            }
        }
        println!("  Total 'different ranges' cases: 429 (expected)");
        println!(
            "  Contains computed sub-face:     {}",
            total_over_split
        );
        println!(
            "  Pair has multiple matches:      {}",
            total_covered
        );
    }

    // ── Step 8: Far matches (periodic analysis) ──
    println!("\n=== FAR MATCHES (d >= 1e-2) - distance histogram ===");
    let mut buckets = [0usize; 5];
    for (_, d, _) in &cat_far {
        if *d < 0.1 { buckets[0] += 1; }
        else if *d < 0.5 { buckets[1] += 1; }
        else if *d < 1.0 { buckets[2] += 1; }
        else if *d < 5.0 { buckets[3] += 1; }
        else { buckets[4] += 1; }
    }
    println!("  0.01 - 0.1:  {}", buckets[0]);
    println!("  0.1  - 0.5:  {}", buckets[1]);
    println!("  0.5  - 1.0:  {}", buckets[2]);
    println!("  1.0  - 5.0:  {}", buckets[3]);
    println!("  >= 5.0:      {}", buckets[4]);

    for (idx, (jm, d, _)) in cat_far.iter().enumerate().take(5) {
        let b1 = jm.block1.block_index;
        let b2 = jm.block2.block_index;
        let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
        let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);
        if n1.imax < blocks[b1].imax && n1.jmax < blocks[b1].jmax && n1.kmax < blocks[b1].kmax
            && n2.imax < blocks[b2].imax && n2.jmax < blocks[b2].jmax && n2.kmax < blocks[b2].kmax
        {
            let (x1, y1, z1) = blocks[b1].xyz(n1.imin, n1.jmin, n1.kmin);
            let (x2, y2, z2) = blocks[b2].xyz(n2.imin, n2.jmin, n2.kmin);
            println!(
                "  [{}] block{}<->block{}: d={:.4}, c1=({:.4},{:.4},{:.4}), c2=({:.4},{:.4},{:.4})",
                idx, b1, b2, d, x1, y1, z1, x2, y2, z2
            );
        }
    }

    // ── Step 9: Near matches ──
    println!("\n=== NEAR MATCHES (1e-6..1e-2) ===");
    let mut at_1e4 = 0;
    let mut at_1e3 = 0;
    for (_, d, _) in &cat_near {
        if *d < 1e-4 { at_1e4 += 1; }
        if *d < 1e-3 { at_1e3 += 1; }
    }
    println!("  Would match at tol=1e-4: {}", at_1e4);
    println!("  Would match at tol=1e-3: {}", at_1e3);
    println!("  Total near:              {}", cat_near.len());

    // ── Step 10: Block pair analysis ──
    println!("\n=== BLOCK PAIR ANALYSIS ===");
    let json_pairs = group_by_block_pair(&json_keys);
    let comp_pairs = group_by_block_pair(&computed_keys);
    let jp_set: HashSet<_> = json_pairs.keys().cloned().collect();
    let cp_set: HashSet<_> = comp_pairs.keys().cloned().collect();

    let pairs_only_json: Vec<_> = jp_set.difference(&cp_set).collect();
    let pairs_only_comp: Vec<_> = cp_set.difference(&jp_set).collect();
    let pairs_both = jp_set.intersection(&cp_set).count();

    let mut diff_faces = 0;
    for pair in jp_set.intersection(&cp_set) {
        let jf: HashSet<_> = json_pairs[pair].iter().collect();
        let cf: HashSet<_> = comp_pairs[pair].iter().collect();
        if jf != cf { diff_faces += 1; }
    }

    println!("  Block pairs in JSON:     {}", jp_set.len());
    println!("  Block pairs in computed:  {}", cp_set.len());
    println!("  In both:                 {}", pairs_both);
    println!("  Only in JSON:            {}", pairs_only_json.len());
    println!("  Only in computed:        {}", pairs_only_comp.len());
    println!("  In both, diff faces:     {}", diff_faces);

    // ── Summary ──
    println!("\n=== FINAL SUMMARY ===");
    println!("  Subset blocks: {}", blocks.len());
    println!("  Remapped JSON: {}", remapped_json.len());
    println!("  Computed:      {}", computed_matches.len());
    println!("  Exact matches: {}", common);
    println!("  Missing:       {}", missing.len());
    println!("  Extra:         {}", extra.len());
    println!("  --- Failure breakdown ---");
    println!("  Exact (d<1e-6):  {}", cat_exact.len());
    println!("  Near:            {}", cat_near.len());
    println!("  Far (periodic?): {}", cat_far.len());
    println!("  Out of bounds:   {}", cat_oob);
    println!();
}
