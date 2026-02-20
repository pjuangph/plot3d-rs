//! Debug test comparing plot3d-rs connectivity against GridPro reference JSON.
//!
//! Usage (release build required for large meshes):
//!   cargo test --release --test test_connectivity_debug -- --nocapture

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use plot3d::{
    connectivity_fast, full_face_match, get_face_intersection, get_outer_faces, read_plot3d_binary,
    reduce_blocks, write_plot3d, BinaryFormat, Endian, FaceMatch, FaceRecord, FloatPrecision,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";
const JSON_FILE: &str =
    "/Users/pjuangph/Documents/GitHub/plot3d_troubleshoot/gridpro_connectivity.json";
const SUBSET_OUTPUT: &str = "/tmp/connectivity_debug_subset.p3d";
const SUBSET_INDICES: &str = "/tmp/connectivity_debug_subset_indices.json";

// ---------------------------------------------------------------------------
// JSON deserialization
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
// Normalized face record (orientation-agnostic, imin<=imax etc.)
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
// Helper: build key sets
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

// ---------------------------------------------------------------------------
// Helper: group by block pair
// ---------------------------------------------------------------------------

fn group_by_block_pair(keys: &HashSet<MatchKey>) -> HashMap<(usize, usize), Vec<MatchKey>> {
    let mut map: HashMap<(usize, usize), Vec<MatchKey>> = HashMap::new();
    for k in keys {
        map.entry(k.block_pair()).or_default().push(k.clone());
    }
    map
}

// ---------------------------------------------------------------------------
// Helper: collect mismatch block indices
// ---------------------------------------------------------------------------

fn collect_mismatch_block_indices(missing: &[&MatchKey], extra: &[&MatchKey]) -> Vec<usize> {
    let mut indices: HashSet<usize> = HashSet::new();
    for key in missing {
        indices.insert(key.face_a.block_index);
        indices.insert(key.face_b.block_index);
    }
    for key in extra {
        indices.insert(key.face_a.block_index);
        indices.insert(key.face_b.block_index);
    }
    let mut sorted: Vec<usize> = indices.into_iter().collect();
    sorted.sort();
    sorted
}

// ---------------------------------------------------------------------------
// Helper: extract blocks and build index remap
// ---------------------------------------------------------------------------

fn extract_blocks(all_blocks: &[plot3d::Block], indices: &[usize]) -> Vec<plot3d::Block> {
    indices.iter().map(|&i| all_blocks[i].clone()).collect()
}

fn build_index_remap(original_indices: &[usize]) -> HashMap<usize, usize> {
    original_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: diagnose a missing match
// ---------------------------------------------------------------------------

fn diagnose_missing_match(
    idx: usize,
    jm: &JsonFaceMatch,
    blocks: &[plot3d::Block],
    computed_matches: &[FaceMatch],
    gcd: usize,
) {
    let b1_idx = jm.block1.block_index;
    let b2_idx = jm.block2.block_index;

    if b1_idx >= blocks.len() || b2_idx >= blocks.len() {
        println!(
            "  [{}] block {} or {} out of range (nblocks={})",
            idx,
            b1_idx,
            b2_idx,
            blocks.len()
        );
        return;
    }

    let block1 = &blocks[b1_idx];
    let block2 = &blocks[b2_idx];

    let norm1 = NormalizedFaceRecord::from_json_block(&jm.block1);
    let norm2 = NormalizedFaceRecord::from_json_block(&jm.block2);

    println!(
        "  [{}] block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
        idx,
        b1_idx,
        norm1.imin,
        norm1.jmin,
        norm1.kmin,
        norm1.imax,
        norm1.jmax,
        norm1.kmax,
        b2_idx,
        norm2.imin,
        norm2.jmin,
        norm2.kmin,
        norm2.imax,
        norm2.jmax,
        norm2.kmax,
    );

    // Block dimensions
    println!(
        "       block{} dims: {}x{}x{}, block{} dims: {}x{}x{}",
        b1_idx, block1.imax, block1.jmax, block1.kmax, b2_idx, block2.imax, block2.jmax,
        block2.kmax,
    );

    // Validate indices are within bounds
    let b1_ok = norm1.imax < block1.imax && norm1.jmax < block1.jmax && norm1.kmax < block1.kmax;
    let b2_ok = norm2.imax < block2.imax && norm2.jmax < block2.jmax && norm2.kmax < block2.kmax;

    if !b1_ok || !b2_ok {
        println!("       WARNING: indices out of block bounds! b1_ok={}, b2_ok={}", b1_ok, b2_ok);
        return;
    }

    // Corner coordinates
    let (x1_lo, y1_lo, z1_lo) = block1.xyz(norm1.imin, norm1.jmin, norm1.kmin);
    let (x1_hi, y1_hi, z1_hi) = block1.xyz(norm1.imax, norm1.jmax, norm1.kmax);
    let (x2_lo, y2_lo, z2_lo) = block2.xyz(norm2.imin, norm2.jmin, norm2.kmin);
    let (x2_hi, y2_hi, z2_hi) = block2.xyz(norm2.imax, norm2.jmax, norm2.kmax);

    // Try both corner pairings: direct and swapped
    let d_lo_lo = ((x1_lo - x2_lo).powi(2) + (y1_lo - y2_lo).powi(2) + (z1_lo - z2_lo).powi(2))
        .sqrt();
    let d_hi_hi = ((x1_hi - x2_hi).powi(2) + (y1_hi - y2_hi).powi(2) + (z1_hi - z2_hi).powi(2))
        .sqrt();
    let d_lo_hi = ((x1_lo - x2_hi).powi(2) + (y1_lo - y2_hi).powi(2) + (z1_lo - z2_hi).powi(2))
        .sqrt();
    let d_hi_lo = ((x1_hi - x2_lo).powi(2) + (y1_hi - y2_lo).powi(2) + (z1_hi - z2_lo).powi(2))
        .sqrt();

    let best_direct = d_lo_lo.max(d_hi_hi);
    let best_swapped = d_lo_hi.max(d_hi_lo);
    let corners_match = best_direct < 1e-6 || best_swapped < 1e-6;

    println!(
        "       corners: direct(lo-lo={:.2e}, hi-hi={:.2e}), swapped(lo-hi={:.2e}, hi-lo={:.2e}) -> {}",
        d_lo_lo, d_hi_hi, d_lo_hi, d_hi_lo,
        if corners_match { "MATCH" } else { "NO MATCH" }
    );

    // GCD divisibility check
    let all_indices = [
        norm1.imin, norm1.jmin, norm1.kmin, norm1.imax, norm1.jmax, norm1.kmax, norm2.imin,
        norm2.jmin, norm2.kmin, norm2.imax, norm2.jmax, norm2.kmax,
    ];
    let non_divisible: Vec<_> = all_indices
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0 && v % gcd != 0)
        .collect();
    if !non_divisible.is_empty() {
        let labels = [
            "b1.imin", "b1.jmin", "b1.kmin", "b1.imax", "b1.jmax", "b1.kmax", "b2.imin",
            "b2.jmin", "b2.kmin", "b2.imax", "b2.jmax", "b2.kmax",
        ];
        print!("       GCD={} NOT divisible:", gcd);
        for (i, &v) in &non_divisible {
            print!(" {}={}(mod {}={})", labels[*i], v, gcd, v % gcd);
        }
        println!();
    }

    // Find all computed matches for this block pair
    let pair_matches: Vec<_> = computed_matches
        .iter()
        .filter(|fm| {
            (fm.block1.block_index == b1_idx && fm.block2.block_index == b2_idx)
                || (fm.block1.block_index == b2_idx && fm.block2.block_index == b1_idx)
        })
        .collect();

    if pair_matches.is_empty() {
        println!("       No computed matches for block pair ({}, {})", b1_idx, b2_idx);
    } else {
        println!(
            "       {} computed match(es) for this block pair:",
            pair_matches.len()
        );
        for (i, fm) in pair_matches.iter().enumerate() {
            println!(
                "         [{}] block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                i,
                fm.block1.block_index,
                fm.block1.il,
                fm.block1.jl,
                fm.block1.kl,
                fm.block1.ih,
                fm.block1.jh,
                fm.block1.kh,
                fm.block2.block_index,
                fm.block2.il,
                fm.block2.jl,
                fm.block2.kl,
                fm.block2.ih,
                fm.block2.jh,
                fm.block2.kh,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn connectivity_debug() {
    // ── Step 1: Read JSON ────────────────────────────────────────────────
    println!("\n=== Step 1: Reading JSON reference ===");
    let json_str = std::fs::read_to_string(JSON_FILE).expect("Failed to read JSON file");
    let json_data: GridProConnectivity =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");
    println!("  JSON face_matches: {}", json_data.face_matches.len());

    // ── Step 2: Read mesh ────────────────────────────────────────────────
    println!("\n=== Step 2: Reading mesh ===");
    let blocks = read_plot3d_binary(MESH_FILE, BinaryFormat::Raw, FloatPrecision::F64, Endian::Little)
        .expect("Failed to read mesh file");
    println!("  Blocks loaded: {}", blocks.len());

    // Print block dimension stats
    let total_points: usize = blocks.iter().map(|b| b.imax * b.jmax * b.kmax).sum();
    println!("  Total grid points: {}", total_points);

    // Compute GCD
    let gcd = plot3d::utils::compute_min_gcd(&blocks);
    println!("  Global GCD: {}", gcd);

    // ── Step 3: Run connectivity_fast ────────────────────────────────────
    println!("\n=== Step 3: Running connectivity_fast ===");
    let (computed_matches, outer_faces) = connectivity_fast(&blocks);
    println!("  Computed face_matches: {}", computed_matches.len());
    println!("  Outer faces: {}", outer_faces.len());

    // ── Step 4: Build comparison keys ────────────────────────────────────
    println!("\n=== Step 4: Exact-key comparison ===");
    let json_keys = build_json_keys(&json_data.face_matches);
    let computed_keys = build_computed_keys(&computed_matches);

    // Deduplicated counts (HashSet removes duplicates)
    println!(
        "  JSON unique keys:     {} (from {} entries)",
        json_keys.len(),
        json_data.face_matches.len()
    );
    println!("  Computed unique keys: {}", computed_keys.len());

    let missing: Vec<_> = json_keys.difference(&computed_keys).collect();
    let extra: Vec<_> = computed_keys.difference(&json_keys).collect();
    let common_count = json_keys.intersection(&computed_keys).count();

    println!("  Exact matches:          {}", common_count);
    println!(
        "  Missing from computed:  {} ({:.1}% of JSON)",
        missing.len(),
        missing.len() as f64 / json_keys.len() as f64 * 100.0
    );
    println!(
        "  Extra in computed:      {} ({:.1}% of computed)",
        extra.len(),
        if computed_keys.is_empty() {
            0.0
        } else {
            extra.len() as f64 / computed_keys.len() as f64 * 100.0
        }
    );

    // ── Step 5: Block-pair level analysis ────────────────────────────────
    println!("\n=== Step 5: Block-pair level analysis ===");
    let json_pairs = group_by_block_pair(&json_keys);
    let computed_pairs = group_by_block_pair(&computed_keys);

    let json_pair_set: HashSet<_> = json_pairs.keys().cloned().collect();
    let computed_pair_set: HashSet<_> = computed_pairs.keys().cloned().collect();

    let pairs_only_json: Vec<_> = json_pair_set.difference(&computed_pair_set).collect();
    let pairs_only_computed: Vec<_> = computed_pair_set.difference(&json_pair_set).collect();
    let pairs_both = json_pair_set.intersection(&computed_pair_set).count();

    println!("  Unique block pairs in JSON:     {}", json_pair_set.len());
    println!("  Unique block pairs in computed:  {}", computed_pair_set.len());
    println!("  Block pairs in both:             {}", pairs_both);
    println!(
        "  Block pairs only in JSON:        {}",
        pairs_only_json.len()
    );
    println!(
        "  Block pairs only in computed:    {}",
        pairs_only_computed.len()
    );

    // Among block pairs present in both, how many have different face indices?
    let mut pairs_with_face_diff = 0usize;
    for pair in json_pair_set.intersection(&computed_pair_set) {
        let json_faces = &json_pairs[pair];
        let comp_faces = &computed_pairs[pair];
        let json_face_set: HashSet<_> = json_faces.iter().collect();
        let comp_face_set: HashSet<_> = comp_faces.iter().collect();
        if json_face_set != comp_face_set {
            pairs_with_face_diff += 1;
        }
    }
    println!(
        "  Block pairs in both but face indices differ: {}",
        pairs_with_face_diff
    );

    // ── Step 6: Print block pairs only in JSON (first 30) ────────────────
    if !pairs_only_json.is_empty() {
        println!(
            "\n=== Step 6a: Block pairs only in JSON (first 30 of {}) ===",
            pairs_only_json.len()
        );
        for (i, pair) in pairs_only_json.iter().take(30).enumerate() {
            let entries = &json_pairs[pair];
            println!("  [{}] blocks ({}, {}): {} face match(es)", i, pair.0, pair.1, entries.len());
            for key in entries {
                println!(
                    "       face_a[{},{},{}->{},{},{}] <-> face_b[{},{},{}->{},{},{}]",
                    key.face_a.imin,
                    key.face_a.jmin,
                    key.face_a.kmin,
                    key.face_a.imax,
                    key.face_a.jmax,
                    key.face_a.kmax,
                    key.face_b.imin,
                    key.face_b.jmin,
                    key.face_b.kmin,
                    key.face_b.imax,
                    key.face_b.jmax,
                    key.face_b.kmax,
                );
            }
        }
    }

    // ── Step 6b: Print block pairs only in computed (first 30) ───────────
    if !pairs_only_computed.is_empty() {
        println!(
            "\n=== Step 6b: Block pairs only in computed (first 30 of {}) ===",
            pairs_only_computed.len()
        );
        for (i, pair) in pairs_only_computed.iter().take(30).enumerate() {
            let entries = &computed_pairs[pair];
            println!("  [{}] blocks ({}, {}): {} face match(es)", i, pair.0, pair.1, entries.len());
        }
    }

    // ── Step 7: Detailed diagnostics for missing matches ─────────────────
    println!(
        "\n=== Step 7: Detailed diagnostics for missing matches (first 20) ==="
    );
    let missing_set: HashSet<_> = missing.iter().cloned().collect();
    let mut diagnosed = 0usize;
    for jm in &json_data.face_matches {
        let a = NormalizedFaceRecord::from_json_block(&jm.block1);
        let b = NormalizedFaceRecord::from_json_block(&jm.block2);
        let key = MatchKey::new(a, b);
        if missing_set.contains(&key) {
            diagnose_missing_match(diagnosed, jm, &blocks, &computed_matches, gcd);
            diagnosed += 1;
            if diagnosed >= 20 {
                break;
            }
        }
    }

    // ── Step 8: GCD divisibility summary ─────────────────────────────────
    println!("\n=== Step 8: GCD divisibility summary ===");
    let mut gcd_fail_count = 0usize;
    for jm in &json_data.face_matches {
        let norm1 = NormalizedFaceRecord::from_json_block(&jm.block1);
        let norm2 = NormalizedFaceRecord::from_json_block(&jm.block2);
        let indices = [
            norm1.imin, norm1.jmin, norm1.kmin, norm1.imax, norm1.jmax, norm1.kmax, norm2.imin,
            norm2.jmin, norm2.kmin, norm2.imax, norm2.jmax, norm2.kmax,
        ];
        if indices.iter().any(|&v| v > 0 && v % gcd != 0) {
            gcd_fail_count += 1;
        }
    }
    println!(
        "  JSON entries with indices not divisible by GCD={}: {} / {}",
        gcd,
        gcd_fail_count,
        json_data.face_matches.len()
    );

    // ── Step 9: Export subset and re-test ─────────────────────────────────
    println!("\n=== Step 9: Subset export and re-test ===");
    let missing_refs: Vec<&MatchKey> = missing.iter().copied().collect();
    let extra_refs: Vec<&MatchKey> = extra.iter().copied().collect();
    let mismatch_indices = collect_mismatch_block_indices(&missing_refs, &extra_refs);
    println!(
        "  Unique blocks in mismatches: {} / {}",
        mismatch_indices.len(),
        blocks.len()
    );

    // Only export if the subset is manageable (< 50% of total blocks)
    if !mismatch_indices.is_empty() && mismatch_indices.len() <= blocks.len() / 2 {
        let subset = extract_blocks(&blocks, &mismatch_indices);
        let remap = build_index_remap(&mismatch_indices);

        write_plot3d(
            SUBSET_OUTPUT,
            &subset,
            true,
            plot3d::write::BinaryFormat::Raw,
            plot3d::write::FloatPrecision::F64,
            Endian::Little,
        )
        .expect("Failed to write subset");
        println!(
            "  Wrote {} blocks to {}",
            subset.len(),
            SUBSET_OUTPUT
        );

        // Save the original block indices for the subset diagnostic test
        let indices_json = serde_json::to_string(&mismatch_indices).unwrap();
        std::fs::write(SUBSET_INDICES, &indices_json).expect("Failed to write indices");
        println!("  Wrote index remap to {}", SUBSET_INDICES);

        // Re-run connectivity on subset
        println!("  Running connectivity_fast on subset...");
        let (subset_matches, subset_outer) = connectivity_fast(&subset);
        println!("  Subset face_matches: {}", subset_matches.len());
        println!("  Subset outer faces:  {}", subset_outer.len());

        // Remap JSON entries to subset indices and compare
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

        println!(
            "  Remapped JSON entries for subset: {}",
            remapped_json.len()
        );

        let subset_json_keys = build_json_keys(&remapped_json);
        let subset_computed_keys = build_computed_keys(&subset_matches);

        let subset_missing: Vec<_> = subset_json_keys.difference(&subset_computed_keys).collect();
        let subset_extra: Vec<_> = subset_computed_keys.difference(&subset_json_keys).collect();
        let subset_common = subset_json_keys.intersection(&subset_computed_keys).count();

        println!("  Subset exact matches:         {}", subset_common);
        println!("  Subset missing from computed:  {}", subset_missing.len());
        println!("  Subset extra in computed:      {}", subset_extra.len());

        // Block-pair analysis on subset
        let subset_json_pairs = group_by_block_pair(&subset_json_keys);
        let subset_comp_pairs = group_by_block_pair(&subset_computed_keys);
        let sjp_set: HashSet<_> = subset_json_pairs.keys().cloned().collect();
        let scp_set: HashSet<_> = subset_comp_pairs.keys().cloned().collect();
        let sub_pairs_only_json: Vec<_> = sjp_set.difference(&scp_set).collect();
        println!(
            "  Subset block pairs only in JSON: {}",
            sub_pairs_only_json.len()
        );
        println!(
            "  Subset block pairs only in computed: {}",
            scp_set.difference(&sjp_set).count()
        );
    } else if mismatch_indices.is_empty() {
        println!("  No mismatches! All JSON entries matched.");
    } else {
        println!(
            "  Too many mismatch blocks ({} / {}), skipping export",
            mismatch_indices.len(),
            blocks.len()
        );
        println!("  Listing first 30 mismatch block indices:");
        for (i, &idx) in mismatch_indices.iter().take(30).enumerate() {
            let b = &blocks[idx];
            println!(
                "    [{}] block {} dims: {}x{}x{}",
                i, idx, b.imax, b.jmax, b.kmax
            );
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n=== FINAL SUMMARY ===");
    println!("  Total blocks:                {}", blocks.len());
    println!("  JSON face_matches:           {}", json_data.face_matches.len());
    println!("  JSON unique keys:            {}", json_keys.len());
    println!("  Computed face_matches:       {}", computed_matches.len());
    println!("  Computed unique keys:        {}", computed_keys.len());
    println!("  Exact matches:               {}", common_count);
    println!("  Missing from computed:       {}", missing.len());
    println!("  Extra in computed:           {}", extra.len());
    println!(
        "  Block pairs only in JSON:    {}",
        pairs_only_json.len()
    );
    println!(
        "  Block pairs only in computed: {}",
        pairs_only_computed.len()
    );
    println!(
        "  GCD non-divisible entries:    {} (GCD={})",
        gcd_fail_count, gcd
    );
    println!();
}

// ---------------------------------------------------------------------------
// Focused diagnostic: classify ALL missing matches by failure type
// ---------------------------------------------------------------------------

/// Compute the minimum corner distance between two JSON face records,
/// trying both direct and swapped orientations.
/// Returns (min_distance, is_swapped).
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

    let d_direct = ((x1_lo - x2_lo).powi(2) + (y1_lo - y2_lo).powi(2) + (z1_lo - z2_lo).powi(2)).sqrt()
        .max(((x1_hi - x2_hi).powi(2) + (y1_hi - y2_hi).powi(2) + (z1_hi - z2_hi).powi(2)).sqrt());
    let d_swapped = ((x1_lo - x2_hi).powi(2) + (y1_lo - y2_hi).powi(2) + (z1_lo - z2_hi).powi(2)).sqrt()
        .max(((x1_hi - x2_lo).powi(2) + (y1_hi - y2_lo).powi(2) + (z1_hi - z2_lo).powi(2)).sqrt());

    if d_direct <= d_swapped {
        Some((d_direct, false))
    } else {
        Some((d_swapped, true))
    }
}

/// Check if two blocks have overlapping AABBs within tolerance.
fn aabb_overlap(b1: &plot3d::Block, b2: &plot3d::Block, tol: f64) -> bool {
    let aabb = |b: &plot3d::Block| -> [f64; 6] {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        let mut zmin = f64::INFINITY;
        let mut zmax = f64::NEG_INFINITY;
        for &x in &b.x { xmin = xmin.min(x); xmax = xmax.max(x); }
        for &y in &b.y { ymin = ymin.min(y); ymax = ymax.max(y); }
        for &z in &b.z { zmin = zmin.min(z); zmax = zmax.max(z); }
        [xmin, xmax, ymin, ymax, zmin, zmax]
    };
    let a = aabb(b1);
    let b = aabb(b2);
    a[1] + tol >= b[0] && b[1] + tol >= a[0]
        && a[3] + tol >= b[2] && b[3] + tol >= a[2]
        && a[5] + tol >= b[4] && b[5] + tol >= a[4]
}

#[test]
fn classify_missing_matches() {
    println!("\n=== CLASSIFY MISSING MATCHES ===\n");

    // Step 1: Load data
    println!("Loading JSON...");
    let json_str = std::fs::read_to_string(JSON_FILE).expect("Failed to read JSON file");
    let json_data: GridProConnectivity =
        serde_json::from_str(&json_str).expect("Failed to parse JSON");
    println!("  JSON face_matches: {}", json_data.face_matches.len());

    println!("Loading mesh...");
    let blocks = read_plot3d_binary(MESH_FILE, BinaryFormat::Raw, FloatPrecision::F64, Endian::Little)
        .expect("Failed to read mesh file");
    println!("  Blocks loaded: {}", blocks.len());

    let gcd = plot3d::utils::compute_min_gcd(&blocks);
    println!("  GCD: {}", gcd);

    // Step 2: Run connectivity
    println!("Running connectivity_fast...");
    let (computed_matches, _outer_faces) = connectivity_fast(&blocks);
    println!("  Computed: {}", computed_matches.len());

    // Step 3: Find missing matches
    let json_keys = build_json_keys(&json_data.face_matches);
    let computed_keys = build_computed_keys(&computed_matches);
    let missing: HashSet<_> = json_keys.difference(&computed_keys).cloned().collect();
    println!("  Missing: {}", missing.len());

    // Step 4: Classify each missing match
    // Collect the original JSON entries for missing matches
    let missing_json: Vec<&JsonFaceMatch> = json_data.face_matches.iter().filter(|jm| {
        let a = NormalizedFaceRecord::from_json_block(&jm.block1);
        let b = NormalizedFaceRecord::from_json_block(&jm.block2);
        let key = MatchKey::new(a, b);
        missing.contains(&key)
    }).collect();
    println!("  Missing JSON entries: {}", missing_json.len());

    // Reduce blocks for direct face matching tests
    let reduced = reduce_blocks(&blocks, gcd);

    // Classification buckets
    let mut cat_exact_match = Vec::new();         // corners at distance < 1e-6
    let mut cat_near_match = Vec::new();          // corners at distance 1e-6 to 1e-2
    let mut cat_far = Vec::new();                 // corners at distance > 1e-2
    let mut cat_out_of_bounds = Vec::new();       // indices out of block bounds

    for jm in &missing_json {
        match min_corner_distance(jm, &blocks) {
            Some((d, _swapped)) => {
                if d < 1e-6 {
                    cat_exact_match.push((*jm, d));
                } else if d < 1e-2 {
                    cat_near_match.push((*jm, d));
                } else {
                    cat_far.push((*jm, d));
                }
            }
            None => {
                cat_out_of_bounds.push(*jm);
            }
        }
    }

    println!("\n=== CLASSIFICATION SUMMARY ===");
    println!("  Exact match (d < 1e-6):        {} -- faces coincide, algorithm should find them", cat_exact_match.len());
    println!("  Near match (1e-6 < d < 1e-2):  {} -- tolerance issue or slight misalignment", cat_near_match.len());
    println!("  Far (d > 1e-2):                {} -- likely periodic or non-adjacent", cat_far.len());
    println!("  Out of bounds:                 {}", cat_out_of_bounds.len());

    // ── Detailed analysis of exact-match failures ──
    println!("\n=== EXACT-MATCH FAILURES (d < 1e-6) - first 50 ===");
    let mut aabb_fail = 0usize;
    let mut aabb_pass_no_face_match = 0usize;
    let mut face_match_found = 0usize;
    let _face_match_consumed: usize;

    for (idx, (jm, d)) in cat_exact_match.iter().enumerate().take(50) {
        let b1_idx = jm.block1.block_index;
        let b2_idx = jm.block2.block_index;

        // Check AABB overlap on reduced blocks
        let overlap = aabb_overlap(&reduced[b1_idx], &reduced[b2_idx], 1e-6);
        if !overlap {
            aabb_fail += 1;
            if idx < 10 {
                println!("  [{}] block{}<->block{}: AABB FAIL (d={:.2e})", idx, b1_idx, b2_idx, d);
            }
            continue;
        }

        // Get outer faces for both blocks (from reduced blocks)
        let (faces1, _) = get_outer_faces(&reduced[b1_idx]);
        let (faces2, _) = get_outer_faces(&reduced[b2_idx]);

        // Try full_face_match on all face pairs
        let mut found = false;
        for f1 in &faces1 {
            for f2 in &faces2 {
                if let Some(_orient) = full_face_match(f1, f2, 1e-6) {
                    found = true;
                    break;
                }
            }
            if found { break; }
        }

        if found {
            // Face match exists -- the algorithm's pipeline lost it (Phase 1 consumption or ordering)
            face_match_found += 1;
            if idx < 10 {
                let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
                let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);
                println!(
                    "  [{}] block{}[{},{},{}->{},{},{}]<->block{}[{},{},{}->{},{},{}]: FACE MATCH EXISTS but not in output (d={:.2e})",
                    idx, b1_idx,
                    n1.imin, n1.jmin, n1.kmin, n1.imax, n1.jmax, n1.kmax,
                    b2_idx,
                    n2.imin, n2.jmin, n2.kmin, n2.imax, n2.jmax, n2.kmax,
                    d
                );

                // Check if this block pair IS in computed matches (with different face indices)
                let has_some_match = computed_matches.iter().any(|fm| {
                    (fm.block1.block_index == b1_idx && fm.block2.block_index == b2_idx)
                    || (fm.block1.block_index == b2_idx && fm.block2.block_index == b1_idx)
                });
                println!("       Block pair has other computed matches: {}", has_some_match);
            }
        } else {
            // No face match even in isolation -- the faces don't actually match as full faces
            // Try node-by-node matching (get_face_intersection)
            let mut partial_found = false;
            for f1 in &faces1 {
                for f2 in &faces2 {
                    let (pts, _, _) = get_face_intersection(
                        f1, f2, &reduced[b1_idx], &reduced[b2_idx], 1e-6,
                    );
                    if !pts.is_empty() {
                        partial_found = true;
                        if idx < 10 {
                            println!(
                                "  [{}] block{}<->block{}: PARTIAL MATCH found ({} points, d={:.2e})",
                                idx, b1_idx, b2_idx, pts.len(), d
                            );
                        }
                        break;
                    }
                }
                if partial_found { break; }
            }
            if !partial_found {
                aabb_pass_no_face_match += 1;
                if idx < 10 {
                    let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
                    let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);
                    println!(
                        "  [{}] block{}[{},{},{}->{},{},{}]<->block{}[{},{},{}->{},{},{}]: NO FACE MATCH at all (d={:.2e})",
                        idx, b1_idx,
                        n1.imin, n1.jmin, n1.kmin, n1.imax, n1.jmax, n1.kmax,
                        b2_idx,
                        n2.imin, n2.jmin, n2.kmin, n2.imax, n2.jmax, n2.kmax,
                        d
                    );
                    // Print face details
                    println!("       Block{} has {} outer faces, block{} has {} outer faces",
                        b1_idx, faces1.len(), b2_idx, faces2.len());
                    println!("       Block{} dims: {}x{}x{} (reduced), block{} dims: {}x{}x{} (reduced)",
                        b1_idx, reduced[b1_idx].imax, reduced[b1_idx].jmax, reduced[b1_idx].kmax,
                        b2_idx, reduced[b2_idx].imax, reduced[b2_idx].jmax, reduced[b2_idx].kmax);
                }
            }
        }
    }

    let exact_total = cat_exact_match.len();
    // Extrapolate for all exact matches
    let sampled = exact_total.min(50);
    println!("\n  Exact-match failure breakdown (from {} sampled of {}):", sampled, exact_total);
    println!("    AABB fail:                {}", aabb_fail);
    println!("    Face match exists (lost): {}", face_match_found);
    println!("    No face match at all:     {}", aabb_pass_no_face_match);

    // ── Analyze "far" matches for periodicity ──
    println!("\n=== FAR MATCHES (d > 1e-2) - likely periodic ===");
    // Compute distance histogram
    let mut dist_buckets = [0usize; 5]; // <0.1, <0.5, <1.0, <5.0, >=5.0
    for (_, d) in &cat_far {
        if *d < 0.1 { dist_buckets[0] += 1; }
        else if *d < 0.5 { dist_buckets[1] += 1; }
        else if *d < 1.0 { dist_buckets[2] += 1; }
        else if *d < 5.0 { dist_buckets[3] += 1; }
        else { dist_buckets[4] += 1; }
    }
    println!("  Distance distribution:");
    println!("    0.01 - 0.1:   {}", dist_buckets[0]);
    println!("    0.1  - 0.5:   {}", dist_buckets[1]);
    println!("    0.5  - 1.0:   {}", dist_buckets[2]);
    println!("    1.0  - 5.0:   {}", dist_buckets[3]);
    println!("    >= 5.0:       {}", dist_buckets[4]);

    // Print first 10 far matches with coordinates
    println!("\n  First 10 far matches:");
    for (idx, (jm, d)) in cat_far.iter().enumerate().take(10) {
        let b1_idx = jm.block1.block_index;
        let b2_idx = jm.block2.block_index;
        let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
        let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);

        if b1_idx < blocks.len() && b2_idx < blocks.len()
            && n1.imax < blocks[b1_idx].imax && n1.jmax < blocks[b1_idx].jmax && n1.kmax < blocks[b1_idx].kmax
            && n2.imax < blocks[b2_idx].imax && n2.jmax < blocks[b2_idx].jmax && n2.kmax < blocks[b2_idx].kmax
        {
            let (x1, y1, z1) = blocks[b1_idx].xyz(n1.imin, n1.jmin, n1.kmin);
            let (x2, y2, z2) = blocks[b2_idx].xyz(n2.imin, n2.jmin, n2.kmin);
            println!(
                "    [{}] block{}<->block{}: d={:.4}, b1_corner=({:.4},{:.4},{:.4}), b2_corner=({:.4},{:.4},{:.4})",
                idx, b1_idx, b2_idx, d, x1, y1, z1, x2, y2, z2
            );
        }
    }

    // ── Analyze "near" matches - try higher tolerance ──
    println!("\n=== NEAR MATCHES (1e-6 < d < 1e-2) ===");
    let mut would_match_1e4 = 0usize;
    let mut would_match_1e3 = 0usize;
    let mut would_match_1e2 = 0usize;
    for (_, d) in &cat_near_match {
        if *d < 1e-4 { would_match_1e4 += 1; }
        if *d < 1e-3 { would_match_1e3 += 1; }
        if *d < 1e-2 { would_match_1e2 += 1; }
    }
    println!("  Would match with tol=1e-4: {}", would_match_1e4);
    println!("  Would match with tol=1e-3: {}", would_match_1e3);
    println!("  Would match with tol=1e-2: {}", would_match_1e2);

    // ── Check block pairs with different face indices ──
    println!("\n=== BLOCK PAIRS IN BOTH BUT DIFFERENT FACES ===");
    let json_pairs = group_by_block_pair(&json_keys);
    let computed_pairs = group_by_block_pair(&computed_keys);

    let mut split_diff_count = 0usize;
    let mut json_has_more_count = 0usize;
    let mut computed_has_more_count = 0usize;
    let mut shown = 0;

    for pair in json_pairs.keys() {
        if let Some(comp_faces) = computed_pairs.get(pair) {
            let json_faces = &json_pairs[pair];
            let json_set: HashSet<_> = json_faces.iter().collect();
            let comp_set: HashSet<_> = comp_faces.iter().collect();
            if json_set != comp_set {
                let json_extra = json_set.difference(&comp_set).count();
                let comp_extra = comp_set.difference(&json_set).count();

                if json_extra > 0 && comp_extra > 0 {
                    split_diff_count += 1;
                    if shown < 5 {
                        println!("  ({},{}): JSON has {} unique, computed has {} unique, {} in common",
                            pair.0, pair.1, json_extra, comp_extra,
                            json_set.intersection(&comp_set).count());
                        shown += 1;
                    }
                } else if json_extra > 0 {
                    json_has_more_count += 1;
                } else {
                    computed_has_more_count += 1;
                }
            }
        }
    }
    println!("  Different subdivision (both have unique): {}", split_diff_count);
    println!("  JSON has extra faces for pair:            {}", json_has_more_count);
    println!("  Computed has extra faces for pair:        {}", computed_has_more_count);

    // ── Final summary ──
    println!("\n=== OVERALL ROOT CAUSE BREAKDOWN ===");
    println!("  Total missing:     {}", missing.len());
    println!("  Exact (d<1e-6):    {} -- algorithm bug (faces coincide but not found)", cat_exact_match.len());
    println!("  Near (1e-6..1e-2): {} -- tolerance issue", cat_near_match.len());
    println!("  Far (d>1e-2):      {} -- likely periodic or non-standard", cat_far.len());
    println!("  Out of bounds:     {}", cat_out_of_bounds.len());
    println!();
}
