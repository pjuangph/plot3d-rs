//! Debug test comparing plot3d-rs connectivity against GridPro reference JSON.
//!
//! Usage (release build required for large meshes):
//!   cargo test --release --test test_connectivity_debug -- --nocapture

mod common;

use std::collections::{HashMap, HashSet};

use plot3d::{
    connectivity_fast, full_face_match, get_face_intersection, get_outer_faces, read_plot3d_binary,
    reduce_blocks, write_plot3d, BinaryFormat, Endian, FaceMatch, FloatPrecision,
};

use common::{
    aabb_overlap, build_computed_keys, build_json_keys, group_by_block_pair, min_corner_distance,
    GridProConnectivity, JsonFaceMatch, MatchKey, NormalizedFaceRecord,
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
        b1_idx,
        block1.imax,
        block1.jmax,
        block1.kmax,
        b2_idx,
        block2.imax,
        block2.jmax,
        block2.kmax,
    );

    // Validate indices are within bounds
    let b1_ok = norm1.imax < block1.imax && norm1.jmax < block1.jmax && norm1.kmax < block1.kmax;
    let b2_ok = norm2.imax < block2.imax && norm2.jmax < block2.jmax && norm2.kmax < block2.kmax;

    if !b1_ok || !b2_ok {
        println!(
            "       WARNING: indices out of block bounds! b1_ok={}, b2_ok={}",
            b1_ok, b2_ok
        );
        return;
    }

    // Corner coordinates
    let (x1_lo, y1_lo, z1_lo) = block1.xyz(norm1.imin, norm1.jmin, norm1.kmin);
    let (x1_hi, y1_hi, z1_hi) = block1.xyz(norm1.imax, norm1.jmax, norm1.kmax);
    let (x2_lo, y2_lo, z2_lo) = block2.xyz(norm2.imin, norm2.jmin, norm2.kmin);
    let (x2_hi, y2_hi, z2_hi) = block2.xyz(norm2.imax, norm2.jmax, norm2.kmax);

    // Try both corner pairings: direct and swapped
    let d_lo_lo =
        ((x1_lo - x2_lo).powi(2) + (y1_lo - y2_lo).powi(2) + (z1_lo - z2_lo).powi(2)).sqrt();
    let d_hi_hi =
        ((x1_hi - x2_hi).powi(2) + (y1_hi - y2_hi).powi(2) + (z1_hi - z2_hi).powi(2)).sqrt();
    let d_lo_hi =
        ((x1_lo - x2_hi).powi(2) + (y1_lo - y2_hi).powi(2) + (z1_lo - z2_hi).powi(2)).sqrt();
    let d_hi_lo =
        ((x1_hi - x2_lo).powi(2) + (y1_hi - y2_lo).powi(2) + (z1_hi - z2_lo).powi(2)).sqrt();

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
            "b1.imin", "b1.jmin", "b1.kmin", "b1.imax", "b1.jmax", "b1.kmax", "b2.imin", "b2.jmin",
            "b2.kmin", "b2.imax", "b2.jmax", "b2.kmax",
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
        println!(
            "       No computed matches for block pair ({}, {})",
            b1_idx, b2_idx
        );
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
    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
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
    println!(
        "  Unique block pairs in computed:  {}",
        computed_pair_set.len()
    );
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
            println!(
                "  [{}] blocks ({}, {}): {} face match(es)",
                i,
                pair.0,
                pair.1,
                entries.len()
            );
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
            println!(
                "  [{}] blocks ({}, {}): {} face match(es)",
                i,
                pair.0,
                pair.1,
                entries.len()
            );
        }
    }

    // ── Step 7: Detailed diagnostics for missing matches ─────────────────
    println!("\n=== Step 7: Detailed diagnostics for missing matches (first 20) ===");
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
        println!("  Wrote {} blocks to {}", subset.len(), SUBSET_OUTPUT);

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
    println!(
        "  JSON face_matches:           {}",
        json_data.face_matches.len()
    );
    println!("  JSON unique keys:            {}", json_keys.len());
    println!("  Computed face_matches:       {}", computed_matches.len());
    println!("  Computed unique keys:        {}", computed_keys.len());
    println!("  Exact matches:               {}", common_count);
    println!("  Missing from computed:       {}", missing.len());
    println!("  Extra in computed:           {}", extra.len());
    println!("  Block pairs only in JSON:    {}", pairs_only_json.len());
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
    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
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
    let missing_json: Vec<&JsonFaceMatch> = json_data
        .face_matches
        .iter()
        .filter(|jm| {
            let a = NormalizedFaceRecord::from_json_block(&jm.block1);
            let b = NormalizedFaceRecord::from_json_block(&jm.block2);
            let key = MatchKey::new(a, b);
            missing.contains(&key)
        })
        .collect();
    println!("  Missing JSON entries: {}", missing_json.len());

    // Reduce blocks for direct face matching tests
    let reduced = reduce_blocks(&blocks, gcd);

    // Classification buckets
    let mut cat_exact_match = Vec::new(); // corners at distance < 1e-6
    let mut cat_near_match = Vec::new(); // corners at distance 1e-6 to 1e-2
    let mut cat_far = Vec::new(); // corners at distance > 1e-2
    let mut cat_out_of_bounds = Vec::new(); // indices out of block bounds

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
    println!(
        "  Exact match (d < 1e-6):        {} -- faces coincide, algorithm should find them",
        cat_exact_match.len()
    );
    println!(
        "  Near match (1e-6 < d < 1e-2):  {} -- tolerance issue or slight misalignment",
        cat_near_match.len()
    );
    println!(
        "  Far (d > 1e-2):                {} -- likely periodic or non-adjacent",
        cat_far.len()
    );
    println!(
        "  Out of bounds:                 {}",
        cat_out_of_bounds.len()
    );

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
                println!(
                    "  [{}] block{}<->block{}: AABB FAIL (d={:.2e})",
                    idx, b1_idx, b2_idx, d
                );
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
            if found {
                break;
            }
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
                println!(
                    "       Block pair has other computed matches: {}",
                    has_some_match
                );
            }
        } else {
            // No face match even in isolation -- the faces don't actually match as full faces
            // Try node-by-node matching (get_face_intersection)
            let mut partial_found = false;
            for f1 in &faces1 {
                for f2 in &faces2 {
                    let (pts, _, _) =
                        get_face_intersection(f1, f2, &reduced[b1_idx], &reduced[b2_idx], 1e-6);
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
                if partial_found {
                    break;
                }
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
                    println!(
                        "       Block{} has {} outer faces, block{} has {} outer faces",
                        b1_idx,
                        faces1.len(),
                        b2_idx,
                        faces2.len()
                    );
                    println!(
                        "       Block{} dims: {}x{}x{} (reduced), block{} dims: {}x{}x{} (reduced)",
                        b1_idx,
                        reduced[b1_idx].imax,
                        reduced[b1_idx].jmax,
                        reduced[b1_idx].kmax,
                        b2_idx,
                        reduced[b2_idx].imax,
                        reduced[b2_idx].jmax,
                        reduced[b2_idx].kmax
                    );
                }
            }
        }
    }

    let exact_total = cat_exact_match.len();
    // Extrapolate for all exact matches
    let sampled = exact_total.min(50);
    println!(
        "\n  Exact-match failure breakdown (from {} sampled of {}):",
        sampled, exact_total
    );
    println!("    AABB fail:                {}", aabb_fail);
    println!("    Face match exists (lost): {}", face_match_found);
    println!("    No face match at all:     {}", aabb_pass_no_face_match);

    // ── Analyze "far" matches for periodicity ──
    println!("\n=== FAR MATCHES (d > 1e-2) - likely periodic ===");
    // Compute distance histogram
    let mut dist_buckets = [0usize; 5]; // <0.1, <0.5, <1.0, <5.0, >=5.0
    for (_, d) in &cat_far {
        if *d < 0.1 {
            dist_buckets[0] += 1;
        } else if *d < 0.5 {
            dist_buckets[1] += 1;
        } else if *d < 1.0 {
            dist_buckets[2] += 1;
        } else if *d < 5.0 {
            dist_buckets[3] += 1;
        } else {
            dist_buckets[4] += 1;
        }
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

        if b1_idx < blocks.len()
            && b2_idx < blocks.len()
            && n1.imax < blocks[b1_idx].imax
            && n1.jmax < blocks[b1_idx].jmax
            && n1.kmax < blocks[b1_idx].kmax
            && n2.imax < blocks[b2_idx].imax
            && n2.jmax < blocks[b2_idx].jmax
            && n2.kmax < blocks[b2_idx].kmax
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
        if *d < 1e-4 {
            would_match_1e4 += 1;
        }
        if *d < 1e-3 {
            would_match_1e3 += 1;
        }
        if *d < 1e-2 {
            would_match_1e2 += 1;
        }
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
                        println!(
                            "  ({},{}): JSON has {} unique, computed has {} unique, {} in common",
                            pair.0,
                            pair.1,
                            json_extra,
                            comp_extra,
                            json_set.intersection(&comp_set).count()
                        );
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
    println!(
        "  Different subdivision (both have unique): {}",
        split_diff_count
    );
    println!(
        "  JSON has extra faces for pair:            {}",
        json_has_more_count
    );
    println!(
        "  Computed has extra faces for pair:        {}",
        computed_has_more_count
    );

    // ── Final summary ──
    println!("\n=== OVERALL ROOT CAUSE BREAKDOWN ===");
    println!("  Total missing:     {}", missing.len());
    println!(
        "  Exact (d<1e-6):    {} -- algorithm bug (faces coincide but not found)",
        cat_exact_match.len()
    );
    println!(
        "  Near (1e-6..1e-2): {} -- tolerance issue",
        cat_near_match.len()
    );
    println!(
        "  Far (d>1e-2):      {} -- likely periodic or non-standard",
        cat_far.len()
    );
    println!("  Out of bounds:     {}", cat_out_of_bounds.len());
    println!();
}

// ---------------------------------------------------------------------------
// Targeted diagnostic for specific blocks with connectivity issues
// ---------------------------------------------------------------------------

#[test]
fn debug_target_blocks() {
    const TARGET_BLOCKS: &[usize] = &[5003, 4997, 1352, 4611, 1437, 2603];

    println!("\n=== DEBUG TARGET BLOCKS: {:?} ===\n", TARGET_BLOCKS);

    // ── Step 1: Read mesh ──
    println!("Reading mesh...");
    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read mesh file");
    println!("  {} blocks loaded\n", blocks.len());

    let gcd = plot3d::utils::compute_min_gcd(&blocks);
    let reduced = reduce_blocks(&blocks, gcd);
    println!("  GCD: {}\n", gcd);

    // ── Step 2: Run connectivity ──
    println!("Running connectivity_fast...");
    let (computed_matches, outer_faces) = connectivity_fast(&blocks);
    println!(
        "  {} face matches, {} outer faces\n",
        computed_matches.len(),
        outer_faces.len()
    );

    // ── Step 3: Per-block diagnostics ──
    for &bi in TARGET_BLOCKS {
        println!("{}", "=".repeat(60));
        println!("=== BLOCK {} ===", bi);
        println!("{}", "=".repeat(60));

        if bi >= blocks.len() {
            println!(
                "  ERROR: block index {} out of range (nblocks={})",
                bi,
                blocks.len()
            );
            continue;
        }

        let block = &blocks[bi];
        let rblock = &reduced[bi];

        // 3a: Block dimensions
        println!(
            "  Dimensions (raw):     {}x{}x{}",
            block.imax, block.jmax, block.kmax
        );
        println!(
            "  Dimensions (reduced): {}x{}x{}",
            rblock.imax, rblock.jmax, rblock.kmax
        );
        let npoints = block.imax * block.jmax * block.kmax;
        println!("  Total points: {}", npoints);

        // 3b: AABB
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for &x in &block.x {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
        }
        for &y in &block.y {
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        for &z in &block.z {
            zmin = zmin.min(z);
            zmax = zmax.max(z);
        }
        println!(
            "  AABB: x=[{:.6},{:.6}] y=[{:.6},{:.6}] z=[{:.6},{:.6}]",
            xmin, xmax, ymin, ymax, zmin, zmax
        );

        // 3c: Outer faces (raw block)
        let (raw_faces, raw_self_matches) = get_outer_faces(block);
        println!("\n  get_outer_faces (raw):");
        println!("    Outer faces: {}", raw_faces.len());
        println!("    Self-match pairs: {}", raw_self_matches.len());
        for (idx, face) in raw_faces.iter().enumerate() {
            let (cx0, cy0, cz0) = block.xyz(face.imin(), face.jmin(), face.kmin());
            let (cx1, cy1, cz1) = block.xyz(face.imax(), face.jmax(), face.kmax());
            println!(
                "    Face[{}]: [{},{},{} -> {},{},{}] corners=({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                idx, face.imin(), face.jmin(), face.kmin(),
                face.imax(), face.jmax(), face.kmax(),
                cx0, cy0, cz0, cx1, cy1, cz1
            );
        }
        for (idx, (fa, fb)) in raw_self_matches.iter().enumerate() {
            println!(
                "    SelfMatch[{}]: [{},{},{}->{},{},{}] <-> [{},{},{}->{},{},{}]",
                idx,
                fa.imin(),
                fa.jmin(),
                fa.kmin(),
                fa.imax(),
                fa.jmax(),
                fa.kmax(),
                fb.imin(),
                fb.jmin(),
                fb.kmin(),
                fb.imax(),
                fb.jmax(),
                fb.kmax(),
            );
        }

        // 3d: Outer faces (reduced block)
        let (red_faces, red_self_matches) = get_outer_faces(rblock);
        println!("\n  get_outer_faces (reduced, GCD={}):", gcd);
        println!("    Outer faces: {}", red_faces.len());
        println!("    Self-match pairs: {}", red_self_matches.len());
        for (idx, face) in red_faces.iter().enumerate() {
            let (cx0, cy0, cz0) = rblock.xyz(face.imin(), face.jmin(), face.kmin());
            let (cx1, cy1, cz1) = rblock.xyz(face.imax(), face.jmax(), face.kmax());
            println!(
                "    Face[{}]: [{},{},{} -> {},{},{}] corners=({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                idx, face.imin(), face.jmin(), face.kmin(),
                face.imax(), face.jmax(), face.kmax(),
                cx0, cy0, cz0, cx1, cy1, cz1
            );
        }
        for (idx, (fa, fb)) in red_self_matches.iter().enumerate() {
            println!(
                "    SelfMatch[{}]: [{},{},{}->{},{},{}] <-> [{},{},{}->{},{},{}]",
                idx,
                fa.imin(),
                fa.jmin(),
                fa.kmin(),
                fa.imax(),
                fa.jmax(),
                fa.kmax(),
                fb.imin(),
                fb.jmin(),
                fb.kmin(),
                fb.imax(),
                fb.jmax(),
                fb.kmax(),
            );
        }

        // 3e: All connectivity matches involving this block
        let block_matches: Vec<_> = computed_matches
            .iter()
            .filter(|fm| fm.block1.block_index == bi || fm.block2.block_index == bi)
            .collect();
        println!(
            "\n  Connectivity matches involving block {}: {}",
            bi,
            block_matches.len()
        );
        for (idx, fm) in block_matches.iter().enumerate() {
            println!(
                "    Match[{}]: block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                idx,
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

        // 3f: Non-connected faces for this block
        let block_outer: Vec<_> = outer_faces.iter().filter(|f| f.block_index == bi).collect();
        println!(
            "\n  Non-connected faces for block {}: {}",
            bi,
            block_outer.len()
        );
        for (idx, f) in block_outer.iter().enumerate() {
            let (cx0, cy0, cz0) = block.xyz(f.i_lo(), f.j_lo(), f.k_lo());
            let (cx1, cy1, cz1) = block.xyz(f.i_hi(), f.j_hi(), f.k_hi());
            println!(
                "    Outer[{}]: [{},{},{}->{},{},{}] corners=({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                idx,
                f.il, f.jl, f.kl, f.ih, f.jh, f.kh,
                cx0, cy0, cz0, cx1, cy1, cz1
            );
        }

        // 3g: AABB overlap — find which blocks overlap with this block
        let mut overlapping_blocks = Vec::new();
        for (j, other) in reduced.iter().enumerate() {
            if j == bi {
                continue;
            }
            if aabb_overlap(rblock, other, 1e-6) {
                overlapping_blocks.push(j);
            }
        }
        println!(
            "\n  Blocks with AABB overlap (reduced): {}",
            overlapping_blocks.len()
        );
        if overlapping_blocks.len() <= 20 {
            println!("    {:?}", overlapping_blocks);
        } else {
            println!("    First 20: {:?}", &overlapping_blocks[..20]);
        }

        // 3h: If block has no connectivity, try manual face matching against overlapping blocks
        if block_matches.is_empty() {
            println!("\n  ** NO CONNECTIVITY ** Trying manual face matching...");
            let (faces_bi, _) = get_outer_faces(rblock);
            let mut any_match_found = false;

            for &oj in &overlapping_blocks {
                let (faces_oj, _) = get_outer_faces(&reduced[oj]);

                // Try full face match
                for (fi, f1) in faces_bi.iter().enumerate() {
                    for (fj, f2) in faces_oj.iter().enumerate() {
                        if let Some(orient) = full_face_match(f1, f2, 1e-6) {
                            println!(
                                "    FULL MATCH: block{}:face{} <-> block{}:face{} orient=({},{},{})",
                                bi, fi, oj, fj,
                                orient.u_reversed, orient.v_reversed, orient.swapped
                            );
                            any_match_found = true;
                        }
                    }
                }

                // Try partial (node-by-node) intersection
                for (fi, f1) in faces_bi.iter().enumerate() {
                    for (fj, f2) in faces_oj.iter().enumerate() {
                        let (pts, _, _) =
                            get_face_intersection(f1, f2, &reduced[bi], &reduced[oj], 1e-6);
                        if !pts.is_empty() {
                            println!(
                                "    PARTIAL MATCH: block{}:face{} <-> block{}:face{} ({} matching points)",
                                bi, fi, oj, fj, pts.len()
                            );
                            any_match_found = true;
                        }
                    }
                }
            }
            if !any_match_found {
                println!("    No matches found against any overlapping block!");
                // Try self-match manually with tighter analysis
                println!("\n    Checking face-to-face distances within block {}:", bi);
                for (fi, f1) in faces_bi.iter().enumerate() {
                    for fj in (fi + 1)..faces_bi.len() {
                        let f2 = &faces_bi[fj];
                        let c1_lo = rblock.xyz(f1.imin(), f1.jmin(), f1.kmin());
                        let c1_hi = rblock.xyz(f1.imax(), f1.jmax(), f1.kmax());
                        let c2_lo = rblock.xyz(f2.imin(), f2.jmin(), f2.kmin());
                        let c2_hi = rblock.xyz(f2.imax(), f2.jmax(), f2.kmax());
                        let d_lo = ((c1_lo.0 - c2_lo.0).powi(2)
                            + (c1_lo.1 - c2_lo.1).powi(2)
                            + (c1_lo.2 - c2_lo.2).powi(2))
                        .sqrt();
                        let d_hi = ((c1_hi.0 - c2_hi.0).powi(2)
                            + (c1_hi.1 - c2_hi.1).powi(2)
                            + (c1_hi.2 - c2_hi.2).powi(2))
                        .sqrt();
                        let d_swap_lo = ((c1_lo.0 - c2_hi.0).powi(2)
                            + (c1_lo.1 - c2_hi.1).powi(2)
                            + (c1_lo.2 - c2_hi.2).powi(2))
                        .sqrt();
                        let d_swap_hi = ((c1_hi.0 - c2_lo.0).powi(2)
                            + (c1_hi.1 - c2_lo.1).powi(2)
                            + (c1_hi.2 - c2_lo.2).powi(2))
                        .sqrt();
                        let best = d_lo.max(d_hi).min(d_swap_lo.max(d_swap_hi));
                        if best < 1e-2 {
                            println!(
                                "    Face{}[{},{},{}->{},{},{}] vs Face{}[{},{},{}->{},{},{}]: d_best={:.2e}",
                                fi, f1.imin(), f1.jmin(), f1.kmin(), f1.imax(), f1.jmax(), f1.kmax(),
                                fj, f2.imin(), f2.jmin(), f2.kmin(), f2.imax(), f2.jmax(), f2.kmax(),
                                best
                            );
                        }
                    }
                }
            }
        }

        // 3i: For ALL non-connected faces, try to find matching blocks
        if !block_outer.is_empty() {
            println!("\n  Searching for matches for non-connected faces...");
            for (oidx, outer_face) in block_outer.iter().enumerate() {
                let face_lo = (outer_face.i_lo(), outer_face.j_lo(), outer_face.k_lo());
                let face_hi = (outer_face.i_hi(), outer_face.j_hi(), outer_face.k_hi());

                // Get the face's corner coordinates
                let (fx0, fy0, fz0) = block.xyz(face_lo.0, face_lo.1, face_lo.2);
                let (fx1, fy1, fz1) = block.xyz(face_hi.0, face_hi.1, face_hi.2);

                println!(
                    "\n    Outer[{}]: [{},{},{}->{},{},{}]",
                    oidx, face_lo.0, face_lo.1, face_lo.2, face_hi.0, face_hi.1, face_hi.2
                );

                // Create a face from the reduced block for matching
                let rf_lo = (face_lo.0 / gcd, face_lo.1 / gcd, face_lo.2 / gcd);
                let rf_hi = (face_hi.0 / gcd, face_hi.1 / gcd, face_hi.2 / gcd);

                // Check bounds
                if rf_hi.0 >= rblock.imax || rf_hi.1 >= rblock.jmax || rf_hi.2 >= rblock.kmax {
                    println!("      Reduced face out of bounds, skipping");
                    continue;
                }

                let target_face = plot3d::create_face_from_diagonals(
                    rblock, rf_lo.0, rf_lo.1, rf_lo.2, rf_hi.0, rf_hi.1, rf_hi.2,
                );

                let mut found_any = false;
                for &oj in &overlapping_blocks {
                    let (faces_oj, _) = get_outer_faces(&reduced[oj]);
                    for (fj, f2) in faces_oj.iter().enumerate() {
                        // Quick corner distance check first
                        let c2_lo = reduced[oj].xyz(f2.imin(), f2.jmin(), f2.kmin());
                        let c2_hi = reduced[oj].xyz(f2.imax(), f2.jmax(), f2.kmax());
                        let d_lo = ((fx0 - c2_lo.0).powi(2)
                            + (fy0 - c2_lo.1).powi(2)
                            + (fz0 - c2_lo.2).powi(2))
                        .sqrt();
                        let d_hi = ((fx1 - c2_hi.0).powi(2)
                            + (fy1 - c2_hi.1).powi(2)
                            + (fz1 - c2_hi.2).powi(2))
                        .sqrt();
                        let d_swap_lo = ((fx0 - c2_hi.0).powi(2)
                            + (fy0 - c2_hi.1).powi(2)
                            + (fz0 - c2_hi.2).powi(2))
                        .sqrt();
                        let d_swap_hi = ((fx1 - c2_lo.0).powi(2)
                            + (fy1 - c2_lo.1).powi(2)
                            + (fz1 - c2_lo.2).powi(2))
                        .sqrt();
                        let best_corner = d_lo.max(d_hi).min(d_swap_lo.max(d_swap_hi));

                        if best_corner < 0.01 {
                            // Try full face match
                            if let Some(orient) = full_face_match(&target_face, f2, 1e-6) {
                                println!(
                                    "      FULL MATCH with block{}:face{} [{},{},{}->{},{},{}] d={:.2e} orient=({},{},{})",
                                    oj, fj,
                                    f2.imin(), f2.jmin(), f2.kmin(), f2.imax(), f2.jmax(), f2.kmax(),
                                    best_corner,
                                    orient.u_reversed, orient.v_reversed, orient.swapped,
                                );
                                found_any = true;
                            }
                            // Try partial intersection
                            let (pts, _, _) =
                                get_face_intersection(&target_face, f2, rblock, &reduced[oj], 1e-6);
                            if !pts.is_empty() {
                                // Map the matched points back to original indices
                                let i1_min = pts.iter().map(|p| p.i1).min().unwrap() * gcd;
                                let j1_min = pts.iter().map(|p| p.j1).min().unwrap() * gcd;
                                let k1_min = pts.iter().map(|p| p.k1).min().unwrap() * gcd;
                                let i1_max = pts.iter().map(|p| p.i1).max().unwrap() * gcd;
                                let j1_max = pts.iter().map(|p| p.j1).max().unwrap() * gcd;
                                let k1_max = pts.iter().map(|p| p.k1).max().unwrap() * gcd;
                                let i2_min = pts.iter().map(|p| p.i2).min().unwrap() * gcd;
                                let j2_min = pts.iter().map(|p| p.j2).min().unwrap() * gcd;
                                let k2_min = pts.iter().map(|p| p.k2).min().unwrap() * gcd;
                                let i2_max = pts.iter().map(|p| p.i2).max().unwrap() * gcd;
                                let j2_max = pts.iter().map(|p| p.j2).max().unwrap() * gcd;
                                let k2_max = pts.iter().map(|p| p.k2).max().unwrap() * gcd;
                                println!(
                                    "      PARTIAL MATCH with block{}: {} pts, block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                                    oj, pts.len(),
                                    bi, i1_min, j1_min, k1_min, i1_max, j1_max, k1_max,
                                    oj, i2_min, j2_min, k2_min, i2_max, j2_max, k2_max,
                                );
                                found_any = true;
                            }
                        }
                    }
                }
                if !found_any {
                    println!(
                        "      No match found (checked {} overlapping blocks)",
                        overlapping_blocks.len()
                    );
                }
            }
        }

        println!();
    }

    // ── Step 4: Check periodicity for non-connected faces ──
    println!("{}", "=".repeat(60));
    println!("=== CHECKING PERIODICITY FOR TARGET BLOCKS ===");
    println!("{}", "=".repeat(60));

    let nblades: usize = 44;
    let rotation_angle_deg: f64 = 360.0 / nblades as f64;
    let rotation_angle_rad: f64 = rotation_angle_deg.to_radians();

    println!("Running rotated_periodicity...");
    let (periodic_faces, remaining_outer) = plot3d::rotated_periodicity(
        &blocks,
        &computed_matches,
        &outer_faces,
        rotation_angle_deg,
        'x',
        true,
    );
    println!(
        "  {} periodic pairs, {} remaining outer",
        periodic_faces.len(),
        remaining_outer.len()
    );

    println!("\nVerifying periodicity...");
    let (verified, mismatched) =
        plot3d::verify_periodicity(&blocks, &periodic_faces, rotation_angle_rad, 'x', 1e-4);
    println!(
        "  {} verified, {} mismatched",
        verified.len(),
        mismatched.len()
    );

    // ── Step 4b: Node-level matching for non-connected faces ──
    // Use reduced blocks for speed. For each non-connected face, sample corner
    // nodes and check if they lie on any AABB-overlapping block's boundary.
    println!("\n{}", "=".repeat(60));
    println!("=== NODE-LEVEL MATCHING FOR NON-CONNECTED FACES ===");
    println!("{}", "=".repeat(60));

    for &bi in TARGET_BLOCKS {
        let block_outer: Vec<_> = outer_faces.iter().filter(|f| f.block_index == bi).collect();
        if block_outer.is_empty() {
            continue;
        }
        println!(
            "\n--- Block {} ({} non-connected faces) ---",
            bi,
            block_outer.len()
        );

        let rblock = &reduced[bi];

        // Find AABB-overlapping blocks (reduced)
        let overlapping: Vec<usize> = reduced
            .iter()
            .enumerate()
            .filter(|(j, other)| *j != bi && aabb_overlap(rblock, other, 1e-6))
            .map(|(j, _)| j)
            .collect();

        for (oidx, of) in block_outer.iter().enumerate() {
            let i_lo = of.i_lo() / gcd;
            let j_lo = of.j_lo() / gcd;
            let k_lo = of.k_lo() / gcd;
            let i_hi = of.i_hi() / gcd;
            let j_hi = of.j_hi() / gcd;
            let k_hi = of.k_hi() / gcd;

            println!(
                "  Outer[{}]: [{},{},{}->{},{},{}] (reduced from [{},{},{}->{},{},{}])",
                oidx,
                i_lo,
                j_lo,
                k_lo,
                i_hi,
                j_hi,
                k_hi,
                of.i_lo(),
                of.j_lo(),
                of.k_lo(),
                of.i_hi(),
                of.j_hi(),
                of.k_hi(),
            );

            // Check all 4 corners of the face
            let corners = [
                (i_lo, j_lo, k_lo),
                (i_hi, j_hi, k_hi),
                (i_lo, j_hi, k_hi),
                (i_hi, j_lo, k_lo),
            ];
            let tol = 1e-6;

            for &(ci, cj, ck) in &corners {
                if ci >= rblock.imax || cj >= rblock.jmax || ck >= rblock.kmax {
                    continue;
                }
                let (cx, cy, cz) = rblock.xyz(ci, cj, ck);

                let mut matching_blocks = Vec::new();
                for &oj in &overlapping {
                    let other = &reduced[oj];
                    let ni = other.imax - 1;
                    let nj = other.jmax - 1;
                    let nk = other.kmax - 1;

                    // Check 6 boundary faces of the other block
                    'face_check: for face_spec in &[
                        (0usize..=0, 0usize..=nj, 0usize..=nk),
                        (ni..=ni, 0..=nj, 0..=nk),
                        (0..=ni, 0..=0, 0..=nk),
                        (0..=ni, nj..=nj, 0..=nk),
                        (0..=ni, 0..=nj, 0..=0),
                        (0..=ni, 0..=nj, nk..=nk),
                    ] {
                        for fi in face_spec.0.clone() {
                            for fj in face_spec.1.clone() {
                                for fk in face_spec.2.clone() {
                                    let (ox, oy, oz) = other.xyz(fi, fj, fk);
                                    let d2 =
                                        (cx - ox).powi(2) + (cy - oy).powi(2) + (cz - oz).powi(2);
                                    if d2 < tol * tol {
                                        matching_blocks.push((oj, fi, fj, fk));
                                        break 'face_check;
                                    }
                                }
                            }
                        }
                    }
                }
                if !matching_blocks.is_empty() {
                    println!(
                        "    Corner({},{},{}) = ({:.6},{:.6},{:.6}) -> matches: {:?}",
                        ci,
                        cj,
                        ck,
                        cx,
                        cy,
                        cz,
                        matching_blocks.iter().take(5).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    // ── Step 4c: Direct matching test between target blocks and their neighbors ──
    println!("\n{}", "=".repeat(60));
    println!("=== DIRECT MATCHING TEST ===");
    println!("{}", "=".repeat(60));

    // For Block 5003: test against blocks 2603, 3201, 3472
    // For Block 1352: test against block 1437
    // For Block 4611: test against block 980, 4793, 4844
    let test_pairs: Vec<(usize, usize)> = vec![
        (2603, 5003),
        (3201, 5003),
        (3472, 5003),
        (3284, 5003),
        (3228, 4997),
        (1722, 4997),
        (24, 1352),
        (1437, 1352),
        (980, 4611),
        (4793, 4611),
        (4844, 4611),
    ];

    for (bi, bj) in &test_pairs {
        let bi = *bi;
        let bj = *bj;
        if bi >= reduced.len() || bj >= reduced.len() {
            continue;
        }

        // Get fresh outer faces for both blocks (reduced)
        let (mut faces_i, _) = get_outer_faces(&reduced[bi]);
        let (mut faces_j, _) = get_outer_faces(&reduced[bj]);
        for f in &mut faces_i {
            f.set_block_index(bi);
        }
        for f in &mut faces_j {
            f.set_block_index(bj);
        }

        println!(
            "\n  Testing blocks {} ({}x{}x{}) vs {} ({}x{}x{}):",
            bi,
            reduced[bi].imax,
            reduced[bi].jmax,
            reduced[bi].kmax,
            bj,
            reduced[bj].imax,
            reduced[bj].jmax,
            reduced[bj].kmax
        );
        println!(
            "    Block {} has {} faces, Block {} has {} faces",
            bi,
            faces_i.len(),
            bj,
            faces_j.len()
        );

        // Try find_matching_blocks directly
        let mut faces_i_clone = faces_i.clone();
        let mut faces_j_clone = faces_j.clone();
        let matches = plot3d::connectivity::find_matching_blocks(
            &reduced[bi],
            &reduced[bj],
            &mut faces_i_clone,
            &mut faces_j_clone,
            1e-6,
        );
        println!("    find_matching_blocks: {} matches found", matches.len());
        for (idx, pts) in matches.iter().enumerate() {
            let i1_min = pts.iter().map(|p| p.i1).min().unwrap() * gcd;
            let j1_min = pts.iter().map(|p| p.j1).min().unwrap() * gcd;
            let k1_min = pts.iter().map(|p| p.k1).min().unwrap() * gcd;
            let i1_max = pts.iter().map(|p| p.i1).max().unwrap() * gcd;
            let j1_max = pts.iter().map(|p| p.j1).max().unwrap() * gcd;
            let k1_max = pts.iter().map(|p| p.k1).max().unwrap() * gcd;
            let i2_min = pts.iter().map(|p| p.i2).min().unwrap() * gcd;
            let j2_min = pts.iter().map(|p| p.j2).min().unwrap() * gcd;
            let k2_min = pts.iter().map(|p| p.k2).min().unwrap() * gcd;
            let i2_max = pts.iter().map(|p| p.i2).max().unwrap() * gcd;
            let j2_max = pts.iter().map(|p| p.j2).max().unwrap() * gcd;
            let k2_max = pts.iter().map(|p| p.k2).max().unwrap() * gcd;
            println!(
                "      [{}] {} pts: block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                idx,
                pts.len(),
                bi,
                i1_min,
                j1_min,
                k1_min,
                i1_max,
                j1_max,
                k1_max,
                bj,
                i2_min,
                j2_min,
                k2_min,
                i2_max,
                j2_max,
                k2_max,
            );
        }
        println!(
            "    Remaining: block{} has {} faces, block{} has {} faces",
            bi,
            faces_i_clone.len(),
            bj,
            faces_j_clone.len()
        );

        // Also check existing connectivity matches between these blocks
        let existing: Vec<_> = computed_matches
            .iter()
            .filter(|fm| {
                (fm.block1.block_index == bi && fm.block2.block_index == bj)
                    || (fm.block1.block_index == bj && fm.block2.block_index == bi)
            })
            .collect();
        println!("    Existing connectivity matches: {}", existing.len());
        for (idx, fm) in existing.iter().enumerate() {
            println!(
                "      [{}] block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                idx,
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

    for &bi in TARGET_BLOCKS {
        println!("\n--- Block {} after periodicity ---", bi);

        let periodic_for_block: Vec<_> = periodic_faces
            .iter()
            .filter(|fm| fm.block1.block_index == bi || fm.block2.block_index == bi)
            .collect();
        println!("  Periodic matches: {}", periodic_for_block.len());
        for (idx, fm) in periodic_for_block.iter().enumerate() {
            println!(
                "    Periodic[{}]: block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
                idx,
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

        let remaining_for_block: Vec<_> = remaining_outer
            .iter()
            .filter(|f| f.block_index == bi)
            .collect();
        println!("  Remaining non-connected: {}", remaining_for_block.len());
        for (idx, f) in remaining_for_block.iter().enumerate() {
            let (cx0, cy0, cz0) = blocks[bi].xyz(f.i_lo(), f.j_lo(), f.k_lo());
            let (cx1, cy1, cz1) = blocks[bi].xyz(f.i_hi(), f.j_hi(), f.k_hi());
            println!(
                "    Remaining[{}]: [{},{},{}->{},{},{}] corners=({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                idx,
                f.il, f.jl, f.kl, f.ih, f.jh, f.kh,
                cx0, cy0, cz0, cx1, cy1, cz1
            );
        }

        // Summary of face accounting
        let conn_count = computed_matches
            .iter()
            .filter(|fm| fm.block1.block_index == bi || fm.block2.block_index == bi)
            .count();
        println!(
            "  SUMMARY: {} connectivity + {} periodic + {} remaining = {} total accounted",
            conn_count,
            periodic_for_block.len(),
            remaining_for_block.len(),
            conn_count + periodic_for_block.len() + remaining_for_block.len()
        );
    }

    println!("\n=== DONE ===");
}
