//! Integration test for the WELD mesh: connectivity + translational periodicity.
//!
//! Reads `/Volumes/T7/WELD/weld_ascii.xyz` (1900 blocks), runs connectivity
//! and z-direction translational periodicity, then verifies results.
//!
//! The test is skipped automatically when the mesh file is not present
//! (e.g. on CI or machines without the external drive).

use std::collections::HashSet;
use std::path::Path;

use serde_json::Value;

use plot3d::{
    apply_permutation, connectivity_fast, determine_plane, extract_canonical_grid,
    face_match_to_diagonal_json, face_match_to_json, face_record_to_diagonal_json,
    face_record_to_json, permutation_matrices_json, read_plot3d_ascii, translational_periodicity,
    try_all_permutations, verify_connectivity, verify_match, verify_partial_match, FaceMatch,
    FaceRecord,
};

const MESH_PATH: &str = "/Volumes/T7/WELD/weld_ascii.xyz";
const CONN_JSON: &str = "/Volumes/T7/WELD/weld_connectivity.json";
const CONN_PERIOD_JSON: &str = "/Volumes/T7/WELD/weld_connectivity-periodicity.json";

/// A canonicalised face-match key for set comparison (order-independent).
type MatchKey = (
    (usize, [usize; 3], [usize; 3]),
    (usize, [usize; 3], [usize; 3]),
);

fn face_key(rec: &FaceRecord) -> (usize, [usize; 3], [usize; 3]) {
    let mut lb = [rec.il, rec.jl, rec.kl];
    let mut ub = [rec.ih, rec.jh, rec.kh];
    for i in 0..3 {
        if lb[i] > ub[i] {
            std::mem::swap(&mut lb[i], &mut ub[i]);
        }
    }
    (rec.block_index, lb, ub)
}

fn match_key(fm: &FaceMatch) -> MatchKey {
    let a = face_key(&fm.block1);
    let b = face_key(&fm.block2);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn json_face_key(entry: &Value) -> (usize, [usize; 3], [usize; 3]) {
    let bi = entry["block_index"].as_u64().unwrap() as usize;
    let lb_arr = entry["lb"].as_array().unwrap();
    let ub_arr = entry["ub"].as_array().unwrap();
    let mut lb = [
        lb_arr[0].as_u64().unwrap() as usize,
        lb_arr[1].as_u64().unwrap() as usize,
        lb_arr[2].as_u64().unwrap() as usize,
    ];
    let mut ub = [
        ub_arr[0].as_u64().unwrap() as usize,
        ub_arr[1].as_u64().unwrap() as usize,
        ub_arr[2].as_u64().unwrap() as usize,
    ];
    for i in 0..3 {
        if lb[i] > ub[i] {
            std::mem::swap(&mut lb[i], &mut ub[i]);
        }
    }
    (bi, lb, ub)
}

fn json_match_key(entry: &Value) -> MatchKey {
    let a = json_face_key(&entry["block1"]);
    let b = json_face_key(&entry["block2"]);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[test]
fn weld_connectivity_and_periodicity() {
    // Skip if mesh not present
    if !Path::new(MESH_PATH).exists() {
        eprintln!("WELD mesh not found at {MESH_PATH}, skipping test.");
        return;
    }

    // ── Step 1: Read mesh ──
    println!("Reading WELD mesh...");
    let blocks = read_plot3d_ascii(MESH_PATH).unwrap();
    assert_eq!(blocks.len(), 1900, "Expected 1900 blocks");
    println!("  Read {} blocks", blocks.len());

    // ── Step 2: Run connectivity ──
    println!("Running connectivity_fast...");
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!(
        "  Raw: {} face matches, {} outer faces",
        face_matches.len(),
        outer_faces.len()
    );

    // ── Step 3: Verify connectivity (informational) ──
    println!("Verifying connectivity (point-by-point)...");
    let (verified, mismatched) = verify_connectivity(&blocks, &face_matches, 1e-6);
    println!(
        "  Verified: {}, False positives filtered: {}",
        verified.len(),
        mismatched.len()
    );

    // ── Step 4: Informational comparison against Python JSON ──
    if Path::new(CONN_JSON).exists() {
        let json_str = std::fs::read_to_string(CONN_JSON).unwrap();
        let json_data: Value = serde_json::from_str(&json_str).unwrap();
        let expected = json_data["face_matches"].as_array().unwrap();

        let rust_keys: HashSet<MatchKey> = verified.iter().map(match_key).collect();
        let python_keys: HashSet<MatchKey> = expected.iter().map(json_match_key).collect();
        let in_both = rust_keys.intersection(&python_keys).count();

        println!(
            "  Python comparison: {} Python matches, {} in both, {} only-Python, {} only-Rust",
            python_keys.len(),
            in_both,
            python_keys.difference(&rust_keys).count(),
            rust_keys.difference(&python_keys).count()
        );
    }

    // ── Step 5: Run translational periodicity (z direction) ──
    // Use outer_faces directly from connectivity_fast.
    // Even though connectivity may have some false positives, the z-boundary
    // faces should still be in the outer set (false positives are mostly
    // cross-axis internal faces, not z-boundary faces).
    println!("Running translational periodicity (z)...");
    let (periodic_matches, remaining_outer) = translational_periodicity(
        &blocks,
        &outer_faces,
        None, // auto-compute delta
        "z",
        None, // adaptive tolerance
        0.02, // min_shared_frac (Python default)
        4,    // min_shared_abs (Python default)
        1,    // stride_u
        1,    // stride_v
    );
    println!(
        "  {} periodic pairs, {} remaining outer faces",
        periodic_matches.len(),
        remaining_outer.len()
    );

    // ── Step 6: Informational comparison against Python periodicity ──
    if Path::new(CONN_PERIOD_JSON).exists() {
        let json_str = std::fs::read_to_string(CONN_PERIOD_JSON).unwrap();
        let json_data: Value = serde_json::from_str(&json_str).unwrap();
        let expected_remaining = json_data["outer_faces"].as_array().unwrap();

        // Periodic matches are in a separate "periodic_faces" array
        let python_periodic = json_data["periodic_faces"]
            .as_array()
            .map(|a| a.iter().collect::<Vec<_>>())
            .unwrap_or_default();
        let rust_periodic_keys: HashSet<MatchKey> =
            periodic_matches.iter().map(match_key).collect();
        let python_periodic_keys: HashSet<MatchKey> =
            python_periodic.iter().map(|e| json_match_key(e)).collect();
        let in_both = rust_periodic_keys
            .intersection(&python_periodic_keys)
            .count();

        println!(
            "  Python comparison: {} Python periodic, {} in both, {} only-Python, {} only-Rust",
            python_periodic_keys.len(),
            in_both,
            python_periodic_keys.difference(&rust_periodic_keys).count(),
            rust_periodic_keys.difference(&python_periodic_keys).count()
        );
        println!(
            "  Python remaining outer: {}, Rust remaining outer: {}",
            expected_remaining.len(),
            remaining_outer.len()
        );
    }

    // ── Step 7: KEY ASSERTION — no z-boundary faces remain unpaired ──
    println!("Checking for unpaired z-boundary faces...");
    let z_min = blocks
        .iter()
        .map(|b| b.z_slice().iter().cloned().fold(f64::INFINITY, f64::min))
        .fold(f64::INFINITY, f64::min);
    let z_max = blocks
        .iter()
        .map(|b| {
            b.z_slice()
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  z range: [{z_min}, {z_max}]");

    let tol = 1e-6;
    let mut on_zmin = 0usize;
    let mut on_zmax = 0usize;
    for rec in &remaining_outer {
        let bi = rec.block_index;
        if bi >= blocks.len() {
            continue;
        }
        let b = &blocks[bi];
        // Bounds-check indices before accessing block data
        let ilo = rec.i_lo();
        let jlo = rec.j_lo();
        let klo = rec.k_lo();
        let ihi = rec.i_hi();
        let jhi = rec.j_hi();
        let khi = rec.k_hi();
        if ilo >= b.imax || jlo >= b.jmax || klo >= b.kmax {
            continue;
        }
        if ihi >= b.imax || jhi >= b.jmax || khi >= b.kmax {
            continue;
        }
        let corners = [b.xyz(ilo, jlo, klo), b.xyz(ihi, jhi, khi)];
        let all_zmin = corners.iter().all(|(_, _, z)| (z - z_min).abs() < tol);
        let all_zmax = corners.iter().all(|(_, _, z)| (z - z_max).abs() < tol);
        if all_zmin {
            on_zmin += 1;
        }
        if all_zmax {
            on_zmax += 1;
        }
    }
    println!("  Remaining on z-min: {on_zmin}, z-max: {on_zmax}");
    assert_eq!(
        on_zmin, 0,
        "Found {on_zmin} unpaired faces on z-min boundary"
    );
    assert_eq!(
        on_zmax, 0,
        "Found {on_zmax} unpaired faces on z-max boundary"
    );

    // ── Step 8: Test new helper functions on a sample verified match ──
    println!("Testing extract_canonical_grid + apply_permutation + verify_match...");
    if !verified.is_empty() {
        let sample = &verified[0];
        let (pts_a, nu_a, nv_a) =
            extract_canonical_grid(&blocks[sample.block1.block_index], &sample.block1).unwrap();
        let (pts_b, nu_b, nv_b) =
            extract_canonical_grid(&blocks[sample.block2.block_index], &sample.block2).unwrap();

        let perm_idx = try_all_permutations(&pts_a, nu_a, nv_a, &pts_b, nu_b, nv_b, 1e-6);
        assert!(
            perm_idx.is_some(),
            "Should find a valid permutation for a verified match"
        );
        let perm = perm_idx.unwrap();

        let (permuted, out_nu, out_nv) = apply_permutation(&pts_b, nu_b, nv_b, perm);
        assert_eq!(
            (out_nu, out_nv),
            (nu_a, nv_a),
            "Shape must match after permutation"
        );
        assert!(
            verify_match(&pts_a, &permuted, 1e-6),
            "Permuted grid should match"
        );
        println!(
            "  Sample: block {}<->{}, perm={} OK",
            sample.block1.block_index, sample.block2.block_index, perm
        );

        // Test verify_partial_match (should also work for full matches)
        let (count, total) = verify_partial_match(&pts_a, &permuted, 1e-6);
        assert_eq!(count, total, "Full match: all points should match");
        println!("  verify_partial_match: {}/{} OK", count, total);

        // Test determine_plane
        let plane = determine_plane(&sample.block1, &sample.block2);
        println!("  determine_plane: {:?}", plane);
    }

    // ── Step 9: Test JSON serialization functions ──
    println!("Testing JSON serialization...");
    if !verified.is_empty() {
        let sample = &verified[0];

        // Default format (lo/hi)
        let json_rec = face_record_to_json(&sample.block1);
        assert!(json_rec["block_index"].is_number());
        assert!(json_rec["lo"].is_array());
        assert!(json_rec["hi"].is_array());

        let json_match = face_match_to_json(sample);
        assert!(json_match["block1"].is_object());
        assert!(json_match["block2"].is_object());
        assert!(json_match["permutation_index"].is_number());

        // Diagonal format (lb/ub)
        let json_diag_rec = face_record_to_diagonal_json(&sample.block1);
        assert!(json_diag_rec["block_index"].is_number());
        assert!(json_diag_rec["lb"].is_array());
        assert!(json_diag_rec["ub"].is_array());

        let json_diag_match = face_match_to_diagonal_json(sample);
        assert!(json_diag_match["block1"].is_object());
        assert!(json_diag_match["block2"].is_object());
        assert!(json_diag_match["permutation_index"].is_number());

        // Permutation matrices
        let perm_mats = permutation_matrices_json();
        assert_eq!(perm_mats.len(), 8, "Should have 8 permutation matrices");

        println!(
            "  lo/hi JSON: {}",
            serde_json::to_string(&json_match).unwrap()
        );
        println!(
            "  lb/ub JSON: {}",
            serde_json::to_string(&json_diag_match).unwrap()
        );
        println!("  All serialization checks passed");
    }

    // ── Summary ──
    println!("\n=== WELD Test Summary ===");
    println!("  Blocks: {}", blocks.len());
    println!(
        "  Connectivity: {} raw -> {} verified ({} false positives filtered)",
        face_matches.len(),
        verified.len(),
        mismatched.len()
    );
    println!("  Periodicity: {} z-periodic pairs", periodic_matches.len());
    println!("  Remaining outer: {}", remaining_outer.len());
    println!("  Z-boundary unpaired: 0 (all paired)");
    println!("  PASS");
}
