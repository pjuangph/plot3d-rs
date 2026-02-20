//! Diagnostic test for investigating why specific face pairs fail (or succeed)
//! in the rotational periodicity matching pipeline.
//!
//! Usage (release build recommended for large meshes):
//!   cargo test --release --test test_debug_periodicity -- --nocapture
//!
//! Edit `diagnostic_cases()` below to add or modify test pairs.

use plot3d::{
    apply_rotation, count_rotated_corners_on_face, create_face_from_diagonals,
    create_rotation_matrix, faces_support_any, faces_support_direction,
    full_face_match_transformed, periodicity_check_with_points, read_plot3d_binary, rotate_block,
    to_radius, to_theta, BinaryFormat, Block, Endian, Float, FloatPrecision,
};

// ===========================================================================
// Configuration
// ===========================================================================

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

const ROTATION_AXIS: char = 'x';
const NBLADES: usize = 44;
const MATCH_TOL: Float = 1e-6;

/// Face spec: (block_index, imin, jmin, kmin, imax, jmax, kmax)
/// Block index is 0-based (same as JSON and tecplot.json).
type FaceSpec = (usize, usize, usize, usize, usize, usize, usize);

struct DiagnosticCase {
    name: &'static str,
    face_a: FaceSpec,
    faces_b: Vec<FaceSpec>,
}

fn diagnostic_cases() -> Vec<DiagnosticCase> {
    vec![
        // Pair 1: Baseline — already working cross-axis J/I pair
        DiagnosticCase {
            name: "Pair 1 (BASELINE) Block 3741 J-const <-> Block 4554 I-const",
            face_a: (3741, 0, 0, 12, 192, 0, 60),
            faces_b: vec![(4554, 36, 0, 0, 36, 24, 216)],
        },
        // Pair 2: Split face — Block 4115 should split and match 4561 (top) + 4565 (bottom)
        DiagnosticCase {
            name: "Pair 2 (SPLIT) Block 4115 J-const <-> Block 4561 + Block 4565 I-const",
            face_a: (4115, 0, 108, 0, 24, 108, 48),
            faces_b: vec![
                (4561, 0, 0, 0, 0, 24, 24),
                (4565, 0, 0, 0, 0, 24, 24),
            ],
        },
        // Pair 3: Missing — Block 3994 (remnant I=12, K=36..60) <-> Block 3664 (K=124)
        DiagnosticCase {
            name: "Pair 3 (MISSING) Block 3994 I-const <-> Block 3664 K-const",
            face_a: (3994, 12, 0, 36, 12, 264, 60),
            faces_b: vec![(3664, 0, 0, 124, 48, 12, 124)],
        },
    ]
}

// ===========================================================================
// 9-step diagnostic for a single (face_a, face_b) sub-pair
// ===========================================================================

fn diagnose_single_pair(
    blocks: &[Block],
    spec_a: FaceSpec,
    spec_b: FaceSpec,
    rotation_angle_deg: Float,
    rotation_angle_rad: Float,
    rot_forward: [[Float; 3]; 3],
    rot_backward: [[Float; 3]; 3],
    label: &str,
) -> bool {
    let (blk_a_idx, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax) = spec_a;
    let (blk_b_idx, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax) = spec_b;

    println!("\n  --- {} ---", label);

    assert!(blk_a_idx < blocks.len(), "Block A {} out of range", blk_a_idx);
    assert!(blk_b_idx < blocks.len(), "Block B {} out of range", blk_b_idx);

    let block_a = &blocks[blk_a_idx];
    let block_b = &blocks[blk_b_idx];

    println!(
        "  Block A [{}]: dims = {}x{}x{}",
        blk_a_idx, block_a.imax, block_a.jmax, block_a.kmax
    );
    println!(
        "  Block B [{}]: dims = {}x{}x{}",
        blk_b_idx, block_b.imax, block_b.jmax, block_b.kmax
    );

    // Step 2: Create faces
    let face_a = create_face_from_diagonals(block_a, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax);
    let face_b = create_face_from_diagonals(block_b, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax);

    println!(
        "  Face A: diag=({},{},{})..({},{},{}), verts={}, const_type={}",
        a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax,
        face_a.vertices().len(),
        face_a.const_type()
    );
    println!(
        "  Face B: diag=({},{},{})..({},{},{}), verts={}, const_type={}",
        b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax,
        face_b.vertices().len(),
        face_b.const_type()
    );

    // Step 3: Face compatibility
    let support_i = faces_support_direction(&face_a, &face_b, "i");
    let support_j = faces_support_direction(&face_a, &face_b, "j");
    let support_k = faces_support_direction(&face_a, &face_b, "k");
    let support_any = faces_support_any(&face_a, &face_b);

    println!("  faces_support: any={} i={} j={} k={}", support_any, support_i, support_j, support_k);

    if !support_any {
        println!("  FAIL: faces_support_any=false, algorithm skips this pair");
        return false;
    }

    // Step 4: Cylindrical coordinate analysis
    let verts_a = face_a.vertices();
    let verts_b = face_b.vertices();

    let theta_a: Vec<Float> = verts_a.iter().map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let theta_b: Vec<Float> = verts_b.iter().map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let radius_a: Vec<Float> = verts_a.iter().map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let radius_b: Vec<Float> = verts_b.iter().map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let axial_a: Vec<Float> = verts_a.iter().map(|v| match ROTATION_AXIS { 'x' => v[0], 'y' => v[1], _ => v[2] }).collect();
    let axial_b: Vec<Float> = verts_b.iter().map(|v| match ROTATION_AXIS { 'x' => v[0], 'y' => v[1], _ => v[2] }).collect();

    let mean = |v: &[Float]| v.iter().sum::<Float>() / v.len() as Float;
    let min_f = |v: &[Float]| v.iter().cloned().fold(Float::INFINITY, Float::min);
    let max_f = |v: &[Float]| v.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

    let theta_a_mean = mean(&theta_a);
    let theta_b_mean = mean(&theta_b);
    let theta_diff = (theta_b_mean - theta_a_mean).abs();

    println!(
        "  Theta: A_mean={:.4} B_mean={:.4} diff={:.4}rad={:.2}deg (expect {:.4}rad={:.2}deg)",
        theta_a_mean, theta_b_mean, theta_diff, theta_diff.to_degrees(),
        rotation_angle_rad, rotation_angle_deg
    );

    let radial_overlap = min_f(&radius_a) <= max_f(&radius_b) && min_f(&radius_b) <= max_f(&radius_a);
    let axial_overlap = min_f(&axial_a) <= max_f(&axial_b) && min_f(&axial_b) <= max_f(&axial_a);
    println!(
        "  Radial: A=[{:.4},{:.4}] B=[{:.4},{:.4}] overlap={}",
        min_f(&radius_a), max_f(&radius_a), min_f(&radius_b), max_f(&radius_b), radial_overlap
    );
    println!(
        "  Axial:  A=[{:.4},{:.4}] B=[{:.4},{:.4}] overlap={}",
        min_f(&axial_a), max_f(&axial_a), min_f(&axial_b), max_f(&axial_b), axial_overlap
    );

    // Step 5: Phase 1 full-face match
    let transform_fwd = |p: [Float; 3]| apply_rotation(p, rot_forward);
    let transform_rev = |p: [Float; 3]| apply_rotation(p, rot_backward);

    let match_fwd = full_face_match_transformed(&face_a, &face_b, transform_fwd, MATCH_TOL);
    let match_rev = full_face_match_transformed(&face_a, &face_b, transform_rev, MATCH_TOL);

    println!("  Phase 1 full-face: fwd={:?} rev={:?}", match_fwd.is_some(), match_rev.is_some());

    // Step 6: Phase 2 corner pre-check
    let corners_fwd = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_forward, MATCH_TOL);
    let corners_rev = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_backward, MATCH_TOL);

    println!("  Phase 2 corners: fwd={}/4 rev={}/4", corners_fwd, corners_rev);

    if corners_fwd < 2 && corners_rev < 2 {
        // Try relaxed tolerances
        for tol_factor in [10.0, 100.0, 1000.0] {
            let relaxed_tol = MATCH_TOL * tol_factor;
            let c_fwd = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_forward, relaxed_tol);
            let c_rev = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_backward, relaxed_tol);
            if c_fwd > corners_fwd || c_rev > corners_rev {
                println!("    (relaxed tol={:.0e}): fwd={} rev={}", relaxed_tol, c_fwd, c_rev);
            }
        }
        // Also try reversed direction: face_b corners on face_a
        let corners_rev_ba = count_rotated_corners_on_face(&face_b, &face_a, block_a, rot_forward, MATCH_TOL);
        let corners_fwd_ba = count_rotated_corners_on_face(&face_b, &face_a, block_a, rot_backward, MATCH_TOL);
        println!("  Phase 2 corners REVERSED (B->A): fwd={}/4 rev={}/4", corners_fwd_ba, corners_rev_ba);
    }

    // Step 7: Phase 2 intersection
    let block_a_rot_fwd = rotate_block(block_a, rot_forward);
    let block_a_rot_rev = rotate_block(block_a, rot_backward);

    let result_fwd = periodicity_check_with_points(&face_a, &face_b, &block_a_rot_fwd, block_b, MATCH_TOL);
    let result_rev = periodicity_check_with_points(&face_a, &face_b, &block_a_rot_rev, block_b, MATCH_TOL);

    match &result_fwd {
        Some((faces, points, splits)) => {
            println!("  Phase 2 intersection fwd: {} face pair(s), {} match pts, {} splits",
                faces.len(), points.len(), splits.len());
        }
        None => println!("  Phase 2 intersection fwd: none"),
    }
    match &result_rev {
        Some((faces, points, splits)) => {
            println!("  Phase 2 intersection rev: {} face pair(s), {} match pts, {} splits",
                faces.len(), points.len(), splits.len());
        }
        None => println!("  Phase 2 intersection rev: none"),
    }

    // Also try reversed intersection (B rotated onto A)
    if result_fwd.is_none() && result_rev.is_none() {
        let block_b_rot_fwd = rotate_block(block_b, rot_forward);
        let block_b_rot_rev = rotate_block(block_b, rot_backward);
        let result_ba_fwd = periodicity_check_with_points(&face_b, &face_a, &block_b_rot_fwd, block_a, MATCH_TOL);
        let result_ba_rev = periodicity_check_with_points(&face_b, &face_a, &block_b_rot_rev, block_a, MATCH_TOL);
        match &result_ba_fwd {
            Some((faces, points, splits)) => {
                println!("  Phase 2 intersection REVERSED fwd: {} face pair(s), {} match pts, {} splits",
                    faces.len(), points.len(), splits.len());
            }
            None => {}
        }
        match &result_ba_rev {
            Some((faces, points, splits)) => {
                println!("  Phase 2 intersection REVERSED rev: {} face pair(s), {} match pts, {} splits",
                    faces.len(), points.len(), splits.len());
            }
            None => {}
        }
        if result_ba_fwd.is_none() && result_ba_rev.is_none() {
            println!("  Phase 2 intersection REVERSED: none in either direction");
        }
    }

    // Step 8: Direct point comparison
    for (dir_label, rot_matrix) in [("Forward", rot_forward), ("Backward", rot_backward)] {
        let rotated_verts: Vec<[Float; 3]> = verts_a
            .iter()
            .map(|&v| apply_rotation(v, rot_matrix))
            .collect();

        let mut min_dist = Float::INFINITY;
        let mut max_min_dist: Float = 0.0;
        let mut sum_min_dist: Float = 0.0;
        let mut within_tol = 0usize;

        for rv in &rotated_verts {
            let mut closest = Float::INFINITY;
            for bv in verts_b {
                let d = ((rv[0] - bv[0]).powi(2) + (rv[1] - bv[1]).powi(2) + (rv[2] - bv[2]).powi(2)).sqrt();
                if d < closest { closest = d; }
            }
            if closest < min_dist { min_dist = closest; }
            if closest > max_min_dist { max_min_dist = closest; }
            sum_min_dist += closest;
            if closest < MATCH_TOL { within_tol += 1; }
        }

        let avg_min_dist = sum_min_dist / rotated_verts.len() as Float;
        println!(
            "  {} rotation: min={:.2e} max_min={:.2e} avg={:.2e} within_tol={}/{}",
            dir_label, min_dist, max_min_dist, avg_min_dist, within_tol, rotated_verts.len()
        );
    }

    // Step 9: Summary
    let phase1_ok = match_fwd.is_some() || match_rev.is_some();
    let phase2_intersect_ok = result_fwd.is_some() || result_rev.is_some();

    if phase1_ok {
        println!("  RESULT: MATCH via Phase 1 (full-face)");
    } else if phase2_intersect_ok {
        println!("  RESULT: MATCH via Phase 2 (intersection)");
    } else {
        let phase2_corners_ok = corners_fwd >= 2 || corners_rev >= 2;
        println!("  RESULT: NO MATCH — first failing step:");
        if !support_any {
            println!("    -> faces_support_any=false");
        } else if !radial_overlap || !axial_overlap {
            println!("    -> No radial/axial overlap");
        } else if (theta_diff - rotation_angle_rad).abs() > 0.1 {
            println!("    -> Theta separation doesn't match rotation angle");
        } else if !phase2_corners_ok {
            println!("    -> Corner pre-check failed (0-1 corners on target)");
        } else {
            println!("    -> Corner check passed but intersection failed");
        }
    }

    phase1_ok || phase2_intersect_ok
}

// ===========================================================================
// Main test
// ===========================================================================

#[test]
fn diagnose_face_pair() {
    let rotation_angle_deg = 360.0 / NBLADES as Float;
    let rotation_angle_rad = rotation_angle_deg.to_radians();
    let rot_forward = create_rotation_matrix(rotation_angle_rad, ROTATION_AXIS);
    let rot_backward = create_rotation_matrix(-rotation_angle_rad, ROTATION_AXIS);

    println!("\n{}", "=".repeat(70));
    println!("ROTATIONAL PERIODICITY DIAGNOSTIC");
    println!("{}", "=".repeat(70));

    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read mesh file");
    println!("[Step 1] Mesh loaded: {} blocks", blocks.len());

    let cases = diagnostic_cases();
    let mut total_pass = 0usize;
    let mut total_sub = 0usize;

    for case in &cases {
        println!("\n{}", "=".repeat(70));
        println!("{}", case.name);
        println!("{}", "=".repeat(70));

        for (bi, spec_b) in case.faces_b.iter().enumerate() {
            let label = if case.faces_b.len() == 1 {
                format!("Block {} <-> Block {}", case.face_a.0, spec_b.0)
            } else {
                format!("Block {} <-> Block {} (sub-face {})", case.face_a.0, spec_b.0, bi + 1)
            };

            let ok = diagnose_single_pair(
                &blocks,
                case.face_a,
                *spec_b,
                rotation_angle_deg,
                rotation_angle_rad,
                rot_forward,
                rot_backward,
                &label,
            );

            total_sub += 1;
            if ok { total_pass += 1; }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERALL: {}/{} sub-pairs matched", total_pass, total_sub);
    println!("{}", "=".repeat(70));
}
