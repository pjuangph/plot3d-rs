//! Diagnostic test for investigating why specific face pairs fail (or succeed)
//! in the rotational periodicity matching pipeline.
//!
//! Usage (release build recommended for large meshes):
//!   cargo test --release --test test_debug_periodicity -- --nocapture
//!
//! Edit the constants below to test different block/face combinations.

use plot3d::{
    apply_rotation, count_rotated_corners_on_face, create_face_from_diagonals,
    create_rotation_matrix, faces_support_any, faces_support_direction,
    full_face_match_transformed, periodicity_check_with_points, read_plot3d_binary, rotate_block,
    to_radius, to_theta, BinaryFormat, Endian, Float, FloatPrecision,
};

// ===========================================================================
// USER CONFIGURATION — edit these values for each investigation
// ===========================================================================

/// Path to the mesh file (binary format, much faster than ASCII for large meshes).
const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

/// Rotation axis ('x', 'y', or 'z').
const ROTATION_AXIS: char = 'x';

/// Number of blades (used to compute rotation angle = 360 / nblades).
const NBLADES: usize = 44;

/// Tolerance for node coincidence checks.
const MATCH_TOL: Float = 1e-6;

/// Face A: (block_index, imin, jmin, kmin, imax, jmax, kmax)
/// Block index is 0-based (same as JSON).
/// Non-connected 4436 — Block 3741, J-constant (block dims: 193x13x61)
const FACE_A: (usize, usize, usize, usize, usize, usize, usize) = (3741, 0, 0, 12, 192, 0, 60);

/// Face B: (block_index, imin, jmin, kmin, imax, jmax, kmax)
/// Block index is 0-based (same as JSON).
/// Non-connected 3567 — Block 4554, I-constant (block dims: 37x25x217)
const FACE_B: (usize, usize, usize, usize, usize, usize, usize) = (4554, 36, 0, 0, 36, 24, 216);

// ===========================================================================

#[test]
fn diagnose_face_pair() {
    let rotation_angle_deg = 360.0 / NBLADES as Float;
    let rotation_angle_rad = rotation_angle_deg.to_radians();

    // ── Step 1: Read mesh ──────────────────────────────────────────────
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
    println!("\n[Step 1] Mesh loaded: {} blocks", blocks.len());

    let (blk_a_idx, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax) = FACE_A;
    let (blk_b_idx, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax) = FACE_B;

    assert!(
        blk_a_idx < blocks.len(),
        "Block A index {} out of range (have {} blocks)",
        blk_a_idx,
        blocks.len()
    );
    assert!(
        blk_b_idx < blocks.len(),
        "Block B index {} out of range (have {} blocks)",
        blk_b_idx,
        blocks.len()
    );

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

    // ── Step 2: Create faces ───────────────────────────────────────────
    let face_a = create_face_from_diagonals(block_a, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax);
    let face_b = create_face_from_diagonals(block_b, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax);

    println!("\n[Step 2] Faces created");
    println!(
        "  Face A: block={}, diag=({},{},{})→({},{},{}), vertices={}, const_type={}",
        blk_a_idx,
        a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax,
        face_a.vertices().len(),
        face_a.const_type()
    );
    println!(
        "  Face B: block={}, diag=({},{},{})→({},{},{}), vertices={}, const_type={}",
        blk_b_idx,
        b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax,
        face_b.vertices().len(),
        face_b.const_type()
    );

    if face_a.const_type() == -1 {
        println!("  ⚠ Face A is NOT constant along any axis (const_type=-1)");
    }
    if face_b.const_type() == -1 {
        println!("  ⚠ Face B is NOT constant along any axis (const_type=-1)");
    }

    // ── Step 3: Face compatibility (faces_support_any) ─────────────────
    let support_i = faces_support_direction(&face_a, &face_b, "i");
    let support_j = faces_support_direction(&face_a, &face_b, "j");
    let support_k = faces_support_direction(&face_a, &face_b, "k");
    let support_any = faces_support_any(&face_a, &face_b);

    println!("\n[Step 3] Face compatibility (faces_support_*)");
    println!("  I-constant: {}", support_i);
    println!("  J-constant: {}", support_j);
    println!("  K-constant: {}", support_k);
    println!("  Any:        {}", support_any);

    if !support_any {
        println!("  ✗ FAIL: faces_support_any=false — the algorithm skips this pair!");
        println!("    Face A: imin={} imax={} jmin={} jmax={} kmin={} kmax={}",
            face_a.imin(), face_a.imax(), face_a.jmin(), face_a.jmax(), face_a.kmin(), face_a.kmax());
        println!("    Face B: imin={} imax={} jmin={} jmax={} kmin={} kmax={}",
            face_b.imin(), face_b.imax(), face_b.jmin(), face_b.jmax(), face_b.kmin(), face_b.kmax());
    }

    // ── Step 4: Cylindrical coordinate analysis ────────────────────────
    println!("\n[Step 4] Cylindrical coordinate analysis (axis='{}')", ROTATION_AXIS);
    println!("  Rotation angle = {:.4} deg = {:.6} rad", rotation_angle_deg, rotation_angle_rad);

    let verts_a = face_a.vertices();
    let verts_b = face_b.vertices();

    let theta_a: Vec<Float> = verts_a.iter().map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let theta_b: Vec<Float> = verts_b.iter().map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS)).collect();

    let radius_a: Vec<Float> = verts_a.iter().map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS)).collect();
    let radius_b: Vec<Float> = verts_b.iter().map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS)).collect();

    let axial_a: Vec<Float> = verts_a.iter().map(|v| match ROTATION_AXIS {
        'x' => v[0], 'y' => v[1], _ => v[2],
    }).collect();
    let axial_b: Vec<Float> = verts_b.iter().map(|v| match ROTATION_AXIS {
        'x' => v[0], 'y' => v[1], _ => v[2],
    }).collect();

    let mean = |v: &[Float]| v.iter().sum::<Float>() / v.len() as Float;
    let min_f = |v: &[Float]| v.iter().cloned().fold(Float::INFINITY, Float::min);
    let max_f = |v: &[Float]| v.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

    let theta_a_mean = mean(&theta_a);
    let theta_b_mean = mean(&theta_b);
    let theta_diff = (theta_b_mean - theta_a_mean).abs();

    println!("  Face A: theta=[{:.4}, {:.4}] mean={:.4}, radius=[{:.4}, {:.4}], axial=[{:.4}, {:.4}]",
        min_f(&theta_a), max_f(&theta_a), theta_a_mean,
        min_f(&radius_a), max_f(&radius_a),
        min_f(&axial_a), max_f(&axial_a));
    println!("  Face B: theta=[{:.4}, {:.4}] mean={:.4}, radius=[{:.4}, {:.4}], axial=[{:.4}, {:.4}]",
        min_f(&theta_b), max_f(&theta_b), theta_b_mean,
        min_f(&radius_b), max_f(&radius_b),
        min_f(&axial_b), max_f(&axial_b));
    println!("  Theta separation:  {:.6} rad = {:.4} deg", theta_diff, theta_diff.to_degrees());
    println!("  Expected (angle):  {:.6} rad = {:.4} deg", rotation_angle_rad, rotation_angle_deg);
    println!("  Difference:        {:.6} rad", (theta_diff - rotation_angle_rad).abs());

    // Check radial/axial overlap
    let radial_overlap = min_f(&radius_a) <= max_f(&radius_b) && min_f(&radius_b) <= max_f(&radius_a);
    let axial_overlap = min_f(&axial_a) <= max_f(&axial_b) && min_f(&axial_b) <= max_f(&axial_a);
    println!("  Radial overlap:    {}", radial_overlap);
    println!("  Axial overlap:     {}", axial_overlap);

    // ── Step 5: Phase 1 — full face match ──────────────────────────────
    println!("\n[Step 5] Phase 1: Full-face match (full_face_match_transformed)");

    let rot_forward = create_rotation_matrix(rotation_angle_rad, ROTATION_AXIS);
    let rot_backward = create_rotation_matrix(-rotation_angle_rad, ROTATION_AXIS);

    let transform_fwd = |p: [Float; 3]| apply_rotation(p, rot_forward);
    let transform_rev = |p: [Float; 3]| apply_rotation(p, rot_backward);

    let match_fwd = full_face_match_transformed(&face_a, &face_b, transform_fwd, MATCH_TOL);
    let match_rev = full_face_match_transformed(&face_a, &face_b, transform_rev, MATCH_TOL);

    println!("  Forward rotation (+angle): {:?}", match_fwd);
    println!("  Backward rotation (-angle): {:?}", match_rev);

    if match_fwd.is_some() {
        println!("  ✓ Phase 1 MATCH with forward rotation");
    } else if match_rev.is_some() {
        println!("  ✓ Phase 1 MATCH with backward rotation");
    } else {
        println!("  ✗ Phase 1: No full-face match in either direction");
    }

    // ── Step 6: Phase 2 — corner pre-check ─────────────────────────────
    println!("\n[Step 6] Phase 2: Corner pre-check (count_rotated_corners_on_face)");

    let corners_fwd = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_forward, MATCH_TOL);
    let corners_rev = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_backward, MATCH_TOL);

    println!("  Forward rotation: {} / 4 corners hit", corners_fwd);
    println!("  Backward rotation: {} / 4 corners hit", corners_rev);

    if corners_fwd < 2 && corners_rev < 2 {
        println!("  ✗ Neither direction has >= 2 corners — Phase 2 would skip this pair");
    }

    // Also try with relaxed tolerances to see if it's a tolerance issue
    for tol_factor in [10.0, 100.0, 1000.0] {
        let relaxed_tol = MATCH_TOL * tol_factor;
        let c_fwd = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_forward, relaxed_tol);
        let c_rev = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_backward, relaxed_tol);
        if c_fwd > corners_fwd || c_rev > corners_rev {
            println!("  (relaxed tol={:.0e}): fwd={} rev={} corners", relaxed_tol, c_fwd, c_rev);
        }
    }

    // ── Step 7: Phase 2 — rotated block intersection ───────────────────
    println!("\n[Step 7] Phase 2: Rotated block intersection (periodicity_check_with_points)");

    let block_a_rot_fwd = rotate_block(block_a, rot_forward);
    let block_a_rot_rev = rotate_block(block_a, rot_backward);

    let result_fwd = periodicity_check_with_points(&face_a, &face_b, &block_a_rot_fwd, block_b, MATCH_TOL);
    let result_rev = periodicity_check_with_points(&face_a, &face_b, &block_a_rot_rev, block_b, MATCH_TOL);

    match &result_fwd {
        Some((faces, points, splits)) => {
            println!("  Forward rotation: MATCH — {} face pair(s), {} match points, {} splits",
                faces.len(), points.len(), splits.len());
        }
        None => println!("  Forward rotation: no intersection"),
    }

    match &result_rev {
        Some((faces, points, splits)) => {
            println!("  Backward rotation: MATCH — {} face pair(s), {} match points, {} splits",
                faces.len(), points.len(), splits.len());
        }
        None => println!("  Backward rotation: no intersection"),
    }

    if result_fwd.is_none() && result_rev.is_none() {
        println!("  ✗ No intersection found in either direction");
    }

    // ── Step 8: Direct point comparison ────────────────────────────────
    println!("\n[Step 8] Direct point comparison (rotated face_a vertices vs face_b vertices)");

    for (label, rot_matrix) in [("Forward", rot_forward), ("Backward", rot_backward)] {
        let rotated_verts: Vec<[Float; 3]> = verts_a
            .iter()
            .map(|&v| apply_rotation(v, rot_matrix))
            .collect();

        let mut min_dist = Float::INFINITY;
        let mut max_min_dist: Float = 0.0;
        let mut sum_min_dist: Float = 0.0;

        for rv in &rotated_verts {
            let mut closest = Float::INFINITY;
            for bv in verts_b {
                let d = ((rv[0] - bv[0]).powi(2)
                    + (rv[1] - bv[1]).powi(2)
                    + (rv[2] - bv[2]).powi(2))
                .sqrt();
                if d < closest {
                    closest = d;
                }
            }
            if closest < min_dist {
                min_dist = closest;
            }
            if closest > max_min_dist {
                max_min_dist = closest;
            }
            sum_min_dist += closest;
        }

        let avg_min_dist = sum_min_dist / rotated_verts.len() as Float;
        println!(
            "  {} rotation: min_dist={:.2e}, max_min_dist={:.2e}, avg_min_dist={:.2e}",
            label, min_dist, max_min_dist, avg_min_dist
        );
    }

    // ── Step 9: Summary ────────────────────────────────────────────────
    println!("\n[Summary]");
    let phase1_ok = match_fwd.is_some() || match_rev.is_some();
    let phase2_corners_ok = corners_fwd >= 2 || corners_rev >= 2;
    let phase2_intersect_ok = result_fwd.is_some() || result_rev.is_some();

    if phase1_ok {
        println!("  MATCH FOUND via Phase 1 (full-face match)");
    } else if phase2_intersect_ok {
        println!("  MATCH FOUND via Phase 2 (split-face intersection)");
    } else {
        println!("  NO MATCH — first failing step:");
        if !support_any {
            println!("    → faces_support_any=false (faces not constant on same axis)");
        } else if !radial_overlap || !axial_overlap {
            println!("    → No radial/axial overlap (faces in different spatial regions)");
        } else if (theta_diff - rotation_angle_rad).abs() > 0.1 {
            println!("    → Theta separation ({:.4} rad) doesn't match rotation angle ({:.4} rad)",
                theta_diff, rotation_angle_rad);
        } else if !phase2_corners_ok {
            println!("    → Corner pre-check failed (0-1 corners land on opposite block)");
            println!("      This may indicate mesh misalignment or tolerance issues");
        } else {
            println!("    → Corner check passed but intersection failed");
            println!("      This may indicate partial overlap or GCD reduction issues");
        }
    }

    println!();
}
