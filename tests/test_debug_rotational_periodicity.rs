//! Diagnostic test for investigating why specific periodic face pairs are
//! missed by the rotational periodicity algorithm.
//!
//! Two-step workflow:
//!   1. Extract test blocks (one-time, slow — reads full 20GB mesh):
//!      cargo test --release --test test_debug_rotational_periodicity extract_test_blocks -- --nocapture
//!
//!   2. Debug the pipeline (fast — reads small extracted mesh):
//!      cargo test --release --test test_debug_rotational_periodicity debug_pipeline -- --nocapture

use plot3d::{
    apply_rotation, connectivity_fast, count_rotated_corners_on_face, create_face_from_diagonals,
    create_rotation_matrix, faces_support_any, faces_support_direction,
    full_face_match_transformed, periodicity_check_with_points, read_plot3d_binary,
    rotated_periodicity, rotate_block, to_radius, to_theta, write_plot3d, BinaryFormat, Endian,
    Float, FloatPrecision,
};

// ===========================================================================
// Configuration
// ===========================================================================

const FULL_MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

/// Small extracted mesh with just the test blocks (written by extract_test_blocks)
const EXTRACTED_MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/test_blocks.p3d";

const ROTATION_AXIS: char = 'x';
const NBLADES: usize = 44;
const MATCH_TOL: Float = 1e-6;

/// Face spec: (block_index, imin, jmin, kmin, imax, jmax, kmax)
/// The block_index refers to the ORIGINAL mesh block numbering.
type FaceSpec = (usize, usize, usize, usize, usize, usize, usize);

struct PairTestCase {
    name: &'static str,
    face_a: FaceSpec,
    faces_b: Vec<FaceSpec>,
}

fn test_cases() -> Vec<PairTestCase> {
    vec![
        // Pair 1: Baseline — already working cross-axis J/I pair
        PairTestCase {
            name: "Pair 1 (BASELINE) Block 3741 J-const <-> Block 4554 I-const",
            face_a: (3741, 0, 0, 12, 192, 0, 60),
            faces_b: vec![(4554, 36, 0, 0, 36, 24, 216)],
        },
        // Pair 2: Missing — cross-axis J/I pair
        PairTestCase {
            name: "Pair 2 (MISSING) Block 4109 J-const <-> Block 4562 I-const",
            face_a: (4109, 0, 0, 48, 192, 0, 72),
            faces_b: vec![(4562, 36, 0, 24, 36, 24, 216)],
        },
        // Pair 3: Missing — split face, face_a should match both B1 and B2
        PairTestCase {
            name: "Pair 3 (MISSING split) Block 4115 J-const <-> Block 4561 + Block 4565 I-const",
            face_a: (4115, 0, 108, 0, 24, 108, 48),
            faces_b: vec![
                (4561, 0, 0, 0, 0, 24, 24),
                (4565, 0, 0, 0, 0, 24, 24),
            ],
        },
        // Pair 4: Missing — same-axis I/I pair (not a cross-axis issue)
        PairTestCase {
            name: "Pair 4 (MISSING) Block 4216 I-const <-> Block 4544 I-const",
            face_a: (4216, 12, 264, 0, 12, 444, 28),
            faces_b: vec![(4544, 24, 12, 0, 24, 40, 180)],
        },
    ]
}

/// Collect all unique original block indices from test cases.
fn test_block_indices() -> Vec<usize> {
    let cases = test_cases();
    let mut indices: Vec<usize> = cases
        .iter()
        .flat_map(|c| {
            let mut v = vec![c.face_a.0];
            v.extend(c.faces_b.iter().map(|b| b.0));
            v
        })
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Map original block index → index in the extracted (small) mesh.
fn build_index_map(orig_indices: &[usize]) -> std::collections::HashMap<usize, usize> {
    orig_indices.iter().enumerate().map(|(new, &orig)| (orig, new)).collect()
}

/// Remap a FaceSpec from original block index to extracted mesh index.
fn remap_spec(spec: &FaceSpec, map: &std::collections::HashMap<usize, usize>) -> FaceSpec {
    let new_idx = *map.get(&spec.0).expect("block not in extracted mesh");
    (new_idx, spec.1, spec.2, spec.3, spec.4, spec.5, spec.6)
}

// ===========================================================================
// Diagnose a single (face_a, face_b) sub-pair
// ===========================================================================
fn diagnose_sub_pair(
    blocks: &[plot3d::Block],
    spec_a: FaceSpec,
    spec_b: FaceSpec,
    rot_forward: [[Float; 3]; 3],
    rot_backward: [[Float; 3]; 3],
    rotation_angle_rad: Float,
    label: &str,
) -> bool {
    let rotation_angle_deg = rotation_angle_rad.to_degrees();
    let (blk_a, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax) = spec_a;
    let (blk_b, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax) = spec_b;

    println!("\n  --- {} ---", label);

    let block_a = &blocks[blk_a];
    let block_b = &blocks[blk_b];

    println!(
        "  Block A [{}]: dims = {}x{}x{}",
        blk_a, block_a.imax, block_a.jmax, block_a.kmax
    );
    println!(
        "  Block B [{}]: dims = {}x{}x{}",
        blk_b, block_b.imax, block_b.jmax, block_b.kmax
    );

    // Step 1: Create faces
    let face_a =
        create_face_from_diagonals(block_a, a_imin, a_jmin, a_kmin, a_imax, a_jmax, a_kmax);
    let face_b =
        create_face_from_diagonals(block_b, b_imin, b_jmin, b_kmin, b_imax, b_jmax, b_kmax);

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

    // Step 2: Face compatibility
    let support_any = faces_support_any(&face_a, &face_b);
    let support_i = faces_support_direction(&face_a, &face_b, "i");
    let support_j = faces_support_direction(&face_a, &face_b, "j");
    let support_k = faces_support_direction(&face_a, &face_b, "k");
    println!(
        "  faces_support: any={} i={} j={} k={}",
        support_any, support_i, support_j, support_k
    );
    if !support_any {
        println!("  FAIL: faces_support_any=false, algorithm skips this pair");
        return false;
    }

    // Step 3: Cylindrical coordinate analysis
    let verts_a = face_a.vertices();
    let verts_b = face_b.vertices();

    let theta_a: Vec<Float> = verts_a
        .iter()
        .map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS))
        .collect();
    let theta_b: Vec<Float> = verts_b
        .iter()
        .map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS))
        .collect();
    let radius_a: Vec<Float> = verts_a
        .iter()
        .map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS))
        .collect();
    let radius_b: Vec<Float> = verts_b
        .iter()
        .map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS))
        .collect();
    let axial_a: Vec<Float> = verts_a.iter().map(|v| v[0]).collect();
    let axial_b: Vec<Float> = verts_b.iter().map(|v| v[0]).collect();

    let mean = |v: &[Float]| v.iter().sum::<Float>() / v.len() as Float;
    let min_f = |v: &[Float]| v.iter().cloned().fold(Float::INFINITY, Float::min);
    let max_f = |v: &[Float]| v.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

    let theta_a_mean = mean(&theta_a);
    let theta_b_mean = mean(&theta_b);
    let theta_diff = (theta_b_mean - theta_a_mean).abs();
    let theta_tol = rotation_angle_rad.abs() * 0.15 + 0.05;

    println!(
        "  Theta: A_mean={:.4} B_mean={:.4} diff={:.4}rad={:.2}deg (expect {:.4}rad={:.2}deg)",
        theta_a_mean, theta_b_mean, theta_diff, theta_diff.to_degrees(),
        rotation_angle_rad, rotation_angle_deg
    );
    println!(
        "  Theta bucketing tol={:.4}rad, |diff-angle|={:.4} {}",
        theta_tol,
        (theta_diff - rotation_angle_rad).abs(),
        if (theta_diff - rotation_angle_rad).abs() < theta_tol { "PASS" } else { "FAIL" }
    );

    let radial_overlap =
        min_f(&radius_a) <= max_f(&radius_b) && min_f(&radius_b) <= max_f(&radius_a);
    let axial_overlap =
        min_f(&axial_a) <= max_f(&axial_b) && min_f(&axial_b) <= max_f(&axial_a);
    println!("  Radial overlap: {} Axial overlap: {}", radial_overlap, axial_overlap);

    // Step 4: Phase 1 full-face match
    let transform_fwd = |p: [Float; 3]| apply_rotation(p, rot_forward);
    let transform_rev = |p: [Float; 3]| apply_rotation(p, rot_backward);

    let match_fwd = full_face_match_transformed(&face_a, &face_b, transform_fwd, MATCH_TOL);
    let match_rev = full_face_match_transformed(&face_a, &face_b, transform_rev, MATCH_TOL);

    println!("  Phase 1 full-face: fwd={:?} rev={:?}", match_fwd.is_some(), match_rev.is_some());

    // Step 5: Phase 2 corner pre-check
    let corners_fwd = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_forward, MATCH_TOL);
    let corners_rev = count_rotated_corners_on_face(&face_a, &face_b, block_b, rot_backward, MATCH_TOL);

    println!("  Phase 2 corners: fwd={}/4 rev={}/4", corners_fwd, corners_rev);

    // Step 6: Phase 2 intersection
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

    // Step 7: Summary
    let phase1_ok = match_fwd.is_some() || match_rev.is_some();
    let phase2_intersect_ok = result_fwd.is_some() || result_rev.is_some();

    if phase1_ok {
        println!("  RESULT: MATCH via Phase 1 (full-face)");
    } else if phase2_intersect_ok {
        println!("  RESULT: MATCH via Phase 2 (intersection)");
    } else {
        println!("  RESULT: NO MATCH");
    }

    phase1_ok || phase2_intersect_ok
}

// ===========================================================================
// TEST 1: Extract test blocks from full mesh (run once, slow)
// ===========================================================================
#[test]
fn extract_test_blocks() {
    let orig_indices = test_block_indices();
    println!("\nExtracting blocks {:?} from full mesh...", orig_indices);

    let all_blocks = read_plot3d_binary(
        FULL_MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read full mesh");
    println!("Full mesh loaded: {} blocks", all_blocks.len());

    let extracted: Vec<_> = orig_indices
        .iter()
        .map(|&idx| {
            assert!(idx < all_blocks.len(), "Block {} out of range", idx);
            all_blocks[idx].clone()
        })
        .collect();

    println!("Extracted {} blocks:", extracted.len());
    for (new_idx, &orig_idx) in orig_indices.iter().enumerate() {
        let b = &extracted[new_idx];
        println!("  new[{}] = orig[{}]: {}x{}x{}", new_idx, orig_idx, b.imax, b.jmax, b.kmax);
    }

    write_plot3d(
        EXTRACTED_MESH_FILE,
        &extracted,
        true,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to write extracted mesh");

    println!("Written to: {}", EXTRACTED_MESH_FILE);

    // Verify by reading back
    let readback = read_plot3d_binary(
        EXTRACTED_MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read back extracted mesh");
    assert_eq!(readback.len(), extracted.len());
    println!("Verified: read back {} blocks", readback.len());
}

// ===========================================================================
// TEST 2: Debug the pipeline (fast — uses extracted blocks)
// ===========================================================================
#[test]
fn debug_pipeline() {
    let rotation_angle_deg: Float = 360.0 / NBLADES as Float;
    let rotation_angle_rad: Float = rotation_angle_deg.to_radians();
    let rot_forward = create_rotation_matrix(rotation_angle_rad, ROTATION_AXIS);
    let rot_backward = create_rotation_matrix(-rotation_angle_rad, ROTATION_AXIS);

    println!("\n{}", "=".repeat(70));
    println!("DEBUG PIPELINE (extracted blocks)");
    println!("{}", "=".repeat(70));

    // Load extracted blocks
    let blocks = read_plot3d_binary(
        EXTRACTED_MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect(&format!("Failed to read {}. Run extract_test_blocks first.", EXTRACTED_MESH_FILE));
    println!("Loaded {} blocks from extracted mesh", blocks.len());

    let orig_indices = test_block_indices();
    let index_map = build_index_map(&orig_indices);
    let cases = test_cases();

    // Show block mapping
    for (new_idx, &orig_idx) in orig_indices.iter().enumerate() {
        let b = &blocks[new_idx];
        println!("  Block new[{}] = orig[{}]: {}x{}x{}", new_idx, orig_idx, b.imax, b.jmax, b.kmax);
    }

    // ── Per-pair diagnostics (on extracted blocks) ────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("PER-PAIR DIAGNOSTICS");
    println!("{}", "=".repeat(70));

    for case in &cases {
        println!("\n{}", "=".repeat(70));
        println!("{}", case.name);
        println!("{}", "=".repeat(70));

        let remapped_a = remap_spec(&case.face_a, &index_map);

        for (bi, spec_b) in case.faces_b.iter().enumerate() {
            let remapped_b = remap_spec(spec_b, &index_map);
            let label = if case.faces_b.len() == 1 {
                format!("A <-> B")
            } else {
                format!("A <-> B{}", bi + 1)
            };

            diagnose_sub_pair(
                &blocks,
                remapped_a,
                remapped_b,
                rot_forward,
                rot_backward,
                rotation_angle_rad,
                &label,
            );
        }
    }

    // ── Pipeline on extracted blocks ──────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("PIPELINE ON EXTRACTED BLOCKS");
    println!("{}", "=".repeat(70));

    println!("\nRunning connectivity_fast on {} blocks...", blocks.len());
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!(
        "  {} face matches, {} non-connected faces",
        face_matches.len(), outer_faces.len()
    );

    // Show outer faces for each block
    println!("\n  Outer faces per block:");
    for (new_idx, &orig_idx) in orig_indices.iter().enumerate() {
        let blk_faces: Vec<_> = outer_faces.iter().filter(|f| f.block_index == new_idx).collect();
        for f in &blk_faces {
            println!(
                "    Block new[{}]=orig[{}] outer: ({},{},{},{},{},{})",
                new_idx, orig_idx, f.il, f.jl, f.kl, f.ih, f.jh, f.kh
            );
        }
        if blk_faces.is_empty() {
            println!("    Block new[{}]=orig[{}]: NO outer faces", new_idx, orig_idx);
        }
    }

    // Run rotated_periodicity with reduce_mesh=false (blocks are small, no need for GCD)
    println!("\nRunning rotated_periodicity (reduce_mesh=false)...");
    let (periodic_faces, remaining_outer) = rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle_deg,
        ROTATION_AXIS,
        false,
    );
    println!(
        "  {} periodic face pairs, {} remaining",
        periodic_faces.len(), remaining_outer.len()
    );

    // Show all periodic matches
    println!("\n  All periodic matches found:");
    for m in &periodic_faces {
        let orig_b1 = orig_indices.get(m.block1.block_index).copied().unwrap_or(m.block1.block_index);
        let orig_b2 = orig_indices.get(m.block2.block_index).copied().unwrap_or(m.block2.block_index);
        println!(
            "    orig[{}] ({},{},{},{},{},{}) <-> orig[{}] ({},{},{},{},{},{})",
            orig_b1,
            m.block1.il, m.block1.jl, m.block1.kl,
            m.block1.ih, m.block1.jh, m.block1.kh,
            orig_b2,
            m.block2.il, m.block2.jl, m.block2.kl,
            m.block2.ih, m.block2.jh, m.block2.kh,
        );
    }

    // Show remaining outer faces
    println!("\n  Remaining outer faces:");
    for r in &remaining_outer {
        let orig = orig_indices.get(r.block_index).copied().unwrap_or(r.block_index);
        println!(
            "    orig[{}] remaining: ({},{},{},{},{},{})",
            orig, r.il, r.jl, r.kl, r.ih, r.jh, r.kh
        );
    }

    // Check each test case
    println!("\n  Test case results:");
    for case in &cases {
        let remapped_a = remap_spec(&case.face_a, &index_map);
        for spec_b in &case.faces_b {
            let remapped_b = remap_spec(spec_b, &index_map);
            let found = periodic_faces.iter().any(|pp| {
                let b1 = pp.block1.block_index;
                let b2 = pp.block2.block_index;
                (b1 == remapped_a.0 && b2 == remapped_b.0)
                    || (b1 == remapped_b.0 && b2 == remapped_a.0)
            });
            let orig_a = case.face_a.0;
            let orig_b = spec_b.0;
            println!(
                "    {} orig[{}]<=>orig[{}]: {}",
                case.name, orig_a, orig_b,
                if found { "FOUND" } else { "NOT FOUND" },
            );
        }
    }

    println!("\nDone.");
}

// ===========================================================================
// TEST 3: Full mesh with GCD reduction
// ===========================================================================
#[test]
fn debug_full_mesh_gcd() {
    let rotation_angle_deg: Float = 360.0 / NBLADES as Float;

    println!("\n{}", "=".repeat(70));
    println!("FULL MESH WITH GCD REDUCTION");
    println!("{}", "=".repeat(70));

    println!("Loading full mesh (this takes a while)...");
    let blocks = read_plot3d_binary(
        FULL_MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read full mesh");
    println!("Loaded {} blocks", blocks.len());

    println!("Running connectivity_fast...");
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!(
        "  {} face matches, {} non-connected faces",
        face_matches.len(),
        outer_faces.len()
    );

    println!("Running rotated_periodicity (reduce_mesh=true, GCD)...");
    let (periodic_faces, remaining_outer) = rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle_deg,
        ROTATION_AXIS,
        true,
    );
    println!(
        "  {} periodic face pairs, {} remaining outer faces",
        periodic_faces.len(),
        remaining_outer.len()
    );

    // Check test cases
    let cases = test_cases();
    println!("\n  Test case results:");
    for case in &cases {
        for spec_b in &case.faces_b {
            let found = periodic_faces.iter().any(|pp| {
                let b1 = pp.block1.block_index;
                let b2 = pp.block2.block_index;
                (b1 == case.face_a.0 && b2 == spec_b.0)
                    || (b1 == spec_b.0 && b2 == case.face_a.0)
            });
            println!(
                "    {} orig[{}]<=>orig[{}]: {}",
                case.name,
                case.face_a.0,
                spec_b.0,
                if found { "FOUND" } else { "NOT FOUND" },
            );
        }
    }

    println!("\nDone.");
}
