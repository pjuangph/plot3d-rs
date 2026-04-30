//! Integration test: VSPT rotational periodicity must verify with
//! declared `permutation_matrix` — no brute-force fallback.
//!
//! Loads the real VSPT mesh and the 3 `periodic_faces` from VSPT's
//! `connectivity.json` (verbatim). For each, we assert the verifier:
//!
//! 1. Verifies geometry to 2e-6 (VSPT's mesh has ~1.87e-6 worst-cell
//!    periodic mismatch baked in by the f32 source-data quantisation;
//!    the production tolerance was relaxed from 1e-6 to 2e-6 on
//!    2026-04-30 to accept this).
//! 2. Returns the canonical `permutation_index` derived from the
//!    declared `permutation_matrix`.
//!
//! The critical case is `periodic_faces[1]`: a cross-block rotational
//! seam between blocks 0 and 1 with `permutation_matrix [[-1, 0], [0, 1]]`
//! (canonical index 1, i.e. u-flip). Before the gold-standard rework,
//! glennht-core's parser dropped the matrix and `try_all_permutations`
//! could not recover the orientation at 1e-6, leading to wrong cell pairs
//! at the seam and the iter-2 NaN cascade. This test pins the correct
//! behaviour.

use plot3d::{
    read_plot3d_ascii, read_plot3d_binary, verify_periodicity, BinaryFormat, Endian, FaceMatch,
    FaceRecord, Float, FloatPrecision, Orientation, OrientationPlane,
};

const VSPT_THETA_RAD: Float = 0.11423973285781065;
const TOL: Float = 2.0e-6;

/// Load VSPT blocks. Prefers the production f64 ASCII mesh
/// (`~/glennht_comparison/mesh/finalmesh-ASCII.xyz`) when available;
/// falls back to the smaller f32 binary fixture in
/// `tests/data/vspt_mesh_scaled.xyz`. Returns `None` if neither
/// mesh is available so CI without the comparison harness can still
/// run unrelated tests.
fn load_vspt_blocks() -> Option<Vec<plot3d::Block>> {
    let ascii_path = "/Users/pjuangph/glennht_comparison/mesh/finalmesh-ASCII.xyz";
    if std::path::Path::new(ascii_path).exists() {
        return Some(read_plot3d_ascii(ascii_path).expect("read finalmesh-ASCII.xyz"));
    }
    let bin_path = "tests/data/vspt_mesh_scaled.xyz";
    if std::path::Path::new(bin_path).exists() {
        return Some(
            read_plot3d_binary(bin_path, BinaryFormat::Raw, FloatPrecision::F32, Endian::Little)
                .expect("read vspt_mesh_scaled.xyz"),
        );
    }
    None
}

fn face_record(block_index: usize, lb: [usize; 3], ub: [usize; 3]) -> FaceRecord {
    FaceRecord {
        block_index,
        il: lb[0],
        jl: lb[1],
        kl: lb[2],
        ih: ub[0],
        jh: ub[1],
        kh: ub[2],
        id: None,
        u_physical: None,
        v_physical: None,
    }
}

fn periodic_face(
    b1: (usize, [usize; 3], [usize; 3]),
    b2: (usize, [usize; 3], [usize; 3]),
    matrix: [[i8; 2]; 2],
) -> FaceMatch {
    FaceMatch {
        block1: face_record(b1.0, b1.1, b1.2),
        block2: face_record(b2.0, b2.1, b2.2),
        points: Vec::new(),
        orientation: Some(Orientation {
            // permutation_index = 0 is a sentinel here: the verifier
            // MUST consult permutation_matrix and resolve the canonical
            // index via Orientation::index_from_permutation_matrix.
            permutation_index: 0,
            plane: OrientationPlane::InPlane,
            permutation_matrix: Some(matrix),
        }),
    }
}

#[test]
fn vspt_periodic_faces_verify_via_matrix() {
    let Some(blocks) = load_vspt_blocks() else {
        eprintln!("skipping: no VSPT mesh available");
        return;
    };
    assert_eq!(blocks.len(), 2, "VSPT mesh has 2 blocks");

    // Verbatim from /Users/pjuangph/glennht_comparison/mesh/connectivity.json
    let pf = vec![
        // [0] blk1 self k-periodic, identity matrix
        periodic_face(
            (1, [0, 0, 0], [40, 100, 0]),
            (1, [0, 0, 52], [40, 100, 52]),
            [[1, 0], [0, 1]],
        ),
        // [1] cross-block rotational seam — THE CRITICAL CASE.
        // permutation_matrix [[-1, 0], [0, 1]] = canonical index 1 (u-flip).
        // Note block2's lb=[168,0,52], ub=[40,0,52] — descending I order
        // is itself a flag of the i-flip; the matrix is the authoritative
        // declaration.
        periodic_face(
            (0, [0, 0, 32], [128, 100, 32]),
            (1, [168, 0, 52], [40, 100, 52]),
            [[-1, 0], [0, 1]],
        ),
        // [2] blk1 self k-periodic, identity matrix
        periodic_face(
            (1, [168, 0, 0], [268, 100, 0]),
            (1, [168, 0, 52], [268, 100, 52]),
            [[1, 0], [0, 1]],
        ),
    ];

    let (verified, mismatched) = verify_periodicity(&blocks, &pf, VSPT_THETA_RAD, 'x', TOL);

    assert_eq!(
        mismatched.len(),
        0,
        "all 3 VSPT periodic_faces must verify at 2e-6; mismatched={}",
        mismatched.len()
    );
    assert_eq!(verified.len(), 3, "expected 3 verified periodic_faces");

    // Verify each carries its canonical permutation_index.
    let resolved: Vec<u8> = verified
        .iter()
        .map(|fm| fm.orientation.as_ref().unwrap().permutation_index)
        .collect();
    assert_eq!(
        resolved,
        vec![0, 1, 0],
        "periodic_faces[1] must resolve to canonical index 1 (u-flip)"
    );

    // The preserved matrices should pass through unchanged.
    let matrices: Vec<[[i8; 2]; 2]> = verified
        .iter()
        .map(|fm| fm.orientation.as_ref().unwrap().permutation_matrix.unwrap())
        .collect();
    assert_eq!(matrices[0], [[1, 0], [0, 1]]);
    assert_eq!(matrices[1], [[-1, 0], [0, 1]]);
    assert_eq!(matrices[2], [[1, 0], [0, 1]]);
}

#[test]
fn vspt_wrong_matrix_does_not_silently_match() {
    // Declaring an identity matrix on the cross-block rotational seam
    // (which actually needs the i-flip) MUST fail verification rather
    // than silently passing — confirms the verifier is matrix-honest
    // and not rounding through brute-force.
    let Some(blocks) = load_vspt_blocks() else {
        eprintln!("skipping: no VSPT mesh available");
        return;
    };
    let bad = vec![periodic_face(
        (0, [0, 0, 32], [128, 100, 32]),
        (1, [168, 0, 52], [40, 100, 52]),
        [[1, 0], [0, 1]], // wrong — should be [[-1, 0], [0, 1]]
    )];

    let (verified, mismatched) =
        verify_periodicity(&blocks, &bad, VSPT_THETA_RAD, 'x', TOL);

    assert_eq!(verified.len(), 0, "wrong matrix must not silently pass");
    assert_eq!(mismatched.len(), 1, "wrong matrix must be reported as mismatched");
}
