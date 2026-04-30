//! Integration test: translational `verify_translational_periodicity`
//! honors a declared `permutation_matrix` — same gold-standard contract
//! the rotational verifier upholds, applied to the CMC009-style case
//! (translational pitch in y, span in z).
//!
//! We synthesize two exactly-periodic blocks because the real CMC009
//! mesh is 1.2 GB / 593 blocks — far too large for a unit-test fixture.
//! The synthetic geometry isolates the matrix-honest decision logic
//! cleanly: declared matrix → canonical index → 1e-6 verification.

use plot3d::{
    verify_translational_periodicity, Block, FaceMatch, FaceRecord, Float, Orientation,
    OrientationPlane,
};

const TOL: Float = 1.0e-6;
const PITCH_Y: Float = 0.4; // CMC009-style cascade pitch
const SPAN_Z: Float = 0.05; // CMC009-style spanwise height

fn make_block(imax: usize, jmax: usize, kmax: usize, y_offset: Float, z_offset: Float) -> Block {
    let mut x = Vec::with_capacity(imax * jmax * kmax);
    let mut y = Vec::with_capacity(imax * jmax * kmax);
    let mut z = Vec::with_capacity(imax * jmax * kmax);
    for k in 0..kmax {
        for j in 0..jmax {
            for i in 0..imax {
                x.push(i as Float * 0.1);
                y.push(j as Float * 0.05 + y_offset);
                z.push(k as Float * 0.02 + z_offset);
            }
        }
    }
    Block::new(imax, jmax, kmax, x, y, z)
}

fn periodic_face(
    b1: (usize, [usize; 3], [usize; 3]),
    b2: (usize, [usize; 3], [usize; 3]),
    matrix: [[i8; 2]; 2],
) -> FaceMatch {
    let face_record = |block_index, lb: [usize; 3], ub: [usize; 3]| FaceRecord {
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
    };
    FaceMatch {
        block1: face_record(b1.0, b1.1, b1.2),
        block2: face_record(b2.0, b2.1, b2.2),
        points: Vec::new(),
        orientation: Some(Orientation {
            permutation_index: 0,
            plane: OrientationPlane::InPlane,
            permutation_matrix: Some(matrix),
        }),
    }
}

#[test]
fn translational_y_pitch_resolves_to_index_zero() {
    // Block A and block B at the same position (representing a periodic
    // pitch where face j=jmax-1 of A meets face j=0 of B' = A shifted
    // by PITCH_Y in y). Verifier checks the J=jmax-1 ↔ J=0 face match
    // with identity matrix.
    let imax = 9;
    let jmax = 5;
    let kmax = 3;
    let a = make_block(imax, jmax, kmax, 0.0, 0.0);
    // Block B is a copy of A — its J=0 face will pair with A's
    // J=jmax-1 face after a Δy = (jmax-1)*0.05 = PITCH_Y shift.
    // Auto-detect Δ from face centroids works because face A's J=jmax-1
    // and face B's J=0 have y-centroids differing by exactly PITCH_Y.
    let b = make_block(imax, jmax, kmax, PITCH_Y, 0.0);
    let blocks = vec![a, b];

    let pf = vec![periodic_face(
        (0, [0, jmax - 1, 0], [imax - 1, jmax - 1, kmax - 1]),
        (1, [0, 0, 0], [imax - 1, 0, kmax - 1]),
        [[1, 0], [0, 1]],
    )];

    let (verified, mismatched) =
        verify_translational_periodicity(&blocks, &pf, None, 'y', TOL);

    assert_eq!(mismatched.len(), 0, "exact translational match must verify");
    assert_eq!(verified.len(), 1);
    let o = verified[0].orientation.as_ref().unwrap();
    assert_eq!(o.permutation_index, 0);
    assert_eq!(o.permutation_matrix, Some([[1, 0], [0, 1]]));
}

#[test]
fn translational_z_span_resolves_to_index_zero() {
    // Span periodicity in z: same block, k=0 face ↔ k=kmax-1 face,
    // shifted by (kmax-1)*0.02 in z.
    let imax = 9;
    let jmax = 5;
    let kmax = 3;
    let a = make_block(imax, jmax, kmax, 0.0, 0.0);
    let blocks = vec![a];

    let pf = vec![periodic_face(
        (0, [0, 0, 0], [imax - 1, jmax - 1, 0]),
        (0, [0, 0, kmax - 1], [imax - 1, jmax - 1, kmax - 1]),
        [[1, 0], [0, 1]],
    )];

    let (verified, mismatched) =
        verify_translational_periodicity(&blocks, &pf, None, 'z', TOL);

    assert_eq!(mismatched.len(), 0);
    assert_eq!(verified.len(), 1);
    assert_eq!(
        verified[0].orientation.as_ref().unwrap().permutation_index,
        0
    );
}

#[test]
fn translational_wrong_axis_does_not_match() {
    // Same y-pitch geometry as the first test, but verifier called with
    // axis='z'. Δ_z is approximately 0 between the faces (no z-shift),
    // so the verifier should leave the match unverified — proving
    // verifiers are axis-honest and not cherry-picking a winning
    // permutation across axes.
    let imax = 9;
    let jmax = 5;
    let kmax = 3;
    let a = make_block(imax, jmax, kmax, 0.0, 0.0);
    let b = make_block(imax, jmax, kmax, PITCH_Y, 0.0);
    let blocks = vec![a, b];

    let pf = vec![periodic_face(
        (0, [0, jmax - 1, 0], [imax - 1, jmax - 1, kmax - 1]),
        (1, [0, 0, 0], [imax - 1, 0, kmax - 1]),
        [[1, 0], [0, 1]],
    )];

    let (verified, mismatched) =
        verify_translational_periodicity(&blocks, &pf, None, 'z', TOL);

    assert_eq!(verified.len(), 0, "axis mismatch must not silently pass");
    assert_eq!(mismatched.len(), 1);
    let _ = SPAN_Z; // silence warning when this constant is reserved for context
}
