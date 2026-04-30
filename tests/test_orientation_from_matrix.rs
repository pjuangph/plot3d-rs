//! Unit tests for `Orientation::index_from_permutation_matrix`.
//!
//! Verifies that all 8 canonical permutation matrices round-trip back to
//! their indices, and that non-canonical input yields `None`. This is the
//! gold-standard helper that lets glennht-core's connectivity.json parser
//! preserve the declared `permutation_matrix` and have plot3d-rs translate
//! it deterministically into a `permutation_index` — no brute force.

use plot3d::face_record::{Orientation, PERMUTATION_MATRICES};

#[test]
fn all_canonical_matrices_resolve_to_their_index() {
    for (expected_idx, matrix) in PERMUTATION_MATRICES.iter().enumerate() {
        let resolved = Orientation::index_from_permutation_matrix(*matrix);
        assert_eq!(
            resolved,
            Some(expected_idx as u8),
            "PERMUTATION_MATRICES[{}] = {:?} should resolve to index {}",
            expected_idx,
            matrix,
            expected_idx
        );
    }
}

#[test]
fn vspt_i_flip_resolves_to_index_one() {
    // VSPT's connectivity.json has periodic_faces[1] with
    // permutation_matrix [[-1, 0], [0, 1]] (u-flip), which is canonical
    // index 1. Asserting this prevents the iter-2 NaN regression.
    let m = [[-1i8, 0], [0, 1]];
    assert_eq!(Orientation::index_from_permutation_matrix(m), Some(1));
}

#[test]
fn identity_matrix_resolves_to_zero() {
    let m = [[1i8, 0], [0, 1]];
    assert_eq!(Orientation::index_from_permutation_matrix(m), Some(0));
}

#[test]
fn non_canonical_matrix_returns_none() {
    // [[2, 0], [0, 1]] is not a permutation matrix.
    let m = [[2i8, 0], [0, 1]];
    assert_eq!(Orientation::index_from_permutation_matrix(m), None);

    // Mixed garbage.
    let m = [[1i8, 1], [1, 1]];
    assert_eq!(Orientation::index_from_permutation_matrix(m), None);
}

#[test]
fn all_eight_indices_are_distinct() {
    use std::collections::HashSet;
    let set: HashSet<_> = PERMUTATION_MATRICES.iter().collect();
    assert_eq!(
        set.len(),
        8,
        "PERMUTATION_MATRICES must contain 8 distinct entries"
    );
}
