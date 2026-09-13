//! Node-for-node face certification, proximity-grid coincidence, full
//! resolution re-validation of reduced-grid discovery, and tolerance
//! parameterisation.
//!
//! Each fixture is built so that the *old* behaviour is demonstrably wrong
//! on it: corners that agree while an interior node does not, coincident
//! nodes that straddle a rounding-bin boundary, a reduced grid that samples
//! only the agreeing nodes.

use plot3d::{
    certify_correspondence, connectivity, connectivity_fast, connectivity_fast_with_tol,
    corner_match, create_face_from_diagonals, full_face_match, get_outer_faces,
    get_outer_faces_with_tol, rotated_periodicity, rotated_periodicity_with_tol,
    translational_periodicity, translational_periodicity_with_tols, Block, Face, FaceRecord,
    Float, MappingFailure, Patch, TranslationalTolerances,
};

/// Structured block with `dims` nodes at `origin + d * (i, j, k)`, passed
/// through `edit(i, j, k, xyz)` so a fixture can displace chosen nodes.
fn grid_block(
    origin: [Float; 3],
    d: [Float; 3],
    dims: [usize; 3],
    edit: impl Fn(usize, usize, usize, [Float; 3]) -> [Float; 3],
) -> Block {
    let [ni, nj, nk] = dims;
    let n = ni * nj * nk;
    let (mut x, mut y, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let p = edit(
                    i,
                    j,
                    k,
                    [
                        origin[0] + d[0] * i as Float,
                        origin[1] + d[1] * j as Float,
                        origin[2] + d[2] * k as Float,
                    ],
                );
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }
        }
    }
    Block::new(ni, nj, nk, x, y, z)
}

fn imax_face(block: &Block, block_index: usize) -> Face {
    let mut f = create_face_from_diagonals(
        block,
        block.imax - 1,
        0,
        0,
        block.imax - 1,
        block.jmax - 1,
        block.kmax - 1,
    );
    f.set_block_index(block_index);
    f
}

fn imin_face(block: &Block, block_index: usize) -> Face {
    let mut f = create_face_from_diagonals(block, 0, 0, 0, 0, block.jmax - 1, block.kmax - 1);
    f.set_block_index(block_index);
    f
}

fn all_outer_records(blocks: &[Block]) -> Vec<FaceRecord> {
    let mut out = Vec::new();
    for (b, block) in blocks.iter().enumerate() {
        let (faces, _) = get_outer_faces(block);
        for mut f in faces {
            f.set_block_index(b);
            out.push(f.to_record());
        }
    }
    out
}

/// Two 5x5x5 blocks stacked along x, sharing the plane x = 0.4, with one
/// *interior* node of block 0's shared face displaced by `bulge` along x.
/// The four corners of the shared face agree exactly. Dimensions 5 give a
/// GCD of 4, so the reduced grid samples only the corners.
fn corner_fooling_pair(bulge: Float) -> Vec<Block> {
    let d = [0.1, 0.1, 0.1];
    let dims = [5, 5, 5];
    let b0 = grid_block([0.0; 3], d, dims, |i, j, k, mut p| {
        if i == 4 && j == 2 && k == 2 {
            p[0] += bulge;
        }
        p
    });
    let b1 = grid_block([0.4, 0.0, 0.0], d, dims, |_, _, _, p| p);
    vec![b0, b1]
}

fn outer_contains(outer: &[FaceRecord], block: usize, lo: [usize; 3], hi: [usize; 3]) -> bool {
    outer.iter().any(|r| {
        let (rlo, rhi) = r.bounds();
        r.block_index == block && rlo == lo && rhi == hi
    })
}

// ---------------------------------------------------------------------------
// (1) Corners propose, every node certifies
// ---------------------------------------------------------------------------

#[test]
fn corners_agree_but_interior_node_does_not_is_not_a_full_face_match() {
    let tol = 1e-6;
    let blocks = corner_fooling_pair(50.0 * tol);
    let fa = imax_face(&blocks[0], 0);
    let fb = imin_face(&blocks[1], 1);

    // The corner-only proposal is fooled — this is exactly what the old
    // `full_face_match_transformed` certified.
    assert!(corner_match(&fa, &fb, tol).is_some());
    // Every-node certification is not.
    assert!(full_face_match(&fa, &blocks[0], &fb, &blocks[1], tol).is_none());

    // The failure names the displaced node.
    let pa = Patch::new(0, [4, 0, 0], [4, 4, 4]).unwrap();
    let pb = Patch::new(1, [0, 0, 0], [0, 4, 4]).unwrap();
    match certify_correspondence(&blocks[0], &pa, &blocks[1], &pb, |p| p, tol) {
        Err(MappingFailure::ExceedsTolerance { worst, .. }) => {
            assert_eq!(worst.node_a, [4, 2, 2]);
            assert!((worst.distance - 50.0 * tol).abs() < 1e-12);
        }
        other => panic!("expected ExceedsTolerance, got {other:?}"),
    }

    // Without the bulge the same faces certify with the identity mapping.
    let exact = corner_fooling_pair(0.0);
    let o = full_face_match(&imax_face(&exact[0], 0), &exact[0], &imin_face(&exact[1], 1), &exact[1], tol)
        .expect("exact faces certify");
    assert_eq!(o.permutation_index, 0);
}

#[test]
fn certification_finds_reversed_and_transposed_mappings_and_verifier_agrees() {
    let tol = 1e-9;
    let d = [0.1, 0.1, 0.1];
    let a = grid_block([0.0; 3], d, [3, 4, 5], |_, _, _, p| p);
    // Block b's jmin face lies on a's imax plane with u/v transposed
    // (b's i runs along a's z, b's k along a's y) and one axis reversed.
    let b = grid_block([0.0; 3], d, [5, 3, 4], |i, j, k, _| {
        [
            0.2 + 0.1 * j as Float,
            0.1 * (3 - k) as Float, // reversed
            0.1 * i as Float,
        ]
    });
    let pa = Patch::new(0, [2, 0, 0], [2, 3, 4]).unwrap();
    let pb = Patch::new(1, [0, 0, 0], [4, 0, 3]).unwrap();
    let m = certify_correspondence(&a, &pa, &b, &pb, |p| p, tol).expect("certifies");
    assert!(m.permutation_index & 4 != 0, "expected a transposed mapping, got {}", m.permutation_index);
    assert_eq!(m.nodes_checked, 20);

    // The certified index is the verifier's index: apply_permutation with it
    // lays b's canonical grid out in a's order, point for point.
    let fm = plot3d::correspondence::build_face_match(&pa, &pb, &m);
    let (verified, mismatched) = plot3d::verify_connectivity(&[a, b], &[fm], tol);
    assert_eq!(verified.len(), 1);
    assert!(mismatched.is_empty());
    assert_eq!(verified[0].orientation.as_ref().unwrap().permutation_index, m.permutation_index);
}

#[test]
fn collapsed_patch_is_reported_ambiguous_not_guessed() {
    // A face whose every node is the same point admits every mapping.
    let a = grid_block([0.0; 3], [0.0, 0.0, 0.0], [3, 3, 3], |_, _, _, p| p);
    let pa = Patch::new(0, [0, 0, 0], [0, 2, 2]).unwrap();
    match certify_correspondence(&a, &pa, &a, &pa, |p| p, 1e-9) {
        Err(MappingFailure::Ambiguous { permutations }) => assert!(permutations.len() > 1),
        other => panic!("expected Ambiguous, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// (3) Reduced grids discover; the full grid decides
// ---------------------------------------------------------------------------

#[test]
fn connectivity_fast_rejects_a_reduced_grid_match_that_fails_at_full_resolution() {
    let tol = 1e-6;
    let blocks = corner_fooling_pair(50.0 * tol);
    assert_eq!(plot3d::compute_min_gcd(&blocks), 4, "fixture must reduce to corners only");

    // The reduced grid (corners only) proposes the interface; the full grid
    // refutes it, and both faces come back as outer faces.
    let (matches, outer) = connectivity_fast_with_tol(&blocks, tol);
    assert!(matches.is_empty(), "bulged interface must not be returned: {matches:?}");
    assert!(outer_contains(&outer, 0, [4, 0, 0], [4, 4, 4]));
    assert!(outer_contains(&outer, 1, [0, 0, 0], [0, 4, 4]));
    assert_eq!(outer.len(), 12);
    // Ids stay unique after the demotion.
    let mut ids: Vec<usize> = outer.iter().map(|r| r.id.unwrap()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 12);

    // Full-resolution discovery agrees.
    let (matches_full, outer_full) = plot3d::connectivity_with_tol(&blocks, tol);
    assert!(matches_full.is_empty());
    assert_eq!(outer_full.len(), 12);

    // Bulge within tolerance: both entry points keep the match.
    let fine = corner_fooling_pair(0.5 * tol);
    let (m_fast, o_fast) = connectivity_fast_with_tol(&fine, tol);
    assert_eq!(m_fast.len(), 1);
    assert_eq!(o_fast.len(), 10);
    let (m_full, _) = plot3d::connectivity_with_tol(&fine, tol);
    assert_eq!(m_full.len(), 1);
}

#[test]
fn connectivity_fast_default_path_is_unchanged_on_a_clean_mesh() {
    let blocks = corner_fooling_pair(0.0);
    let (m, o) = connectivity_fast(&blocks);
    let (m2, o2) = connectivity(&blocks);
    assert_eq!(m.len(), 1);
    assert_eq!(m2.len(), 1);
    assert_eq!(o.len(), 10);
    assert_eq!(o2.len(), 10);
    assert_eq!(m[0].block1.bounds(), m2[0].block1.bounds());
    assert_eq!(m[0].block2.bounds(), m2[0].block2.bounds());
}

/// Annular sector: i along x, j along r, k along theta in [0, pitch]; the
/// kmin and kmax faces are periodic under rotation about x by `pitch`.
fn sector(dims: [usize; 3], pitch: Float, edit: impl Fn(usize, usize, usize, [Float; 3]) -> [Float; 3]) -> Block {
    let [ni, nj, nk] = dims;
    let n = ni * nj * nk;
    let (mut x, mut y, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let ax = i as Float / (ni - 1) as Float;
                let r = 1.0 + j as Float / (nj - 1) as Float;
                let th = pitch * k as Float / (nk - 1) as Float;
                let p = edit(i, j, k, [ax, r * th.cos(), r * th.sin()]);
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }
        }
    }
    Block::new(ni, nj, nk, x, y, z)
}

#[test]
fn rotated_periodicity_rejects_a_reduced_grid_pair_that_fails_at_full_resolution() {
    let pitch_deg: Float = 360.0 / 55.0;
    let pitch = pitch_deg.to_radians();
    let tol = 1e-6;
    // 9x9x9 reduces by 8 to corners only; displace an interior node of the
    // kmax (theta = pitch) face radially.
    let bulged = sector([9, 9, 9], pitch, |i, j, k, mut p| {
        if k == 8 && i == 4 && j == 4 {
            p[1] += 50.0 * tol;
        }
        p
    });
    let blocks = vec![bulged];
    let outer = all_outer_records(&blocks);
    assert_eq!(outer.len(), 6);

    let (pairs, remaining) = rotated_periodicity_with_tol(&blocks, &[], &outer, pitch_deg, 'x', true, tol);
    assert!(pairs.is_empty(), "bulged periodic pair must not survive full-resolution certification: {pairs:?}");
    assert_eq!(remaining.len(), 6);
    assert!(outer_contains(&remaining, 0, [0, 0, 0], [8, 8, 0]));
    assert!(outer_contains(&remaining, 0, [0, 0, 8], [8, 8, 8]));

    // Full-resolution discovery agrees.
    let (pairs_full, _) = rotated_periodicity_with_tol(&blocks, &[], &outer, pitch_deg, 'x', false, tol);
    assert!(pairs_full.is_empty());

    // The exact sector is periodic on both paths.
    let exact = vec![sector([9, 9, 9], pitch, |_, _, _, p| p)];
    let outer = all_outer_records(&exact);
    let (pairs, remaining) = rotated_periodicity_with_tol(&exact, &[], &outer, pitch_deg, 'x', true, tol);
    assert_eq!(pairs.len(), 1);
    assert_eq!(remaining.len(), 4);
    let (pairs_full, _) = rotated_periodicity_with_tol(&exact, &[], &outer, pitch_deg, 'x', false, tol);
    assert_eq!(pairs_full.len(), 1);
    // And the pair verifies at full resolution with the orientation returned.
    let (verified, mismatched) = plot3d::verify_periodicity(&exact, &pairs_full, pitch, 'x', tol);
    assert_eq!(verified.len(), 1);
    assert!(mismatched.is_empty());
}

#[test]
fn partial_face_match_certifies_the_whole_claimed_subpatch() {
    let tol = 1e-6;
    let d = [0.1, 0.1, 0.1];
    // Block A: imax face spans j 0..8, k 0..4. Block B covers j 0..4 exactly.
    // Block C covers j 4..8 with the same node count but a different
    // interior distribution (non-conformal): corners and edges agree, the
    // interior does not.
    let a = grid_block([0.0; 3], d, [3, 9, 5], |_, _, _, p| p);
    let b = grid_block([0.2, 0.0, 0.0], d, [3, 5, 5], |_, _, _, p| p);
    let c = grid_block([0.2, 0.4, 0.0], d, [3, 5, 5], |_, j, k, mut p| {
        if (1..4).contains(&j) && (1..4).contains(&k) {
            p[1] += 0.03; // interior nodes slide along the face
        }
        p
    });
    let blocks = vec![a, b, c];
    let (matches, outer) = plot3d::connectivity_with_tol(&blocks, tol);
    let ab: Vec<_> = matches
        .iter()
        .filter(|m| (m.block1.block_index, m.block2.block_index) == (0, 1))
        .collect();
    let ac: Vec<_> = matches
        .iter()
        .filter(|m| (m.block1.block_index, m.block2.block_index) == (0, 2))
        .collect();
    assert_eq!(ab.len(), 1, "A-B is a conformal half-face interface: {matches:?}");
    assert_eq!(ab[0].block1.bounds(), ([2, 0, 0], [2, 4, 4]));
    assert!(ac.is_empty(), "A-C is non-conformal and must not be reported as an interface: {ac:?}");
    // C's imin face and the unmatched half of A's face are exterior.
    assert!(outer_contains(&outer, 2, [0, 0, 0], [0, 4, 4]));
    assert!(outer_contains(&outer, 0, [2, 4, 0], [2, 8, 4]));
}

// ---------------------------------------------------------------------------
// (2) Coincidence is a distance, not a bin
// ---------------------------------------------------------------------------

#[test]
fn translational_periodicity_finds_pairs_whose_nodes_straddle_rounding_bins() {
    // Single block, y-periodic. Dimensions with GCD 1 so no reduction.
    // In-plane spacing 0.1 gives the adaptive pair tolerance
    // 0.03 * 0.1 = 3e-3; the ymax face is slid in x by 0.9 of that, so
    // Phase 1 (1e-6) misses it and Phase 2 must find it.
    let tol_pair: Float = 3e-3;
    let slide = 0.9 * tol_pair;
    let d = [0.1, 0.1, 0.1];
    let dims = [12, 11, 10];
    // Base x offset puts the lower-face nodes at fractional parts
    // {0.95, 0.28, 0.62} of a bin (x spacing is 33.33 bins), so the slid
    // partner 0.9 bin away always lands in a different bin under both
    // `round` (the old key) and `floor` (the grid's key).
    let x0 = 0.95 * tol_pair;
    let block = grid_block([x0, 0.0, 0.0], d, dims, |_, j, _, mut p| {
        if j == dims[1] - 1 {
            p[0] += slide;
        }
        p
    });
    // Every node pair really does straddle a bin, whichever way bins are cut.
    for i in 0..dims[0] {
        let xl = x0 + 0.1 * i as Float;
        let xu = xl + slide;
        assert_ne!((xl / tol_pair).round() as i64, (xu / tol_pair).round() as i64, "round i={i}");
        assert_ne!((xl / tol_pair).floor() as i64, (xu / tol_pair).floor() as i64, "floor i={i}");
    }
    let blocks = vec![block];
    let outer = all_outer_records(&blocks);
    let (pairs, remaining) = translational_periodicity(&blocks, &outer, None, "y", None, 0.02, 4, 1, 1);
    assert_eq!(pairs.len(), 1, "periodic pair within tolerance must be found: {pairs:?}");
    assert_eq!(remaining.len(), 4);
    let m = &pairs[0];
    let faces: Vec<_> = [&m.block1, &m.block2].iter().map(|r| r.bounds()).collect();
    assert!(faces.contains(&([0, 0, 0], [11, 0, 9])));
    assert!(faces.contains(&([0, 10, 0], [11, 10, 9])));

    // Slid past the tolerance: no pair, and the faces stay exterior rather
    // than being matched by a looser bin.
    let far = grid_block([x0, 0.0, 0.0], d, dims, |_, j, _, mut p| {
        if j == dims[1] - 1 {
            p[0] += 1.1 * tol_pair;
        }
        p
    });
    let blocks = vec![far];
    let outer = all_outer_records(&blocks);
    let (pairs, remaining) = translational_periodicity(&blocks, &outer, None, "y", None, 0.02, 4, 1, 1);
    assert!(pairs.is_empty(), "{pairs:?}");
    assert_eq!(remaining.len(), 6);
}

#[test]
fn translational_pair_with_a_bulged_interior_node_is_not_certified() {
    let d = [0.1, 0.1, 0.1];
    let dims = [12, 11, 10];
    let block = grid_block([0.0; 3], d, dims, |i, j, k, mut p| {
        if j == dims[1] - 1 && i == 5 && k == 5 {
            p[2] += 0.02; // beyond the 3e-3 adaptive tolerance, corners intact
        }
        p
    });
    let blocks = vec![block];
    let outer = all_outer_records(&blocks);
    let (pairs, remaining) = translational_periodicity(&blocks, &outer, None, "y", None, 0.02, 4, 1, 1);
    assert!(pairs.is_empty(), "{pairs:?}");
    assert_eq!(remaining.len(), 6);
}

// ---------------------------------------------------------------------------
// (4) Tolerances are the caller's to set
// ---------------------------------------------------------------------------

#[test]
fn connectivity_fast_with_tol_lets_the_caller_decide() {
    let nudge = 3e-5;
    let d = [0.1, 0.1, 0.1];
    let b0 = grid_block([0.0; 3], d, [5, 5, 5], |i, _, _, mut p| {
        if i == 4 {
            p[0] += nudge; // whole shared face offset by a known amount
        }
        p
    });
    let b1 = grid_block([0.4, 0.0, 0.0], d, [5, 5, 5], |_, _, _, p| p);
    let blocks = vec![b0, b1];
    let (m_tight, _) = connectivity_fast_with_tol(&blocks, 1e-6);
    assert!(m_tight.is_empty());
    let (m_loose, o_loose) = connectivity_fast_with_tol(&blocks, 1e-4);
    assert_eq!(m_loose.len(), 1);
    assert_eq!(o_loose.len(), 10);
}

#[test]
fn rotated_periodicity_with_tol_lets_the_caller_decide() {
    let pitch_deg: Float = 360.0 / 55.0;
    let pitch = pitch_deg.to_radians();
    let noise = 3e-5;
    let block = sector([9, 9, 9], pitch, |_, _, k, mut p| {
        if k == 8 {
            p[0] += noise;
        }
        p
    });
    let blocks = vec![block];
    let outer = all_outer_records(&blocks);
    let (tight, _) = rotated_periodicity_with_tol(&blocks, &[], &outer, pitch_deg, 'x', false, 1e-6);
    assert!(tight.is_empty());
    let (loose, _) = rotated_periodicity_with_tol(&blocks, &[], &outer, pitch_deg, 'x', false, 1e-4);
    assert_eq!(loose.len(), 1);
    // The default is the historical 1e-4.
    assert_eq!(plot3d::DEFAULT_MATCH_TOL, 1e-4);
    let (default, _) = rotated_periodicity(&blocks, &[], &outer, pitch_deg, 'x', false);
    assert_eq!(default.len(), 1);
}

#[test]
fn translational_periodicity_with_tols_lets_the_caller_decide() {
    let d = [0.1, 0.1, 0.1];
    let dims = [12, 11, 10];
    let slide = 2e-3; // below the default 3e-3 adaptive tolerance
    let block = grid_block([0.0; 3], d, dims, |_, j, _, mut p| {
        if j == dims[1] - 1 {
            p[0] += slide;
        }
        p
    });
    let blocks = vec![block];
    let outer = all_outer_records(&blocks);
    let defaults = TranslationalTolerances::default();
    assert_eq!(defaults.full_face_tol, 1e-6);
    assert_eq!(defaults.adaptive_floor, 1e-4);
    assert_eq!(defaults.adaptive_spacing_fraction, 0.03);

    let (found, _) = translational_periodicity_with_tols(&blocks, &outer, None, "y", None, 0.02, 4, 1, 1, &defaults);
    assert_eq!(found.len(), 1);

    // A caller who knows the mesh is accurate to 1e-3 refuses the slid face.
    let strict = TranslationalTolerances {
        adaptive_spacing_fraction: 0.001,
        adaptive_floor: 1e-3,
        ..Default::default()
    };
    let (none, _) = translational_periodicity_with_tols(&blocks, &outer, None, "y", None, 0.02, 4, 1, 1, &strict);
    assert!(none.is_empty(), "{none:?}");

    // And node_tol_xyz still overrides everything.
    let (found, _) = translational_periodicity_with_tols(&blocks, &outer, None, "y", Some(5e-3), 0.02, 4, 1, 1, &strict);
    assert_eq!(found.len(), 1);
}

#[test]
fn outer_face_self_match_and_vertex_match_tolerances_are_parameters() {
    // A block whose imin and imax faces coincide to within 5e-8.
    let d = [0.1, 0.1, 0.1];
    let block = grid_block([0.0; 3], d, [3, 3, 3], |i, _, _, mut p| {
        if i == 2 {
            p[0] = 5e-8;
        }
        p
    });
    let (_, self_pairs_default) = get_outer_faces(&block);
    assert!(self_pairs_default.is_empty(), "default 1e-8 does not pair them");
    let (_, self_pairs) = get_outer_faces_with_tol(&block, 1e-7);
    assert_eq!(self_pairs.len(), 1);
    assert_eq!(plot3d::block_face_functions::DEFAULT_TOL, 1e-8);

    let fa = imin_face(&block, 0);
    let fb = imax_face(&block, 0);
    assert_eq!(fa.match_indices(&fb).len(), 4, "5e-8 is within the 1e-6 default");
    assert!(fa.match_indices_with_tol(&fb, 1e-9).is_empty());
    assert_eq!(plot3d::block_face_functions::VERTEX_MATCH_TOL, 1e-6);
}
