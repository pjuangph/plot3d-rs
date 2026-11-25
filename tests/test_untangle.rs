//! Folded-grid-line untangling: detection, repair, refusal.
//!
//! Fixtures are cartesian blocks with a thin cell layer (a wall-resolved
//! spacing) under the `jmax` face, where moving one node beyond its `j+1`
//! neighbour reverses interior faces without inverting a cell — the defect
//! the repair targets.

use plot3d::{
    find_inverted_geometry, untangle_folded_lines, untangle_mesh, Block, Float, RepairConstruction,
    ReversedFace,
};

/// Cartesian block: `x = 0.25 i`, `y = ys[j]`, `z = 0.25 k`.
fn block(ni: usize, ys: &[Float], nk: usize) -> Block {
    let nj = ys.len();
    let n = ni * nj * nk;
    let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let id = (k * nj + j) * ni + i;
                x[id] = 0.25 * i as Float;
                y[id] = ys[j];
                z[id] = 0.25 * k as Float;
            }
        }
    }
    Block::new(ni, nj, nk, x, y, z)
}

fn with_y(b: &Block, node: [usize; 3], y: Float) -> Block {
    let mut out = b.clone();
    let id = b.idx(node[0], node[1], node[2]);
    out.y[id] = y;
    out
}

fn p(b: &Block, n: [usize; 3]) -> [Float; 3] {
    let (x, y, z) = b.xyz(n[0], n[1], n[2]);
    [x, y, z]
}

/// Wall layer: last j-spacing 0.01 against 0.25 in i and k.
const YS: [Float; 5] = [0.0, 0.4, 0.7, 0.99, 1.0];

#[test]
fn a_folded_node_next_to_a_face_is_rebuilt_from_its_neighbouring_lines() {
    let clean = block(5, &YS, 5);
    // Node (2,3,2) placed 0.02 beyond its jmax neighbour (2,4,2).
    let folded = with_y(&clean, [2, 3, 2], 1.02);

    let (faces, cells) = find_inverted_geometry(&folded, 7);
    assert!(
        !faces.is_empty(),
        "fixture must reverse interior faces, else the repair path is never entered"
    );
    assert!(
        cells.is_empty(),
        "the fold reverses faces without inverting a cell"
    );
    assert!(faces
        .iter()
        .all(|f: &ReversedFace| f.block == 7 && f.cos <= 0.0));

    let (repaired, report) = untangle_folded_lines(&folded, 7).expect("repairable");

    // Exactly one edge folded and one node moved: the interior endpoint.
    assert_eq!(report.folded_edges.len(), 1);
    assert_eq!(report.folded_edges[0].axis, 1);
    assert_eq!(report.folded_edges[0].start, [2, 3, 2]);
    assert_eq!(report.moved.len(), 1);
    let m = &report.moved[0];
    assert_eq!((m.block, m.node), (7, [2, 3, 2]));

    // Neighbour-mean construction, computed here independently:
    // new = X(anchor) − mean of the j-edge at (1,3,2), (3,3,2), (2,3,1), (2,3,3).
    let edge = |n: [usize; 3]| {
        let a = p(&folded, n);
        let b = p(&folded, [n[0], n[1] + 1, n[2]]);
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    };
    let mut mean = [0.0 as Float; 3];
    for n in [[1, 3, 2], [3, 3, 2], [2, 3, 1], [2, 3, 3]] {
        let e = edge(n);
        for q in 0..3 {
            mean[q] += 0.25 * e[q];
        }
    }
    let anchor = p(&folded, [2, 4, 2]);
    let expected = [
        anchor[0] - mean[0],
        anchor[1] - mean[1],
        anchor[2] - mean[2],
    ];
    assert_eq!(m.new, expected);
    assert_eq!(p(&repaired, [2, 3, 2]), expected);
    // On this cartesian fixture the construction restores the unfolded node.
    for q in 0..3 {
        assert!((expected[q] - p(&clean, [2, 3, 2])[q]).abs() < 1e-12);
    }
    assert_eq!(m.old, [0.5, 1.02, 0.5]);
    assert!((m.displacement - 0.03).abs() < 1e-12);
    assert_eq!(report.max_displacement, m.displacement);

    // The printed report names each reversed face, the folded edge and node
    // it was localised to, the move, and the clean re-check.
    assert_eq!(report.blocks_scanned, 1);
    assert_eq!(report.rechecked_blocks, vec![7]);
    let text = report.format_report();
    assert!(text.starts_with("untangle folded grid lines: scanned 1 block(s);"));
    assert!(text.contains("\nblock 7: "));
    let f = &report.reversed_faces[0];
    let nb = {
        let mut n = f.owner;
        n[f.axis] += 1;
        n
    };
    assert!(text.contains(&format!(
        "cell ({},{},{}) -> ({},{},{}), cos(a,d) {:.4e}; folded j-edge (2,3,2)-(2,4,2), node (2,3,2) moved",
        f.owner[0], f.owner[1], f.owner[2], nb[0], nb[1], nb[2], f.cos
    )));
    assert!(text.contains("  moved node (2,3,2): ("));
    assert!(text.contains("  re-check: 0 reversed face(s), 0 non-positive cell(s)"));
    assert_eq!(
        text.matches("  reversed ").count(),
        report.reversed_faces.len(),
        "every reversed face is printed"
    );

    // Re-check is clean and every other node is bit-identical.
    let (faces_after, cells_after) = find_inverted_geometry(&repaired, 7);
    assert!(faces_after.is_empty() && cells_after.is_empty());
    let moved_id = folded.idx(2, 3, 2);
    for id in 0..folded.npoints() {
        if id != moved_id {
            assert_eq!(
                (repaired.x[id], repaired.y[id], repaired.z[id]),
                (folded.x[id], folded.y[id], folded.z[id])
            );
        }
    }
}

#[test]
fn a_clean_block_is_returned_bit_identical_with_an_empty_report() {
    // Skewed so the clean block is not trivially cartesian.
    let mut b = block(5, &YS, 4);
    for id in 0..b.npoints() {
        b.x[id] += 0.1 * b.y[id] + 0.03 * b.z[id] * b.z[id];
    }
    let (out, report) = untangle_folded_lines(&b, 0).expect("clean block");
    assert!(report.is_empty());
    assert_eq!(report.max_displacement, 0.0);
    assert_eq!((out.imax, out.jmax, out.kmax), (b.imax, b.jmax, b.kmax));
    assert!(out
        .x
        .iter()
        .zip(&b.x)
        .all(|(s, t)| s.to_bits() == t.to_bits()));
    assert!(out
        .y
        .iter()
        .zip(&b.y)
        .all(|(s, t)| s.to_bits() == t.to_bits()));
    assert!(out
        .z
        .iter()
        .zip(&b.z)
        .all(|(s, t)| s.to_bits() == t.to_bits()));
}

#[test]
fn untangle_mesh_repairs_only_the_folded_block_and_labels_it() {
    let clean = block(5, &YS, 5);
    let folded = with_y(&clean, [2, 3, 2], 1.02);
    let (out, report) = untangle_mesh(&[clean.clone(), folded]).expect("repairable");
    assert_eq!(out[0].y, clean.y);
    assert_eq!(report.moved.len(), 1);
    assert_eq!(report.moved[0].block, 1);
    assert!(report.reversed_faces.iter().all(|f| f.block == 1));
    assert!(report.format_report().contains("1 node(s) moved"));
}

#[test]
fn a_left_handed_block_is_judged_relative_to_its_handedness() {
    // Reverse i: left-handed, so every face would read reversed without the
    // orientation sign. The fold at i = 2 maps onto itself (ni = 5).
    let clean = block(5, &YS, 5);
    let folded = with_y(&clean, [2, 3, 2], 1.02);
    let mut lh = folded.clone();
    for k in 0..5 {
        for j in 0..5 {
            for i in 0..5 {
                let (dst, src) = (lh.idx(i, j, k), folded.idx(4 - i, j, k));
                lh.x[dst] = folded.x[src];
                lh.y[dst] = folded.y[src];
                lh.z[dst] = folded.z[src];
            }
        }
    }
    assert_eq!(
        plot3d::block_handedness(&lh),
        plot3d::Handedness::LeftHanded
    );
    let (faces, _) = find_inverted_geometry(&lh, 0);
    let (faces_rh, _) = find_inverted_geometry(&folded, 0);
    assert_eq!(faces.len(), faces_rh.len());
    let (_, report) = untangle_folded_lines(&lh, 0).expect("repairable");
    assert_eq!(report.moved.len(), 1);
    assert_eq!(report.moved[0].node, [2, 3, 2]);
}

#[test]
fn a_fold_on_a_block_face_is_refused() {
    let clean = block(5, &YS, 5);
    // Node (0,3,2) lies on the i = 0 face; its folded j-edge ends on the
    // jmax face, so both endpoints are face nodes.
    let folded = with_y(&clean, [0, 3, 2], 1.02);
    let (faces, _) = find_inverted_geometry(&folded, 3);
    assert!(!faces.is_empty(), "fixture must reverse an interior face");
    let err = untangle_folded_lines(&folded, 3).expect_err("must refuse");
    assert!(!err.refusals.is_empty());
    assert!(err
        .refusals
        .iter()
        .any(|r| r.block == 3 && r.node == Some([0, 3, 2]) && r.reason.contains("both endpoints")));
    let text = err.to_string();
    assert!(text.contains("REFUSED"));
    assert!(text.contains("block 3 (0,3,2): folded j-edge has both endpoints"));
    // The refusal carries the same per-location detail as a repair report.
    assert!(text.contains("re-check: not run (refused before repair)"));
    assert_eq!(err.report.reversed_faces.len(), faces.len());
    // All-or-nothing at mesh level too.
    assert!(untangle_mesh(&[clean, folded]).is_err());
}

#[test]
fn an_interior_fold_with_no_unambiguous_anchor_is_refused() {
    // Both endpoints of the folded j-edge (3,3,3)->(3,4,3) are interior and
    // neither has another folded edge, so no anchor can be chosen.
    let ys = [0.0, 0.3, 0.6, 0.89, 0.9, 0.95, 1.0];
    let clean = block(7, &ys, 7);
    let folded = with_y(&clean, [3, 3, 3], 0.92);
    let (faces, _) = find_inverted_geometry(&folded, 0);
    assert!(!faces.is_empty(), "fixture must reverse an interior face");
    let err = untangle_folded_lines(&folded, 0).expect_err("must refuse");
    assert!(err
        .refusals
        .iter()
        .any(|r| r.node == Some([3, 3, 3]) && r.reason.contains("unambiguous anchor")));
}

#[test]
fn a_dented_face_under_a_thin_layer_is_repaired_on_the_chord() {
    // jmax face node (2,4,2) dented from y = 1.0 to 0.91: below its j = 3
    // neighbour (0.98) and close to the j = 2 node (0.9). Rebuilding the
    // folded edge from the neighbours' 0.02 would put (2,3,2) at 0.89, behind
    // (2,2,2), so the node goes on the chord (2,2,2)->(2,4,2) instead.
    let ys = [0.0, 0.5, 0.9, 0.98, 1.0];
    let folded = with_y(&block(5, &ys, 5), [2, 4, 2], 0.91);
    let (faces, _) = find_inverted_geometry(&folded, 0);
    assert!(!faces.is_empty(), "fixture must reverse an interior face");

    let (repaired, report) = untangle_folded_lines(&folded, 0).expect("repairable");
    assert_eq!(report.moved.len(), 1);
    let m = &report.moved[0];
    assert_eq!(m.node, [2, 3, 2]);
    assert_eq!(m.construction, RepairConstruction::ChordRedistribution);

    // Independent construction: r = |w| / (|w| + |u|), with u the mean
    // (·,3,·)->(·,4,·) edge and w the mean (·,2,·)->(·,3,·) edge on the four
    // neighbouring lines; X = X(2,2,2) + r (X(2,4,2) − X(2,2,2)).
    let (u, w) = (ys[4] - ys[3], ys[3] - ys[2]);
    let r = w / (w + u);
    let (f, a) = (p(&folded, [2, 2, 2]), p(&folded, [2, 4, 2]));
    let expected = [
        f[0] + r * (a[0] - f[0]),
        f[1] + r * (a[1] - f[1]),
        f[2] + r * (a[2] - f[2]),
    ];
    for q in 0..3 {
        assert!(
            (m.new[q] - expected[q]).abs() < 1e-14,
            "{:?} vs {:?}",
            m.new,
            expected
        );
    }
    assert!(m.new[1] > 0.9 && m.new[1] < 0.91);
    let (faces_after, cells_after) = find_inverted_geometry(&repaired, 0);
    assert!(faces_after.is_empty() && cells_after.is_empty());
    // The dented face node itself is untouched.
    assert_eq!(p(&repaired, [2, 4, 2]), a);
}

#[test]
fn a_long_report_prints_the_first_entries_and_counts_the_rest() {
    let n = plot3d::UNTANGLE_REPORT_LIMIT + 5;
    let report = plot3d::UntangleReport {
        blocks_scanned: 1,
        reversed_faces: (0..n)
            .map(|i| ReversedFace {
                block: 0,
                axis: 0,
                owner: [i, 0, 0],
                cos: -1.0,
            })
            .collect(),
        ..Default::default()
    };
    let text = report.format_report();
    assert_eq!(
        text.matches("  reversed i-face").count(),
        plot3d::UNTANGLE_REPORT_LIMIT
    );
    assert!(text.contains("... 5 more reversed face(s) not shown"));
    assert!(text.contains(&format!("{n} reversed interior face(s)")));
}
