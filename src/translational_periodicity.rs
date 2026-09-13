//! Translational periodicity detection for structured multi-block grids.
//!
//! Identifies periodic face pairs along a translational axis (x, y, or z).
//! The algorithm uses [`find_bounding_faces`] to collect faces on the min/max
//! extremes of the specified axis, then matches them using
//! `full_face_match_transformed` with a translation offset.
//!
//! [`find_bounding_faces`]: crate::block_analysis::find_bounding_faces
//!
//! Every pair this module returns has been certified node-for-node — every
//! node of the claimed patch on each side, not just the corners or a
//! fraction of nodes — under the translation, on the grid it was found on,
//! and again on the full-resolution grid when discovery ran on a GCD-reduced
//! copy. Node proximity is always evaluated as an actual distance found
//! through a proximity grid, never as equality of rounded coordinate bins.
//!
//! Tolerances are parameterised through [`TranslationalTolerances`]; the
//! defaults reproduce the historical constants.
//!
//! While there is not yet a dedicated Rust integration test, the `tests/test_rotational_periodicity.rs`
//! fixture demonstrates the expected data flow for the periodicity modules and should be referenced
//! when extending this module. Run `cargo doc --open` to view these notes alongside the generated API
//! documentation.

use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressStyle};

use crate::{
    block::Block,
    block_analysis::find_bounding_faces,
    block_face_functions::{full_face_match_transformed, outer_face_records_to_list, Face},
    correspondence::{certify_correspondence, Patch},
    face_record::{FaceKey, FaceMatch, FaceRecord, Orientation},
    geometry::{PointGrid2, PointGrid3},
    utils::compute_min_gcd,
    Float,
};

/// Default spatial tolerance for bounding-face detection and for the
/// Phase 1 full-face comparison when no `node_tol_xyz` is given.
pub const DEFAULT_TOL: Float = 1e-6;

/// Default floor for the adaptive (spacing-derived) node-matching tolerance
/// used by Phases 2 and 3 when no `node_tol_xyz` is given.
pub const ADAPTIVE_TOL_FLOOR: Float = 1e-4;

/// Default fraction of the median in-plane spacing used for the adaptive
/// node-matching tolerance.
pub const ADAPTIVE_SPACING_FRACTION: Float = 0.03;

/// The tolerances [`translational_periodicity_with_tols`] uses.
///
/// `Default` reproduces the historical constants, so
/// [`translational_periodicity`] is unchanged for callers that pass nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationalTolerances {
    /// Relative and absolute tolerance passed to `find_bounding_faces` when
    /// collecting the faces on the axis extremes.
    pub bounding_face_tol: Float,
    /// Node tolerance for the Phase 1 full-face comparison when the caller
    /// gives no `node_tol_xyz`.
    pub full_face_tol: Float,
    /// Floor of the adaptive per-pair tolerance
    /// (`max(adaptive_spacing_fraction * spacing, adaptive_floor)`).
    pub adaptive_floor: Float,
    /// Fraction of the larger median in-plane spacing of the two faces used
    /// for the adaptive per-pair tolerance.
    pub adaptive_spacing_fraction: Float,
}

impl Default for TranslationalTolerances {
    fn default() -> Self {
        Self {
            bounding_face_tol: DEFAULT_TOL,
            full_face_tol: DEFAULT_TOL,
            adaptive_floor: ADAPTIVE_TOL_FLOOR,
            adaptive_spacing_fraction: ADAPTIVE_SPACING_FRACTION,
        }
    }
}

/// Compute corrected lb2/ub2 for a periodic face pair.
///
/// Mirrors Python's `_compute_periodic_lb_ub_orientation`:
///   1. Shifts face1's lb/ub corners by `shift_amount` along `shift_axis`.
///   2. Builds all face2 grid points within the face2 index range.
///   3. Finds the nearest face2 point to each shifted face1 corner (brute-force).
///
/// Returns `(corrected_lb2, corrected_ub2)` as `[i, j, k]` indices. These
/// are directed corners: `lb2` is the image of `lb1`, `ub2` of `ub1`. The
/// orientation itself is taken from the node-for-node certification that
/// follows, not inferred here.
#[allow(clippy::too_many_arguments)]
fn compute_periodic_lb_ub(
    blk1: &Block,
    lb1: [usize; 3],
    ub1: [usize; 3],
    blk2: &Block,
    lb2_orig: [usize; 3],
    ub2_orig: [usize; 3],
    shift_axis: usize,
    shift_amount: Float,
) -> ([usize; 3], [usize; 3]) {
    let (x1l, y1l, z1l) = blk1.xyz(lb1[0], lb1[1], lb1[2]);
    let (x1u, y1u, z1u) = blk1.xyz(ub1[0], ub1[1], ub1[2]);
    let mut p1_lb = [x1l, y1l, z1l];
    let mut p1_ub = [x1u, y1u, z1u];
    p1_lb[shift_axis] += shift_amount;
    p1_ub[shift_axis] += shift_amount;

    let lo2 = [
        lb2_orig[0].min(ub2_orig[0]),
        lb2_orig[1].min(ub2_orig[1]),
        lb2_orig[2].min(ub2_orig[2]),
    ];
    let hi2 = [
        lb2_orig[0].max(ub2_orig[0]),
        lb2_orig[1].max(ub2_orig[1]),
        lb2_orig[2].max(ub2_orig[2]),
    ];

    let mut indices2: Vec<[usize; 3]> = Vec::new();
    let mut coords2: Vec<[Float; 3]> = Vec::new();
    for i in lo2[0]..=hi2[0] {
        for j in lo2[1]..=hi2[1] {
            for k in lo2[2]..=hi2[2] {
                indices2.push([i, j, k]);
                let (x, y, z) = blk2.xyz(i, j, k);
                coords2.push([x, y, z]);
            }
        }
    }

    let nearest = |query: &[Float; 3]| -> usize {
        let mut best_idx = 0;
        let mut best_dist = Float::INFINITY;
        for (i, c) in coords2.iter().enumerate() {
            let d =
                (c[0] - query[0]).powi(2) + (c[1] - query[1]).powi(2) + (c[2] - query[2]).powi(2);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    };

    (indices2[nearest(&p1_lb)], indices2[nearest(&p1_ub)])
}

/// Get the coordinate value along `axis_idx` (0=x, 1=y, 2=z) at a given IJK index.
#[inline]
fn block_axis_val(block: &Block, ijk: [usize; 3], axis_idx: usize) -> Float {
    let (x, y, z) = block.xyz(ijk[0], ijk[1], ijk[2]);
    match axis_idx {
        0 => x,
        1 => y,
        _ => z,
    }
}

/// Certify a claimed periodic pair node-for-node: every node of `rec1`'s
/// patch on `blk1`, shifted by `shift_amount` along `shift_axis`, must lie
/// within `tol` of its counterpart on `rec2`'s patch under exactly one
/// structured mapping. Returns that mapping's orientation.
#[allow(clippy::too_many_arguments)]
fn certify_periodic_pair(
    blk1: &Block,
    rec1: &FaceRecord,
    blk2: &Block,
    rec2: &FaceRecord,
    shift_axis: usize,
    shift_amount: Float,
    tol: Float,
) -> Option<Orientation> {
    let p1 = Patch::from_record(rec1).ok()?;
    let p2 = Patch::from_record(rec2).ok()?;
    p1.check_in(std::slice::from_ref(blk1)).ok()?;
    p2.check_in(std::slice::from_ref(blk2)).ok()?;
    let shift = |mut p: [Float; 3]| {
        p[shift_axis] += shift_amount;
        p
    };
    certify_correspondence(blk1, &p1, blk2, &p2, shift, tol)
        .ok()
        .map(|m| m.orientation())
}

/// Build the periodic [`FaceMatch`] for a certified pair: `rec2` gets the
/// directed corners found by nearest-corner search, and the certified
/// orientation. Returns `None` when the pair does not certify.
#[allow(clippy::too_many_arguments)]
fn build_periodic_match(
    blk1: &Block,
    rec1: &FaceRecord,
    blk2: &Block,
    rec2_orig: &FaceRecord,
    axis_idx: usize,
    shift_amt: Float,
    tol: Float,
) -> Option<FaceMatch> {
    let lb1 = [rec1.il, rec1.jl, rec1.kl];
    let ub1 = [rec1.ih, rec1.jh, rec1.kh];
    let lb2_orig = [rec2_orig.il, rec2_orig.jl, rec2_orig.kl];
    let ub2_orig = [rec2_orig.ih, rec2_orig.jh, rec2_orig.kh];
    let (corrected_lb2, corrected_ub2) =
        compute_periodic_lb_ub(blk1, lb1, ub1, blk2, lb2_orig, ub2_orig, axis_idx, shift_amt);
    let mut rec2 = rec2_orig.clone();
    rec2.il = corrected_lb2[0];
    rec2.jl = corrected_lb2[1];
    rec2.kl = corrected_lb2[2];
    rec2.ih = corrected_ub2[0];
    rec2.jh = corrected_ub2[1];
    rec2.kh = corrected_ub2[2];
    let orient = certify_periodic_pair(blk1, rec1, blk2, &rec2, axis_idx, shift_amt, tol)?;
    Some(FaceMatch {
        block1: rec1.clone(),
        block2: rec2,
        points: Vec::new(),
        orientation: Some(orient),
    })
}

/// Discover translational periodicity along a chosen axis with the default
/// [`TranslationalTolerances`].
///
/// # Testing
/// End-to-end validation is planned to follow the pattern established in
/// `tests/test_rotational_periodicity.rs`. Until then, exercising this function in a binary or
/// ad-hoc script is recommended to mirror the original Python examples.
#[allow(clippy::too_many_arguments)]
pub fn translational_periodicity(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    delta: Option<Float>,
    translational_direction: &str,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    translational_periodicity_with_tols(
        blocks,
        outer_faces,
        delta,
        translational_direction,
        node_tol_xyz,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
        &TranslationalTolerances::default(),
    )
}

/// [`translational_periodicity`] with explicit tolerances.
///
/// * `node_tol_xyz` — when `Some`, overrides every per-pair tolerance
///   (Phase 1's full-face check and Phases 2–3's adaptive tolerance).
/// * `tols` — the defaults behind everything else; see
///   [`TranslationalTolerances`].
///
/// The GCD-reduced grid is used for discovery only. Every pair found there
/// is re-certified node-for-node on the full-resolution blocks, under the
/// shift the pair itself implies (its centroid offset along the axis) and
/// at the tolerance it was accepted with. A pair that fails at full
/// resolution is left in the outer-face list with a warning on stderr.
#[allow(clippy::too_many_arguments)]
pub fn translational_periodicity_with_tols(
    blocks: &[Block],
    outer_faces: &[FaceRecord],
    delta: Option<Float>,
    translational_direction: &str,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
    tols: &TranslationalTolerances,
) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    if blocks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let axis = translational_direction.trim().to_ascii_lowercase();
    assert!(matches!(axis.as_str(), "x" | "y" | "z"));
    let axis_idx: usize = match axis.as_str() {
        "x" => 0,
        "y" => 1,
        _ => 2,
    };

    let (lower_faces_records, upper_faces_records, _, _) = find_bounding_faces(
        blocks,
        outer_faces,
        &axis,
        "both",
        tols.bounding_face_tol,
        tols.bounding_face_tol,
    );

    let gcd_to_use = compute_min_gcd(blocks);

    let blocks_reduced = crate::block_face_functions::reduce_blocks(blocks, gcd_to_use);
    // find_bounding_faces already returns records at reduced resolution,
    // so pass gcd=1 to avoid dividing indices a second time.
    let lower_faces = outer_face_records_to_list(&blocks_reduced, &lower_faces_records, 1);
    let upper_faces = outer_face_records_to_list(&blocks_reduced, &upper_faces_records, 1);

    let delta_axis = delta.unwrap_or_else(|| {
        let global_min = blocks_reduced
            .iter()
            .map(|b| {
                b.axis_slice(axis_idx)
                    .iter()
                    .cloned()
                    .fold(Float::INFINITY, Float::min)
            })
            .fold(Float::INFINITY, Float::min);
        let global_max = blocks_reduced
            .iter()
            .map(|b| {
                b.axis_slice(axis_idx)
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max)
            })
            .fold(Float::NEG_INFINITY, Float::max);
        global_max - global_min
    });

    let axis_char = axis.chars().next().unwrap();
    let blocks_up: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(delta_axis, axis_char))
        .collect();
    let blocks_dn: Vec<Block> = blocks_reduced
        .iter()
        .map(|b| b.shifted(-delta_axis, axis_char))
        .collect();

    let mut periodic_matches: Vec<FaceMatch> = Vec::new();
    // The tolerance each match was accepted with, for the full-resolution
    // re-certification at the end.
    let mut match_tols: Vec<Float> = Vec::new();
    // Track original (pre-correction) block2 FaceRecords so we can remove
    // the correct outer faces later. The corrected lb2/ub2 may differ from
    // the original outer face keys.
    let mut original_block2_recs: Vec<FaceRecord> = Vec::new();

    let lower_pool = dedup_faces(lower_faces);
    let upper_pool = dedup_faces(upper_faces);

    // ── Phase 1: Full-face matching, every node under the shift ──
    let full_tol = node_tol_xyz.unwrap_or(tols.full_face_tol);
    let shift_up = |mut p: [Float; 3]| -> [Float; 3] {
        p[axis_idx] += delta_axis;
        p
    };
    let shift_dn = |mut p: [Float; 3]| -> [Float; 3] {
        p[axis_idx] -= delta_axis;
        p
    };
    let mut consumed_lower = HashSet::<FaceKey>::new();
    let mut consumed_upper = HashSet::<FaceKey>::new();

    let pb1 = ProgressBar::new(lower_pool.len() as u64);
    pb1.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb1.set_message("Translational Phase 1 (full faces)");

    for face_l in &lower_pool {
        pb1.inc(1);
        if consumed_lower.contains(&face_l.index_key()) {
            continue;
        }
        let Some(bl) = face_l.block_index() else { continue };
        let matched = upper_pool.iter().find_map(|face_u| {
            if consumed_upper.contains(&face_u.index_key()) {
                return None;
            }
            let bu = face_u.block_index()?;
            // Lower shifted up vs upper original
            if full_face_match_transformed(
                face_l,
                &blocks_reduced[bl],
                face_u,
                &blocks_reduced[bu],
                shift_up,
                full_tol,
            )
            .is_some()
            {
                return Some(face_u.clone());
            }
            // Upper shifted down vs lower original
            if full_face_match_transformed(
                face_u,
                &blocks_reduced[bu],
                face_l,
                &blocks_reduced[bl],
                shift_dn,
                full_tol,
            )
            .is_some()
            {
                return Some(face_u.clone());
            }
            None
        });
        if let Some(face_u) = matched {
            let rec1 = FaceRecord::from_face(face_l);
            let rec2_orig = FaceRecord::from_face(&face_u);
            let blk1_r = &blocks_reduced[rec1.block_index];
            let blk2_r = &blocks_reduced[rec2_orig.block_index];
            let lb1 = [rec1.il, rec1.jl, rec1.kl];
            let lb2 = [rec2_orig.il, rec2_orig.jl, rec2_orig.kl];
            let shift_amt = if block_axis_val(blk1_r, lb1, axis_idx)
                < block_axis_val(blk2_r, lb2, axis_idx)
            {
                delta_axis
            } else {
                -delta_axis
            };
            if let Some(fm) =
                build_periodic_match(blk1_r, &rec1, blk2_r, &rec2_orig, axis_idx, shift_amt, full_tol)
            {
                consumed_lower.insert(face_l.index_key());
                consumed_upper.insert(face_u.index_key());
                original_block2_recs.push(rec2_orig);
                periodic_matches.push(fm);
                match_tols.push(full_tol);
            }
        }
    }
    pb1.finish_with_message("Translational Phase 1 done");

    // Build remainder pools for Phase 2
    let lower_remainder: Vec<Face> = lower_pool
        .iter()
        .filter(|f| !consumed_lower.contains(&f.index_key()))
        .cloned()
        .collect();
    let upper_remainder: Vec<Face> = upper_pool
        .iter()
        .filter(|f| !consumed_upper.contains(&f.index_key()))
        .cloned()
        .collect();

    // ── Phase 2: Centroid-sorted greedy node-by-node matching on remainder ──
    // Build upper candidate pool with in-plane centroids for nearest-first matching
    struct UpperCandidate {
        face: Face,
        centroid_2d: [Float; 2],
    }
    let mut upper_pool_2: Vec<UpperCandidate> = upper_remainder
        .into_iter()
        .map(|f| {
            let c = f.centroid();
            let centroid_2d = match axis_idx {
                0 => [c[1], c[2]],
                1 => [c[0], c[2]],
                _ => [c[0], c[1]],
            };
            UpperCandidate {
                face: f,
                centroid_2d,
            }
        })
        .collect();

    let pb2 = ProgressBar::new(lower_remainder.len() as u64);
    pb2.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40.cyan/blue}] {pos}/{len} faces ({eta} remaining)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb2.set_message("Translational Phase 2 (nodes)");

    for face_l in &lower_remainder {
        pb2.inc(1);
        if upper_pool_2.is_empty() {
            break;
        }

        // Compute lower face's in-plane centroid for distance sorting
        let c = face_l.centroid();
        let lower_c2d = match axis_idx {
            0 => [c[1], c[2]],
            1 => [c[0], c[2]],
            _ => [c[0], c[1]],
        };

        // Sort upper candidates by distance to lower face centroid (nearest first)
        let mut indices: Vec<usize> = (0..upper_pool_2.len()).collect();
        indices.sort_by(|&a, &b| {
            let da = (upper_pool_2[a].centroid_2d[0] - lower_c2d[0]).powi(2)
                + (upper_pool_2[a].centroid_2d[1] - lower_c2d[1]).powi(2);
            let db = (upper_pool_2[b].centroid_2d[0] - lower_c2d[0]).powi(2)
                + (upper_pool_2[b].centroid_2d[1] - lower_c2d[1]).powi(2);
            da.partial_cmp(&db).unwrap()
        });

        // Node-sharing evidence proposes a candidate; the claimed pair must
        // then certify node-for-node before it is accepted. A candidate that
        // does not certify is skipped and the next-nearest one is tried.
        let mut accepted: Option<(usize, FaceMatch, Float)> = None;
        for &idx in &indices {
            let face_u = &upper_pool_2[idx].face;
            let Some((_mode, tol_pair)) = faces_translational_match(
                face_l,
                face_u,
                &blocks_reduced,
                &blocks_up,
                &blocks_dn,
                axis.as_str(),
                delta_axis,
                node_tol_xyz,
                min_shared_frac,
                min_shared_abs,
                stride_u,
                stride_v,
                tols,
            ) else {
                continue;
            };
            let rec1 = FaceRecord::from_face(face_l);
            let rec2_orig = FaceRecord::from_face(face_u);
            let blk1_r = &blocks_reduced[rec1.block_index];
            let blk2_r = &blocks_reduced[rec2_orig.block_index];
            let lb1 = [rec1.il, rec1.jl, rec1.kl];
            let lb2 = [rec2_orig.il, rec2_orig.jl, rec2_orig.kl];
            let shift_amt = if block_axis_val(blk1_r, lb1, axis_idx)
                < block_axis_val(blk2_r, lb2, axis_idx)
            {
                delta_axis
            } else {
                -delta_axis
            };
            if let Some(fm) =
                build_periodic_match(blk1_r, &rec1, blk2_r, &rec2_orig, axis_idx, shift_amt, tol_pair)
            {
                accepted = Some((idx, fm, tol_pair));
                break;
            }
        }

        if let Some((idx, fm, tol_pair)) = accepted {
            let face_u = upper_pool_2.remove(idx);
            original_block2_recs.push(FaceRecord::from_face(&face_u.face));
            periodic_matches.push(fm);
            match_tols.push(tol_pair);
        }
    }
    pb2.finish_with_message("Translational Phase 2 done");

    // Free shifted block copies now that matching is complete
    drop(blocks_up);
    drop(blocks_dn);

    // ── Phase 3: Oblique fallback (bladed-cascade pitch boundaries) ──
    //
    // Phases 1–2 only see faces collected by `find_bounding_faces`, i.e.
    // faces lying at the GLOBAL axis extremes — flat constant-coordinate
    // pitch planes. A bladed cascade's pitch boundaries are oblique
    // (they follow the metal-angle inlet/outlet extensions and hug the
    // O-grid), so those pools come back empty and nothing is found even
    // on an exactly-periodic mesh. This phase considers ALL still-
    // unmatched outer faces, exploiting two properties of a pure
    // translation along `axis`:
    //   1. A periodic face is a single-valued height field over the
    //      orthogonal plane (its footprint). Faces whose normal is ⊥ to
    //      the axis collapse to a line; wrap-around blade walls are
    //      multi-valued — both are rejected by footprint-quality stats.
    //   2. Δ is the median per-footprint-point axis offset (no global-
    //      extent assumption — that heuristic is wrong for cascades
    //      where the domain spans more than one pitch).
    // Verification is a 3D node intersection under the estimated shift
    // (the smaller side must be ≥ 95 % covered) followed by node-for-node
    // certification of the claimed pair. Faces may match PARTIALLY (an
    // unsplit pitch face hosts several smaller counterparts), so only the
    // fully-covered side is retired.
    {
        let consumed_keys: HashSet<FaceKey> = periodic_matches
            .iter()
            .map(|m| m.block1.index_key())
            .chain(original_block2_recs.iter().map(|r| r.index_key()))
            .collect();
        let all_faces = outer_face_records_to_list(&blocks_reduced, outer_faces, gcd_to_use);
        let remaining_faces: Vec<Face> = all_faces
            .into_iter()
            .filter(|f| !consumed_keys.contains(&f.index_key()))
            .collect();

        let project = |p: &[Float; 3]| -> [Float; 2] {
            match axis_idx {
                0 => [p[1], p[2]],
                1 => [p[0], p[2]],
                _ => [p[0], p[1]],
            }
        };

        // Footprint of a face: its projections onto the orthogonal plane,
        // with the proximity grid over them. `None` when the face is not a
        // single-valued height field over that plane (ratio / multi-valued
        // thresholds calibrated on the tgs-py cascade mesh; the 95 % 3D
        // coverage and the node-for-node certification are the real
        // acceptance tests — these are cheap prunes). Proximity is an
        // actual in-plane distance within `tol`, not bin equality.
        struct Footprint {
            proj: Vec<[Float; 2]>,
            grid: PointGrid2,
        }
        let footprint = |pts: &[[Float; 3]], tol: Float| -> Option<Footprint> {
            let proj: Vec<[Float; 2]> = pts.iter().map(project).collect();
            let grid = PointGrid2::new(&proj, tol);
            let mut n_distinct = 0usize;
            let mut n_multi = 0usize;
            for (i, p) in proj.iter().enumerate() {
                match grid.nearest_within_where(*p, tol, |j| j < i) {
                    None => n_distinct += 1,
                    Some((j, _)) => {
                        if (pts[i][axis_idx] - pts[j][axis_idx]).abs() > tol {
                            n_multi += 1;
                        }
                    }
                }
            }
            let ratio = n_distinct as Float / pts.len().max(1) as Float;
            let mfrac = n_multi as Float / n_distinct.max(1) as Float;
            if ratio < 0.5 || mfrac > 0.30 {
                return None;
            }
            Some(Footprint { proj, grid })
        };

        // Cache grid points per face.
        let face_pts: Vec<Vec<[Float; 3]>> = remaining_faces
            .iter()
            .map(|f| f.grid_points(&blocks_reduced[f.block_index().unwrap()], 1, 1))
            .collect();

        // (coverage_frac, n_shared, ia, ib, d_pair, tol_pair)
        let mut pair_hits: Vec<(Float, usize, usize, usize, Float, Float)> = Vec::new();
        for ia in 0..remaining_faces.len() {
            for ib in (ia + 1)..remaining_faces.len() {
                let (fa, fb) = (&remaining_faces[ia], &remaining_faces[ib]);
                let Some(tol_pair) =
                    pair_tolerance(fa, fb, &blocks_reduced, node_tol_xyz, axis.as_str(), tols)
                else {
                    continue;
                };
                let (ma, mb) = match (
                    footprint(&face_pts[ia], tol_pair),
                    footprint(&face_pts[ib], tol_pair),
                ) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                };
                // Footprint-overlap gate + Δ estimation (median): each A
                // point pairs with the nearest B point in the plane.
                let mut diffs: Vec<Float> = ma
                    .proj
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        mb.grid
                            .nearest_within(*p, tol_pair)
                            .map(|(j, _)| face_pts[ib][j][axis_idx] - face_pts[ia][i][axis_idx])
                    })
                    .collect();
                let n_small = ma.proj.len().min(mb.proj.len());
                if diffs.len() < min_shared_abs.max(n_small / 2) {
                    continue;
                }
                diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let d_pair = diffs[diffs.len() / 2];
                if d_pair.abs() <= tol_pair {
                    continue; // coincident along axis — interface, not periodic
                }
                if let Some(d) = delta {
                    if (d_pair.abs() - d.abs()).abs() > 10.0 * tol_pair {
                        continue; // caller pinned the pitch
                    }
                }
                // 3D verification under the estimated shift: actual
                // distances through a proximity grid.
                let shifted: Vec<[Float; 3]> = face_pts[ia]
                    .iter()
                    .map(|p| {
                        let mut q = *p;
                        q[axis_idx] += d_pair;
                        q
                    })
                    .collect();
                let grid_b = PointGrid3::new(&face_pts[ib], tol_pair);
                let inter = grid_b.count_with_partner(&shifted, tol_pair);
                let n3_small = face_pts[ia].len().min(face_pts[ib].len());
                let need = min_shared_abs.max((0.95 * n3_small as Float) as usize);
                if inter < need {
                    continue;
                }
                let frac = inter as Float / n3_small.max(1) as Float;
                pair_hits.push((frac, inter, ia, ib, d_pair, tol_pair));
            }
        }

        // Best-coverage pairs first; retire only fully-covered sides.
        pair_hits.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(b.1.cmp(&a.1))
        });
        let mut fully_used: HashSet<FaceKey> = HashSet::new();
        for (frac, _n_shared, ia, ib, d_pair, tol_pair) in pair_hits {
            let (fa, fb) = (&remaining_faces[ia], &remaining_faces[ib]);
            if fully_used.contains(&fa.index_key()) || fully_used.contains(&fb.index_key()) {
                continue;
            }
            // block1 = smaller (contained) face; Δ maps block1 → block2.
            let (f1, f2, d12) = if face_pts[ia].len() <= face_pts[ib].len() {
                (fa, fb, d_pair)
            } else {
                (fb, fa, -d_pair)
            };
            let rec1 = FaceRecord::from_face(f1);
            let rec2_orig = FaceRecord::from_face(f2);
            let blk1_r = &blocks_reduced[rec1.block_index];
            let blk2_r = &blocks_reduced[rec2_orig.block_index];
            let Some(fm) =
                build_periodic_match(blk1_r, &rec1, blk2_r, &rec2_orig, axis_idx, d12, tol_pair)
            else {
                continue;
            };
            original_block2_recs.push(rec2_orig);
            periodic_matches.push(fm);
            match_tols.push(tol_pair);
            if frac >= 0.95 {
                let smaller_key = if face_pts[ia].len() <= face_pts[ib].len() {
                    fa.index_key()
                } else {
                    fb.index_key()
                };
                fully_used.insert(smaller_key);
            }
        }
    }

    // Scale periodic matches back to original resolution FIRST so that
    // periodic_keys are at the same resolution as outer_faces (which are
    // already at original resolution from connectivity_fast).
    if gcd_to_use > 1 {
        for rec in &mut periodic_matches {
            rec.block1.scale_indices(gcd_to_use);
            rec.block2.scale_indices(gcd_to_use);
        }
        for rec in &mut original_block2_recs {
            rec.scale_indices(gcd_to_use);
        }
        // Reduced-grid discovery is a proposal; certify on the full grid.
        let (kept, kept_recs) =
            revalidate_translational(blocks, periodic_matches, &original_block2_recs, &match_tols, axis_idx);
        periodic_matches = kept;
        original_block2_recs = kept_recs;
    }

    let mut periodic_keys = HashSet::new();
    for rec in &periodic_matches {
        periodic_keys.insert(rec.block1.index_key());
        periodic_keys.insert(rec.block2.index_key()); // corrected block2 keys
    }
    // Add original (pre-correction) block2 keys — these match the outer_faces entries
    for rec in &original_block2_recs {
        periodic_keys.insert(rec.index_key());
    }

    // outer_faces are already at original resolution — do NOT scale remaining.
    let mut remaining = Vec::new();
    for record in outer_faces {
        if !periodic_keys.contains(&record.index_key()) {
            remaining.push(record.clone());
        }
    }

    (periodic_matches, remaining)
}

/// Re-certify reduced-grid periodic proposals on the full-resolution blocks.
///
/// Each pair is checked under the shift it implies — the centroid offset of
/// its two faces along the axis — at the tolerance it was accepted with.
/// Failures are dropped (their faces stay in the outer-face list) with a
/// warning on stderr.
fn revalidate_translational(
    blocks: &[Block],
    proposed: Vec<FaceMatch>,
    original_block2_recs: &[FaceRecord],
    match_tols: &[Float],
    axis_idx: usize,
) -> (Vec<FaceMatch>, Vec<FaceRecord>) {
    let centroid_along = |block: &Block, rec: &FaceRecord| -> Float {
        let (lo, hi) = rec.bounds();
        let mut sum = 0.0;
        let mut n = 0usize;
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    sum += block_axis_val(block, [i, j, k], axis_idx);
                    n += 1;
                }
            }
        }
        sum / n.max(1) as Float
    };
    let mut kept = Vec::with_capacity(proposed.len());
    let mut kept_recs = Vec::with_capacity(proposed.len());
    for ((fm, orig2), &tol) in proposed.into_iter().zip(original_block2_recs).zip(match_tols) {
        let ok = (|| {
            let b1 = blocks.get(fm.block1.block_index)?;
            let b2 = blocks.get(fm.block2.block_index)?;
            let shift = centroid_along(b2, &fm.block2) - centroid_along(b1, &fm.block1);
            certify_periodic_pair(b1, &fm.block1, b2, &fm.block2, axis_idx, shift, tol)
        })();
        if ok.is_some() {
            kept.push(fm);
            kept_recs.push(orig2.clone());
        } else {
            eprintln!(
                "translational_periodicity: reduced-grid pair block {} [{},{},{} -> {},{},{}] <-> block {} [{},{},{} -> {},{},{}] does not certify node-for-node at full resolution (tol {tol:e}); leaving both faces unmatched",
                fm.block1.block_index, fm.block1.il, fm.block1.jl, fm.block1.kl,
                fm.block1.ih, fm.block1.jh, fm.block1.kh,
                fm.block2.block_index, fm.block2.il, fm.block2.jl, fm.block2.kl,
                fm.block2.ih, fm.block2.jh, fm.block2.kh,
            );
        }
    }
    (kept, kept_recs)
}

/// Assess one lower/upper face combo. Returns the match mode and the
/// per-pair tolerance it was assessed with when the faces share enough
/// nodes; `None` when they do not, or when no tolerance can be derived for
/// the pair.
#[allow(clippy::too_many_arguments)]
fn faces_translational_match(
    face_l: &Face,
    face_u: &Face,
    blocks: &[Block],
    blocks_up: &[Block],
    blocks_dn: &[Block],
    axis: &str,
    delta_axis: Float,
    node_tol_xyz: Option<Float>,
    min_shared_frac: Float,
    min_shared_abs: usize,
    stride_u: usize,
    stride_v: usize,
    tols: &TranslationalTolerances,
) -> Option<(String, Float)> {
    let tol_pair = pair_tolerance(face_l, face_u, blocks, node_tol_xyz, axis, tols)?;

    if orthogonal_precheck(
        face_l,
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some((format!("{axis}_precheck_lower_up"), tol_pair));
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks_up[face_l.block_index().unwrap()],
        &blocks[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some(("lower_up_vs_upper_orig".to_string(), tol_pair));
    }
    if face_l.touches_by_nodes(
        face_u,
        &blocks[face_l.block_index().unwrap()],
        &blocks_dn[face_u.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some(("lower_orig_vs_upper_dn".to_string(), tol_pair));
    }
    if orthogonal_precheck(
        face_u,
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        delta_axis,
        tol_pair,
        axis,
        min_shared_frac,
        min_shared_abs,
    ) {
        return Some((format!("{axis}_precheck_upper_up"), tol_pair));
    }
    if face_u.touches_by_nodes(
        face_l,
        &blocks_up[face_u.block_index().unwrap()],
        &blocks[face_l.block_index().unwrap()],
        tol_pair,
        min_shared_frac,
        min_shared_abs,
        stride_u,
        stride_v,
    ) {
        return Some(("upper_up_vs_lower_orig".to_string(), tol_pair));
    }
    face_u
        .touches_by_nodes(
            face_l,
            &blocks[face_u.block_index().unwrap()],
            &blocks_dn[face_l.block_index().unwrap()],
            tol_pair,
            min_shared_frac,
            min_shared_abs,
            stride_u,
            stride_v,
        )
        .then(|| ("upper_orig_vs_lower_dn".to_string(), tol_pair))
}

/// Decide the XYZ tolerance for a particular face pair, optionally honoring
/// a global override.
///
/// Without an override the tolerance is derived from the faces' median
/// in-plane spacing. `None` when neither face has enough nodes to measure a
/// spacing: that is insufficient information, not a spacing of 1.
fn pair_tolerance(
    face_a: &Face,
    face_b: &Face,
    blocks: &[Block],
    override_tol: Option<Float>,
    axis: &str,
    tols: &TranslationalTolerances,
) -> Option<Float> {
    if let Some(tol) = override_tol {
        return Some(tol);
    }
    let spacing_a = median_inplane_spacing(face_a, &blocks[face_a.block_index()?], axis);
    let spacing_b = median_inplane_spacing(face_b, &blocks[face_b.block_index()?], axis);
    let spacing = match (spacing_a, spacing_b) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return None,
    };
    Some((tols.adaptive_spacing_fraction * spacing).max(tols.adaptive_floor))
}

/// Median edge length of the face in the non-periodic directions, or `None`
/// when the face has fewer than two grid points and no spacing exists.
fn median_inplane_spacing(face: &Face, block: &Block, axis: &str) -> Option<Float> {
    let points = face.grid_points(block, 1, 1);
    if points.len() <= 1 {
        return None;
    }
    let mut spacings = Vec::new();
    for window in points.windows(2) {
        let p0 = window[0];
        let p1 = window[1];
        let diff = match axis {
            "x" => [(p0[1] - p1[1]).abs(), (p0[2] - p1[2]).abs()],
            "y" => [(p0[0] - p1[0]).abs(), (p0[2] - p1[2]).abs()],
            _ => [(p0[0] - p1[0]).abs(), (p0[1] - p1[1]).abs()],
        };
        spacings.push(diff[0].hypot(diff[1]));
    }
    spacings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(spacings[spacings.len() / 2])
}

/// Quick planar-projection test to reject clearly mismatched faces.
///
/// A node of `face_a` (shifted) counts as shared when a node of `face_b`
/// lies within `tol` of it in the plane orthogonal to `axis` — an actual
/// in-plane distance found through a proximity grid, so two nodes that
/// straddle a bin boundary are still recognised.
#[allow(clippy::too_many_arguments)]
fn orthogonal_precheck(
    face_a: &Face,
    face_b: &Face,
    block_a: &Block,
    block_b: &Block,
    delta: Float,
    tol: Float,
    axis: &str,
    min_shared_frac: Float,
    min_shared_abs: usize,
) -> bool {
    let mut pts_a = face_a.grid_points(block_a, 1, 1);
    let pts_b = face_b.grid_points(block_b, 1, 1);
    if pts_a.is_empty() || pts_b.is_empty() {
        return false;
    }
    match axis {
        "x" => pts_a.iter_mut().for_each(|p| p[0] += delta),
        "y" => pts_a.iter_mut().for_each(|p| p[1] += delta),
        _ => pts_a.iter_mut().for_each(|p| p[2] += delta),
    }

    let proj_a = project_plane(&pts_a, axis);
    let proj_b = project_plane(&pts_b, axis);

    let grid_b = PointGrid2::new(&proj_b, tol);
    let shared = grid_b.count_with_partner(&proj_a, tol);
    shared >= min_shared_abs
        && (shared as Float) >= min_shared_frac * (proj_a.len().min(proj_b.len()) as Float)
}

/// Project 3D points onto the plane orthogonal to `axis`.
fn project_plane(points: &[[Float; 3]], axis: &str) -> Vec<[Float; 2]> {
    points
        .iter()
        .map(|p| match axis {
            "x" => [p[1], p[2]],
            "y" => [p[0], p[2]],
            _ => [p[0], p[1]],
        })
        .collect()
}

/// Remove duplicate faces while preserving the first occurrence.
fn dedup_faces(mut faces: Vec<Face>) -> Vec<Face> {
    let mut seen = HashSet::new();
    faces.retain(|f| seen.insert(f.index_key()));
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_of_a_single_point_face_is_insufficient_information() {
        // A "face" collapsed to one node: no spacing exists, so no tolerance
        // can be derived from it.
        let block = Block::new(1, 1, 1, vec![0.0], vec![0.0], vec![0.0]);
        let mut face = Face::new();
        face.add_vertex(0.0, 0.0, 0.0, 0, 0, 0);
        face.set_block_index(0);
        assert_eq!(median_inplane_spacing(&face, &block, "x"), None);
        let tols = TranslationalTolerances::default();
        assert_eq!(pair_tolerance(&face, &face, &[block], None, "x", &tols), None);
    }

    #[test]
    fn override_tolerance_needs_no_spacing() {
        let block = Block::new(1, 1, 1, vec![0.0], vec![0.0], vec![0.0]);
        let mut face = Face::new();
        face.add_vertex(0.0, 0.0, 0.0, 0, 0, 0);
        face.set_block_index(0);
        let tols = TranslationalTolerances::default();
        assert_eq!(
            pair_tolerance(&face, &face, &[block], Some(3e-5), "x", &tols),
            Some(3e-5)
        );
    }

    #[test]
    fn adaptive_tolerance_uses_the_configured_fraction_and_floor() {
        // 3x3 K-face with 0.1 spacing.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                x.push(0.1 * i as Float);
                y.push(0.1 * j as Float);
                z.push(0.0);
            }
        }
        let block = Block::new(3, 3, 1, x, y, z);
        let face = crate::block_face_functions::create_face_from_diagonals(&block, 0, 0, 0, 2, 2, 0);
        let mut face = face;
        face.set_block_index(0);
        let blocks = [block];
        let custom = TranslationalTolerances {
            adaptive_floor: 1e-9,
            adaptive_spacing_fraction: 0.5,
            ..Default::default()
        };
        // Spacing along the sampled point sequence: median of the
        // consecutive-point in-plane distances (0.1 or the row-wrap jump).
        let s = median_inplane_spacing(&face, &blocks[0], "z").unwrap();
        let t = pair_tolerance(&face, &face, &blocks, None, "z", &custom).unwrap();
        assert!((t - 0.5 * s).abs() < 1e-12);
        let floored = TranslationalTolerances {
            adaptive_floor: 10.0,
            ..custom
        };
        assert_eq!(pair_tolerance(&face, &face, &blocks, None, "z", &floored), Some(10.0));
    }
}
