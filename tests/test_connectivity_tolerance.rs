//! Regression tests for the adaptive connectivity tolerance.
//!
//! Background: `connectivity()` used a fixed `1e-6` node-matching tolerance.
//! Coordinate *storage* loses precision in proportion to coordinate
//! magnitude (binary `f32`, or ASCII written with a fixed number of
//! significant digits), so on a mesh whose coordinates are large the two
//! stored copies of a shared interface node differ by much more than
//! `1e-6` and the interface is silently reported as two outer faces
//! instead of one match. `adaptive_tolerance()` derives the tolerance from
//! the mesh instead; see its docs for the formula and its two bounds.
//!
//! The tests below pin both directions:
//!   * the gap is real — `connectivity_with_tol(&blocks, TOL_FLOOR)`
//!     (literally the pre-change behaviour) misses the interface;
//!   * the fix works — `connectivity(&blocks)` finds it;
//!   * nothing else moves — ordinary order-one meshes get a bit-identical
//!     tolerance and bit-identical results, and blocks that are genuinely
//!     apart still do not match.
//!
//! Noise is injected the way real files lose precision: every coordinate is
//! pushed through a storage round-trip, and block 1's copy of the shared
//! face is offset by one storage quantum first, modelling two
//! independently produced copies of the same physical node landing on
//! adjacent quanta. (Plain translation of an exact fixture reproduces
//! nothing: both copies shift identically and stay bit-identical.)

// Some helpers are only reachable in the default (f64) build; the f32
// feature compiles out the large-magnitude scenarios that use them.
#![allow(dead_code)]

use plot3d::block_face_functions::reduce_blocks;
use plot3d::{
    adaptive_tolerance, connectivity, connectivity_fast, connectivity_with_tol, Block, FaceMatch,
    Float, TOL_FLOOR,
};

// ---------------------------------------------------------------------------
// Storage models
// ---------------------------------------------------------------------------

/// Round-trip through binary single precision.
fn f32_store(v: Float) -> Float {
    v as f32 as Float
}

/// Distance to the next representable `f32` above `|v|`.
fn f32_quantum(v: Float) -> Float {
    let a = v.abs() as f32;
    if a == 0.0 || !a.is_finite() {
        return 0.0;
    }
    let next = f32::from_bits(a.to_bits() + 1);
    (next - a) as Float
}

/// Round-trip through ASCII scientific notation with 7 significant digits.
fn ascii7_store(v: Float) -> Float {
    format!("{:.6e}", v).parse::<Float>().unwrap()
}

/// Decimal quantum of a 7-significant-digit representation of `v`.
fn ascii7_quantum(v: Float) -> Float {
    if v == 0.0 || !v.is_finite() {
        return 0.0;
    }
    (10.0 as Float).powi(v.abs().log10().floor() as i32 - 6)
}

/// Exact storage: no precision loss at all.
fn exact_store(v: Float) -> Float {
    v
}
fn no_quantum(_v: Float) -> Float {
    0.0
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct Storage {
    store: fn(Float) -> Float,
    quantum: fn(Float) -> Float,
}

const EXACT: Storage = Storage {
    store: exact_store,
    quantum: no_quantum,
};
const F32: Storage = Storage {
    store: f32_store,
    quantum: f32_quantum,
};
const ASCII7: Storage = Storage {
    store: ascii7_store,
    quantum: ascii7_quantum,
};

#[allow(clippy::too_many_arguments)]
fn make_block(
    x_start: Float,
    y0: Float,
    z0: Float,
    d: [Float; 3],
    dims: [usize; 3],
    shared_at_imin: bool,
    nudge_shared: bool,
    s: &Storage,
) -> Block {
    let [ni, nj, nk] = dims;
    let n = ni * nj * nk;
    let mut x = vec![0.0 as Float; n];
    let mut y = vec![0.0 as Float; n];
    let mut z = vec![0.0 as Float; n];
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let idx = (k * nj + j) * ni + i;
                let mut cx = x_start + d[0] * i as Float;
                let mut cy = y0 + d[1] * j as Float;
                let mut cz = z0 + d[2] * k as Float;
                let on_shared = if shared_at_imin { i == 0 } else { i == ni - 1 };
                if on_shared && nudge_shared {
                    cx += (s.quantum)(cx);
                    cy += (s.quantum)(cy);
                    cz += (s.quantum)(cz);
                }
                x[idx] = (s.store)(cx);
                y[idx] = (s.store)(cy);
                z[idx] = (s.store)(cz);
            }
        }
    }
    Block::new(ni, nj, nk, x, y, z)
}

/// Two blocks stacked along +x sharing the plane `x = x0 + d[0]*(ni-1)`.
///
/// `gap` displaces block 1 along +x; `gap == 0.0` means the blocks touch.
#[allow(clippy::too_many_arguments)]
fn stacked_pair(
    x0: Float,
    y0: Float,
    z0: Float,
    d: [Float; 3],
    dims: [usize; 3],
    s: &Storage,
    nudge_shared: bool,
    gap: Float,
) -> Vec<Block> {
    let x_shared = x0 + d[0] * (dims[0] - 1) as Float;
    vec![
        make_block(x0, y0, z0, d, dims, false, false, s),
        make_block(x_shared + gap, y0, z0, d, dims, true, nudge_shared, s),
    ]
}

/// A two-block pair whose coordinates all stay within magnitude 1 — the
/// regime in which the adaptive tolerance must reduce exactly to the
/// historical fixed constant.
fn unit_pair(s: &Storage, nudge_shared: bool) -> Vec<Block> {
    stacked_pair(
        0.0,
        0.0,
        0.0,
        [0.1, 0.1, 0.1],
        [5, 4, 3],
        s,
        nudge_shared,
        0.0,
    )
}

/// Largest node-to-node distance across the shared interface of a pair built
/// by [`stacked_pair`] — i.e. how far apart the two stored copies of the
/// same physical node actually are.
fn interface_discrepancy(blocks: &[Block]) -> Float {
    let (b0, b1) = (&blocks[0], &blocks[1]);
    let mut worst: Float = 0.0;
    for k in 0..b0.kmax {
        for j in 0..b0.jmax {
            let a = b0.xyz(b0.imax - 1, j, k);
            let c = b1.xyz(0, j, k);
            let d = ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2) + (a.2 - c.2).powi(2)).sqrt();
            worst = worst.max(d);
        }
    }
    worst
}

/// Minimum non-zero *edge* length (`i`/`i+1`, `j`/`j+1`, `k`/`k+1`),
/// implemented independently of the crate. On orthogonal fixtures this is
/// the crate's `h` (every diagonal is longer than every edge); on sheared
/// fixtures it is deliberately the *old*, edge-only definition, which
/// `sheared_cells_bound_the_tolerance_by_the_short_diagonal` uses to
/// reproduce the pre-fix behaviour.
fn min_spacing(blocks: &[Block]) -> Float {
    let mut best = Float::INFINITY;
    for b in blocks {
        for k in 0..b.kmax {
            for j in 0..b.jmax {
                for i in 0..b.imax {
                    let p = b.xyz(i, j, k);
                    let mut upd = |q: (Float, Float, Float)| {
                        let d = ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2) + (p.2 - q.2).powi(2))
                            .sqrt();
                        if d > 0.0 && d < best {
                            best = d;
                        }
                    };
                    if i + 1 < b.imax {
                        upd(b.xyz(i + 1, j, k));
                    }
                    if j + 1 < b.jmax {
                        upd(b.xyz(i, j + 1, k));
                    }
                    if k + 1 < b.kmax {
                        upd(b.xyz(i, j, k + 1));
                    }
                }
            }
        }
    }
    best
}

fn max_magnitude(blocks: &[Block]) -> Float {
    blocks
        .iter()
        .flat_map(|b| b.x.iter().chain(b.y.iter()).chain(b.z.iter()))
        .fold(0.0 as Float, |a, v| a.max(v.abs()))
}

/// Two blocks stacked along +x whose cells are sheared in the j/k plane to
/// an included angle `theta` (radians): the j-edge is `h * (0, 1, 0)` and
/// the k-edge is `h * (0, cos θ, sin θ)`, so the short face diagonal is
/// `h * sqrt(2 - 2 cos θ)`. All three coordinates sit near `origin`, so a
/// storage nudge displaces the shared face in-plane as well as normal to
/// it — the situation in which a node can be nearer a wrong corner than
/// its true partner.
fn sheared_pair(
    origin: Float,
    h: Float,
    theta: Float,
    dims: [usize; 3],
    s: &Storage,
    nudge_shared: bool,
) -> Vec<Block> {
    let [ni, nj, nk] = dims;
    let build = |x0: Float, shared_at_imin: bool, nudge: bool| {
        let n = ni * nj * nk;
        let mut x = vec![0.0 as Float; n];
        let mut y = vec![0.0 as Float; n];
        let mut z = vec![0.0 as Float; n];
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    let idx = (k * nj + j) * ni + i;
                    let mut cx = x0 + h * i as Float;
                    let mut cy = origin + h * j as Float + h * theta.cos() * k as Float;
                    let mut cz = origin + h * theta.sin() * k as Float;
                    let on_shared = if shared_at_imin { i == 0 } else { i == ni - 1 };
                    if on_shared && nudge {
                        cx += (s.quantum)(cx);
                        cy += (s.quantum)(cy);
                        cz += (s.quantum)(cz);
                    }
                    x[idx] = (s.store)(cx);
                    y[idx] = (s.store)(cy);
                    z[idx] = (s.store)(cz);
                }
            }
        }
        Block::new(ni, nj, nk, x, y, z)
    };
    let x_shared = origin + h * (ni - 1) as Float;
    vec![
        build(origin, false, false),
        build(x_shared, true, nudge_shared),
    ]
}

/// Number of match points that pair a node with anything other than the
/// node of the same in-plane `(j, k)` index on the other face. Every
/// fixture in this file lines the two faces up index-for-index, so any
/// non-identity pairing is a node matched to a neighbour of its true
/// partner.
fn mispaired(matches: &[FaceMatch]) -> usize {
    matches
        .iter()
        .flat_map(|m| m.points.iter())
        .filter(|p| p.j1 != p.j2 || p.k1 != p.k2)
        .count()
}

// ---------------------------------------------------------------------------
// The tolerance value itself
// ---------------------------------------------------------------------------

#[test]
fn unit_scale_mesh_gets_the_legacy_tolerance_bit_for_bit() {
    // Coordinates of magnitude <= 1: the adaptive value must be *exactly*
    // the historical constant, not merely close to it.
    let blocks = unit_pair(&EXACT, false);
    assert!(max_magnitude(&blocks) <= 1.0);
    let tol = adaptive_tolerance(&blocks);
    assert_eq!(
        tol.to_bits(),
        TOL_FLOOR.to_bits(),
        "unit-scale mesh must keep the historical 1e-6 exactly, got {tol:e}"
    );
}

#[test]
fn ordinary_magnitudes_stay_at_or_barely_above_the_legacy_tolerance() {
    // The empirically established crossover for real quantization noise sits
    // between coordinate magnitude 5 and 10, so the tolerance is expected to
    // hold at 1e-6 through magnitude 1 and to have started tracking the
    // coordinate magnitude by magnitude 10. It must never *drop* below 1e-6.
    for &mag in &[0.5 as Float, 1.0, 2.0, 5.0, 10.0, 20.0] {
        let blocks = stacked_pair(
            mag,
            0.0,
            0.0,
            [0.25, 0.25, 0.25],
            [5, 4, 3],
            &EXACT,
            false,
            0.0,
        );
        let tol = adaptive_tolerance(&blocks);
        assert!(
            tol >= TOL_FLOOR,
            "tolerance must never be tighter than the legacy 1e-6 (mag {mag}, tol {tol:e})"
        );
        let scale = max_magnitude(&blocks);
        assert!(
            tol <= (1e-6 as Float) * scale.max(1.0) * 1.000_001,
            "tolerance must not exceed the relative noise model (mag {mag}, tol {tol:e})"
        );
    }
    // Magnitude 1 is the exact break-even of the 1e-6-relative model.
    assert_eq!(adaptive_tolerance(&unit_pair(&EXACT, false)), TOL_FLOOR);
}

#[test]
fn tolerance_is_bounded_by_a_quarter_of_the_node_spacing() {
    // A mesh far from the origin but finely resolved: the relative-noise
    // model alone would ask for 1e-6 * 6.4e6 = 6.4, which is 25x the node
    // spacing.  The spacing ceiling must clamp it.
    let blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [0.25, 0.25, 0.25],
        [5, 4, 3],
        &EXACT,
        false,
        0.0,
    );
    let h = min_spacing(&blocks);
    let tol = adaptive_tolerance(&blocks);
    assert!(
        tol <= (0.25 as Float) * h,
        "tolerance {tol:e} must not exceed a quarter of the node spacing {h:e}"
    );
    assert!(tol >= TOL_FLOOR);
}

#[test]
fn spacing_ceiling_never_tightens_below_the_legacy_tolerance() {
    // Node spacing (1e-7) is far finer than 4 * 1e-6, so a bare `0.25 * h`
    // ceiling would be 2.5e-8 — tighter than working code. The ceiling is
    // floored at 1e-6 precisely to stop that.
    let blocks = stacked_pair(
        50.0,
        0.0,
        0.0,
        [1e-7, 1e-7, 1e-7],
        [5, 4, 3],
        &EXACT,
        false,
        0.0,
    );
    assert!(min_spacing(&blocks) < (4e-6 as Float));
    assert_eq!(
        adaptive_tolerance(&blocks).to_bits(),
        TOL_FLOOR.to_bits(),
        "a mesh finer than 4e-6 must stay at exactly the legacy 1e-6"
    );
}

#[test]
// ECEF-magnitude coordinates cannot be represented at all when the crate is
// compiled with `Float = f32`, so this scenario is f64-only.
#[cfg(not(feature = "f32"))]
fn collapsed_edge_falls_back_to_the_legacy_tolerance() {
    // A block with a *nearly* collapsed edge (an O-grid pole line whose two
    // stored copies differ by round-off) pins `h` at that round-off, so the
    // adaptive widening deliberately does not engage. Documented fail-safe
    // behaviour: never wrong, only conservative.
    let mut blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &EXACT,
        false,
        0.0,
    );
    assert!(
        adaptive_tolerance(&blocks) > TOL_FLOOR,
        "sanity: without the collapsed edge this mesh does widen"
    );
    // Collapse (i=0,j=0,k=0) onto (i=1,j=0,k=0) to within round-off.
    let b = &mut blocks[0];
    let target = b.xyz(1, 0, 0);
    let idx = b.idx(0, 0, 0);
    b.x[idx] = target.0 * (1.0 + (1e-13 as Float));
    b.y[idx] = target.1;
    b.z[idx] = target.2;
    assert_eq!(
        adaptive_tolerance(&blocks).to_bits(),
        TOL_FLOOR.to_bits(),
        "a near-collapsed edge must fall back to the legacy 1e-6"
    );
}

// ---------------------------------------------------------------------------
// The confirmed gap: interfaces the fixed 1e-6 misses
// ---------------------------------------------------------------------------

/// Assert the "before" baseline (fixed 1e-6 misses) and the "after"
/// behaviour (adaptive tolerance finds exactly one full-face match).
fn assert_gap_is_closed(blocks: &[Block], what: &str) {
    let discrepancy = interface_discrepancy(blocks);
    let scale = max_magnitude(blocks);
    let tol = adaptive_tolerance(blocks);
    println!(
        "{what}: max|coord|={scale:e} interface discrepancy={discrepancy:e} \
         legacy tol={TOL_FLOOR:e} adaptive tol={tol:e}"
    );

    // The injected noise is realistic *and* genuinely beyond the old
    // tolerance — otherwise this test would prove nothing.
    assert!(
        discrepancy > TOL_FLOOR,
        "{what}: fixture is not a repro — discrepancy {discrepancy:e} is within the legacy 1e-6"
    );

    // BEFORE: the exact pre-change behaviour, kept executable rather than
    // described in a comment.
    let (legacy_matches, legacy_outer) = connectivity_with_tol(blocks, TOL_FLOOR);
    assert!(
        legacy_matches.is_empty(),
        "{what}: the fixed 1e-6 tolerance was expected to MISS this interface, \
         but found {} match(es)",
        legacy_matches.len()
    );
    assert_eq!(
        legacy_outer.len(),
        12,
        "{what}: with the interface missed, all 12 block faces stay outer"
    );

    // AFTER: one full-face match, every node paired.
    let (matches, outer) = connectivity(blocks);
    assert_eq!(
        matches.len(),
        1,
        "{what}: expected exactly one interface match, got {}",
        matches.len()
    );
    let expected_points = blocks[0].jmax * blocks[0].kmax;
    assert_eq!(
        matches[0].points.len(),
        expected_points,
        "{what}: the whole shared face should be paired"
    );
    assert_eq!(outer.len(), 10, "{what}: 12 faces minus the matched pair");
}

#[test]
#[cfg(not(feature = "f32"))]
fn vspt_scale_ascii_quantization_gap_is_closed() {
    // Coordinate magnitude ~22, the regime of the repo's own VSPT fixture
    // (real coordinates -21.77 .. 47.80), stored as 7-significant-digit
    // ASCII. The decimal quantum there is 1e-5 — ten times the old
    // tolerance — so a one-quantum disagreement between the two copies of
    // the interface is invisible to a fixed 1e-6.
    let blocks = stacked_pair(
        20.0,
        0.0,
        0.0,
        [0.5, 0.25, 0.25],
        [5, 4, 3],
        &ASCII7,
        true,
        0.0,
    );
    assert_gap_is_closed(&blocks, "VSPT-scale / ASCII 7 significant digits");
}

#[test]
#[cfg(not(feature = "f32"))]
fn ecef_scale_float32_gap_is_closed() {
    // Coordinate magnitude ~6.4e6 (ECEF metres), stored in single
    // precision: one `f32` ulp there is 0.5 m, five *million* times the old
    // tolerance.
    let blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &F32,
        true,
        0.0,
    );
    assert_gap_is_closed(&blocks, "ECEF-scale / float32 round-trip");
}

#[test]
#[cfg(not(feature = "f32"))]
fn ecef_scale_ascii_quantization_gap_is_closed() {
    let blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &ASCII7,
        true,
        0.0,
    );
    assert_gap_is_closed(&blocks, "ECEF-scale / ASCII 7 significant digits");
}

// ---------------------------------------------------------------------------
// No-op and no-false-positive guarantees
// ---------------------------------------------------------------------------

#[test]
fn ordinary_small_mesh_is_completely_unaffected() {
    // An order-one mesh must produce not just the same tolerance but the
    // same connectivity result as the pre-change code.
    for storage in [&EXACT, &F32, &ASCII7] {
        let blocks = unit_pair(storage, true);
        assert!(max_magnitude(&blocks) <= 1.0);
        assert_eq!(adaptive_tolerance(&blocks).to_bits(), TOL_FLOOR.to_bits());

        let (new_m, new_o) = connectivity(&blocks);
        let (old_m, old_o) = connectivity_with_tol(&blocks, TOL_FLOOR);
        assert_eq!(new_m.len(), old_m.len());
        assert_eq!(new_o.len(), old_o.len());
        assert_eq!(new_m.len(), 1, "the clean interface should still match");
        for (a, b) in new_m.iter().zip(old_m.iter()) {
            assert_eq!(a.block1.bounds(), b.block1.bounds());
            assert_eq!(a.block2.bounds(), b.block2.bounds());
            assert_eq!(a.points.len(), b.points.len());
        }
    }
}

#[test]
#[cfg(not(feature = "f32"))]
fn genuinely_separated_blocks_still_do_not_match_at_large_magnitude() {
    // The widened tolerance must not turn a real gap into an interface.
    // The gap here is one full cell (10 m) at ECEF magnitude, which is four
    // times the widened tolerance the spacing ceiling permits.
    let blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &F32,
        true,
        10.0,
    );
    let tol = adaptive_tolerance(&blocks);
    assert!(
        tol < (10.0 as Float),
        "sanity: tolerance {tol:e} must be under the gap"
    );
    let (matches, outer) = connectivity(&blocks);
    assert!(
        matches.is_empty(),
        "blocks separated by a full cell must not match, got {} match(es)",
        matches.len()
    );
    assert_eq!(outer.len(), 12);
}

#[test]
#[cfg(not(feature = "f32"))]
fn sub_tolerance_gap_is_matched_only_because_it_is_within_storage_noise() {
    // Companion to the test above, pinning where the boundary sits: a gap
    // *smaller* than one storage quantum is indistinguishable from noise
    // and is (correctly) matched; the previous test's full-cell gap is not.
    let blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &F32,
        false,
        0.25,
    );
    let tol = adaptive_tolerance(&blocks);
    assert!(tol > (0.25 as Float));
    let (matches, _) = connectivity(&blocks);
    assert_eq!(matches.len(), 1);
}

// ---------------------------------------------------------------------------
// Audit additions
// ---------------------------------------------------------------------------

#[test]
fn exactly_collapsed_edge_is_skipped_and_does_not_suppress_widening() {
    // Companion to `collapsed_edge_falls_back_to_the_legacy_tolerance`, which
    // only covers the *near*-zero case. An O-grid pole line whose two stored
    // endpoints are bit-identical is the common case, and it must be skipped
    // outright so the adaptive widening still engages for the rest of the
    // mesh. (Near-zero-but-nonzero is the documented fail-safe: it pins the
    // tolerance back at TOL_FLOOR. Both behaviours are deliberate.)
    let mut blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &EXACT,
        false,
        0.0,
    );
    let widened = adaptive_tolerance(&blocks);
    assert!(
        widened > TOL_FLOOR,
        "sanity: this mesh widens before collapse"
    );

    // Collapse (0,0,0) onto (1,0,0) *exactly* — a zero-length edge.
    let b = &mut blocks[0];
    let target = b.xyz(1, 0, 0);
    let idx = b.idx(0, 0, 0);
    b.x[idx] = target.0;
    b.y[idx] = target.1;
    b.z[idx] = target.2;

    assert_eq!(
        adaptive_tolerance(&blocks).to_bits(),
        widened.to_bits(),
        "an exactly-zero-length collapsed edge must be skipped, leaving the \
         tolerance exactly as it was"
    );
}

#[test]
fn a_fine_block_elsewhere_can_only_clamp_to_the_floor_never_below_it() {
    // Item-2 regression guard. `h` is a single global minimum over the whole
    // mesh, so one very fine block bounds the tolerance everywhere. That must
    // only ever *limit growth*: a coarse, far-from-origin block sharing a mesh
    // with a fine block must still get at least the historical 1e-6, never a
    // value tightened below it by the fine block's spacing.
    let mut blocks = stacked_pair(
        6.4e6,
        0.0,
        0.0,
        [10.0, 10.0, 10.0],
        [5, 4, 3],
        &EXACT,
        false,
        0.0,
    );
    assert!(
        adaptive_tolerance(&blocks) > TOL_FLOOR,
        "sanity: widens alone"
    );

    // Append an unrelated, extremely fine block. 0.25 * 1e-9 = 2.5e-10,
    // which is 4000x tighter than the floor.
    blocks.push(make_block(
        0.0,
        0.0,
        0.0,
        [1e-9, 1e-9, 1e-9],
        [4, 4, 4],
        false,
        false,
        &EXACT,
    ));
    assert!(min_spacing(&blocks) < (4e-6 as Float));

    let tol = adaptive_tolerance(&blocks);
    assert_eq!(
        tol.to_bits(),
        TOL_FLOOR.to_bits(),
        "a fine block must clamp growth back to exactly the floor, not below \
         it (got {tol:e})"
    );
    assert!(tol >= TOL_FLOOR);
}

#[test]
fn connectivity_fast_uses_the_full_resolution_tolerance() {
    // `connectivity_fast` matches on a GCD-reduced copy of the mesh and
    // scales the result back up without re-verifying it. Reduction leaves
    // the storage noise unchanged but multiplies the cell size by the GCD,
    // so a tolerance derived from the *reduced* grid would have a ceiling
    // `gcd` times looser — and would accept genuinely separated faces that
    // `connectivity` rejects. Pin that `connectivity_fast` uses the
    // full-resolution tolerance instead, by placing a gap between the two
    // candidate values.
    //
    // max|coord| ~50 -> noise 5e-5. Cells 1e-4 -> full ceiling 2.5e-5.
    // dims 9x5x5 -> gcd_three(8, 4, 4) == 4 -> reduced cells 4e-4 -> reduced
    // ceiling 1e-4, so the reduced-grid tolerance would be the full 5e-5.
    let gap = 3.5e-5 as Float;
    let blocks = stacked_pair(
        50.0,
        0.0,
        0.0,
        [1e-4, 1e-4, 1e-4],
        [9, 5, 5],
        &EXACT,
        false,
        gap,
    );
    let reduced = reduce_blocks(&blocks, 4);
    let tol_full = adaptive_tolerance(&blocks);
    let tol_reduced = adaptive_tolerance(&reduced);
    println!("gap={gap:e} full tol={tol_full:e} reduced-grid tol={tol_reduced:e}");
    assert!(
        tol_full < gap && gap < tol_reduced,
        "fixture must sit between the two candidate tolerances"
    );

    // The reduced-grid tolerance really would have accepted the gap.
    let (would_match, _) = connectivity_with_tol(&reduced, tol_reduced);
    assert_eq!(
        would_match.len(),
        1,
        "sanity: the looser reduced-grid tolerance accepts a {gap:e} gap"
    );

    // Neither real entry point does.
    let (full_m, full_o) = connectivity(&blocks);
    let (fast_m, fast_o) = connectivity_fast(&blocks);
    assert!(
        full_m.is_empty(),
        "connectivity must reject a gap of {gap:e}"
    );
    assert!(
        fast_m.is_empty(),
        "connectivity_fast must reject the same gap, got {} match(es)",
        fast_m.len()
    );
    assert_eq!(full_o.len(), 12);
    assert_eq!(fast_o.len(), 12);

    // And with the gap closed, both still find the interface identically.
    let touching = stacked_pair(
        50.0,
        0.0,
        0.0,
        [1e-4, 1e-4, 1e-4],
        [9, 5, 5],
        &EXACT,
        false,
        0.0,
    );
    let (full_m, _) = connectivity(&touching);
    let (fast_m, _) = connectivity_fast(&touching);
    assert_eq!(full_m.len(), 1);
    assert_eq!(fast_m.len(), 1);
    assert_eq!(full_m[0].block1.bounds(), fast_m[0].block1.bounds());
    assert_eq!(full_m[0].block2.bounds(), fast_m[0].block2.bounds());
}

#[test]
fn sheared_cells_bound_the_tolerance_by_the_short_diagonal() {
    // On a cell sheared to a 10° included angle the short face diagonal is
    // sqrt(2 - 2 cos 10°) ≈ 0.174 edge lengths — under a quarter of an
    // edge. With coordinates near 50 in f32 (quantum q ≈ 3.8e-6) and cells
    // of 7q, an edge-only ceiling of 0.25 * edge ≈ 1.75q lets a
    // one-quantum-per-axis nudge (≈ 1.73q) push the true partner just
    // *outside* the tolerance while the diagonal corner (≈ 1.2q away) is
    // inside it. Phase 2's nearest-node pairing then has exactly one
    // candidate — the wrong one — and emits a face match whose interior
    // nodes are paired diagonally. Found by an adversarial sweep; the old
    // fixed 1e-6 tolerance reported no match at all here, so widening the
    // tolerance must not turn "missed" into "wrong".
    let theta = (10.0 as Float).to_radians();
    let q = f32_quantum(50.0);
    let h = 7.0 * q;
    let blocks = sheared_pair(50.0, h, theta, [5, 5, 5], &F32, true);
    let short_diag = h * (2.0 - 2.0 * theta.cos()).sqrt();
    assert!(
        short_diag < 0.25 * h,
        "sanity: the diagonal is the closest corner"
    );

    // BEFORE: the edge-only ceiling this crate shipped with first. Built
    // from the test's own edge-length helper so the number is independent
    // of the implementation under test.
    let edge_only_tol = ((1e-6 as Float) * max_magnitude(&blocks))
        .min(0.25 * min_spacing(&blocks))
        .max(TOL_FLOOR);
    let (bad, _) = connectivity_with_tol(&blocks, edge_only_tol);
    println!(
        "edge-only tol={edge_only_tol:e} -> {} match(es), {} mis-paired nodes",
        bad.len(),
        mispaired(&bad)
    );
    assert!(
        !bad.is_empty() && mispaired(&bad) > 0,
        "fixture is not a repro: an edge-only ceiling was expected to pair \
         nodes with the diagonal corner"
    );

    // AFTER: the ceiling is a quarter of the *shortest corner-to-corner*
    // distance, so it drops to the floor here and nothing is mis-paired.
    // (The true partners sit outside the tolerance, so the honest answer for
    // this mesh is "no match", and that is what it reports.)
    let tol = adaptive_tolerance(&blocks);
    assert!(
        tol <= (0.25 * short_diag).max(TOL_FLOOR) * 1.000_001,
        "tolerance {tol:e} must be bounded by a quarter of the short diagonal {short_diag:e}"
    );
    let (matches, _) = connectivity(&blocks);
    assert_eq!(
        mispaired(&matches),
        0,
        "no node may be paired with a neighbour of its true partner"
    );
    assert!(
        matches.is_empty(),
        "the true partners are out of tolerance here"
    );
}

#[test]
fn shear_is_harmless_when_cells_are_well_resolved_by_storage() {
    // Companion: the same 10° shear with 300 quanta per cell. The short
    // diagonal (≈ 52q) is now far longer than the noise (≈ 1.7q), so the
    // ceiling does not bind, the interface is found, and every node is
    // paired with its true partner.
    let theta = (10.0 as Float).to_radians();
    let q = f32_quantum(50.0);
    let blocks = sheared_pair(50.0, 300.0 * q, theta, [5, 5, 5], &F32, true);
    let (matches, outer) = connectivity(&blocks);
    assert_eq!(matches.len(), 1);
    assert_eq!(outer.len(), 10);
    assert_eq!(matches[0].points.len(), 25);
    assert_eq!(mispaired(&matches), 0);
}
