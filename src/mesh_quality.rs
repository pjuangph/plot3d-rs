//! Structured-grid mesh-quality battery.
//!
//! A Rust port of the CFD-readiness quality checks in
//! `tgs-py-grc/python/tgs_py/quality/` (`metrics_3d.py`, `checks_3d.py`,
//! `thresholds.py`). The math is transcribed faithfully from that
//! reference; the additions here are **block handedness** (left-handed /
//! negative-Jacobian detection + fix) and **per-cell negative-volume
//! detection** — the Python reference `.abs()`'s the cell volume and so
//! cannot see either.
//!
//! Per-cell metric kernels operate on a [`Block`]; each reported
//! [`Violation`] carries a [`CellLocation`] (block index, `i/j/k`, and
//! the cell centroid) so a caller can point the user straight at the
//! bad cell.
//!
//! Severity policy (caller-facing): negative / degenerate cell volume is
//! `Error` (it breaks the finite-volume discretization); skewness,
//! aspect ratio, and orthogonality are `Warn` (the solver runs on an
//! imperfect mesh, the user just needs to know where). Left-handed
//! blocks are *fixable* — [`make_right_handed`] flips them.

use crate::{Block, Float};

// =============================================================================
// vec3 helpers
// =============================================================================

#[inline]
fn sub(a: [Float; 3], b: [Float; 3]) -> [Float; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [Float; 3], b: [Float; 3]) -> Float {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: [Float; 3], b: [Float; 3]) -> [Float; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: [Float; 3]) -> Float {
    dot(a, a).sqrt()
}

// =============================================================================
// Per-cell metric kernels — port of tgs-py-grc `metrics_3d.py`
// =============================================================================

/// The three edge vectors `(e_i, e_j, e_k)` emanating from corner
/// `(i, j, k)` of cell `(i, j, k)`. Matches `metrics_3d.cell_edges`:
/// `e_i = P(i+1,j,k) - P(i,j,k)`, etc.
fn cell_edges(
    b: &Block,
    i: usize,
    j: usize,
    k: usize,
) -> ([Float; 3], [Float; 3], [Float; 3]) {
    let p = |ii: usize, jj: usize, kk: usize| -> [Float; 3] {
        let (x, y, z) = b.xyz(ii, jj, kk);
        [x, y, z]
    };
    let p0 = p(i, j, k);
    (
        sub(p(i + 1, j, k), p0),
        sub(p(i, j + 1, k), p0),
        sub(p(i, j, k + 1), p0),
    )
}

/// Centroid of cell `(i, j, k)` — the average of its 8 corner nodes.
fn cell_centroid(b: &Block, i: usize, j: usize, k: usize) -> [Float; 3] {
    let mut c = [0.0 as Float; 3];
    for &di in &[0usize, 1] {
        for &dj in &[0usize, 1] {
            for &dk in &[0usize, 1] {
                let (x, y, z) = b.xyz(i + di, j + dj, k + dk);
                c[0] += x;
                c[1] += y;
                c[2] += z;
            }
        }
    }
    [c[0] / 8.0, c[1] / 8.0, c[2] / 8.0]
}

/// Signed cell volume `e_i · (e_j × e_k)` (the scalar triple product —
/// proportional to the hex volume). The **sign is the handedness /
/// Jacobian indicator**: `> 0` right-handed, `< 0` left-handed, `≈ 0`
/// degenerate. `metrics_3d.cell_volume` returns the `.abs()` of this.
pub fn cell_signed_volume(b: &Block, i: usize, j: usize, k: usize) -> Float {
    let (ei, ej, ek) = cell_edges(b, i, j, k);
    dot(ei, cross(ej, ek))
}

/// True hexahedral cell volume by the **divergence theorem**,
/// `V = (1/3) Σ_faces (r_c · S)`, with each quad face's area vector taken
/// as `0.5 (d1 × d2)` from its diagonals (the standard treatment for a
/// non-planar quad). This is the SAME operator the solver integrates with
/// (`metrics::compute_cell_volumes`), so it is the authority on whether a
/// cell has usable volume.
///
/// Contrast [`cell_signed_volume`], which anchors on ONE corner and is a
/// handedness indicator: on a cell with a collapsed edge at that corner it
/// returns exactly `0` even though the cell has real, positive volume.
/// Measured on the reference Rotor 35 tip block, the fold cells return
/// `0.0`/`-0.0` from the triple product and `+1e-17` from this function.
pub fn cell_volume_divergence(b: &Block, i: usize, j: usize, k: usize) -> Float {
    let p = |ii: usize, jj: usize, kk: usize| -> [Float; 3] {
        let (x, y, z) = b.xyz(ii, jj, kk);
        [x, y, z]
    };
    let n = [
        p(i, j, k),
        p(i + 1, j, k),
        p(i, j + 1, k),
        p(i + 1, j + 1, k),
        p(i, j, k + 1),
        p(i + 1, j, k + 1),
        p(i, j + 1, k + 1),
        p(i + 1, j + 1, k + 1),
    ];
    // Each face as 4 corner indices, wound so the area vector points OUT.
    const FACES: [[usize; 4]; 6] = [
        [0, 4, 6, 2], // i-low
        [1, 3, 7, 5], // i-high
        [0, 1, 5, 4], // j-low
        [2, 6, 7, 3], // j-high
        [0, 2, 3, 1], // k-low
        [4, 5, 7, 6], // k-high
    ];
    let mut v = 0.0 as Float;
    for f in FACES.iter() {
        let (a, bb, c, d) = (n[f[0]], n[f[1]], n[f[2]], n[f[3]]);
        let centroid = [
            (a[0] + bb[0] + c[0] + d[0]) * 0.25,
            (a[1] + bb[1] + c[1] + d[1]) * 0.25,
            (a[2] + bb[2] + c[2] + d[2]) * 0.25,
        ];
        let s = cross(sub(c, a), sub(d, bb));
        v += dot(centroid, [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5]);
    }
    v / 3.0
}

/// Whether cell `(i, j, k)` has at least one pair of **coincident corner
/// nodes** — i.e. it sits on a collapsed/pinched grid line.
///
/// Such lines are a deliberate, standard turbomachinery construction (an
/// O-grid folded onto a blade-tip camber line; the analogue of a C-grid
/// wake cut), so their presence is a fact about the grid, not damage. The
/// nodes are typically bit-identical, which is why an exact comparison is
/// the right test — a tolerance would sweep in merely-close nodes.
pub fn cell_has_collapsed_edge(b: &Block, i: usize, j: usize, k: usize) -> bool {
    let mut n = [[0.0 as Float; 3]; 8];
    let mut m = 0;
    for &dk in &[0usize, 1] {
        for &dj in &[0usize, 1] {
            for &di in &[0usize, 1] {
                let (x, y, z) = b.xyz(i + di, j + dj, k + dk);
                n[m] = [x, y, z];
                m += 1;
            }
        }
    }
    for a in 0..8 {
        for c in (a + 1)..8 {
            if n[a] == n[c] {
                return true;
            }
        }
    }
    false
}

/// Per-cell aspect ratio `max(edge_len) / min(edge_len)` over the three
/// edge vectors. Always `>= 1`; a perfect cube returns `1.0`. Port of
/// `metrics_3d.cell_aspect_ratio`.
///
/// **A cell with a zero-length edge has NO aspect ratio, and this returns
/// [`Float::INFINITY`] for it rather than a finite number.** That cell sits
/// on a collapsed grid line — see [`cell_has_collapsed_edge`], which
/// documents why such lines are a deliberate turbomachinery construction and
/// not damage — and the condition belongs to the degenerate-cell check, which
/// already reports it by name.
///
/// This previously floored the denominator at `1e-30`, which did not avoid the
/// problem so much as disguise it: a legitimate 13 µm edge over a collapsed
/// one returned `1.3e25`, a precise-looking number that is pure arithmetic
/// artefact. Measured on rotor35 (2.94 M cells, 0.5 µm first cell at
/// `wall_spacing: 5.0e-7`) it was reported as `wall aspect ratio
/// 13045882770553017259261952 > 100000` — twenty orders of magnitude beyond
/// anything the geometry can produce, and reported as an ASPECT-RATIO fault
/// on a mesh whose real and already-reported condition was 960 cells on a
/// collapsed line. `INFINITY` still exceeds any threshold, so nothing is
/// silenced; a reader just sees a degeneracy instead of believing a number.
pub fn cell_aspect_ratio(b: &Block, i: usize, j: usize, k: usize) -> Float {
    let (ei, ej, ek) = cell_edges(b, i, j, k);
    let (li, lj, lk) = (norm(ei), norm(ej), norm(ek));
    let mx = li.max(lj).max(lk);
    let mn = li.min(lj).min(lk);
    if mn == 0.0 {
        return Float::INFINITY;
    }
    mx / mn
}

/// Acute angle in degrees between two edge vectors, in `[0, 90]`
/// (`degrees(arccos(|cos θ|))`). Port of the `_angle` closure in
/// `metrics_3d.cell_orthogonality`.
fn edge_angle_deg(a: [Float; 3], b: [Float; 3]) -> Float {
    let denom = (norm(a) * norm(b)).max(1e-30 as Float);
    let cos_t = (dot(a, b) / denom).abs().clamp(0.0, 1.0);
    cos_t.acos().to_degrees()
}

/// Per-cell equiangle skewness in degrees: `90° − min(orthogonality)`,
/// where each orthogonality angle is [`edge_angle_deg`] of an edge-vector
/// pair. `0°` = perfectly orthogonal, `90°` = degenerate. Port of
/// `metrics_3d.cell_skewness`.
pub fn cell_skewness(b: &Block, i: usize, j: usize, k: usize) -> Float {
    let (ei, ej, ek) = cell_edges(b, i, j, k);
    let ortho_min = edge_angle_deg(ei, ej)
        .min(edge_angle_deg(ej, ek))
        .min(edge_angle_deg(ei, ek));
    90.0 - ortho_min
}

/// Cell counts `(nci, ncj, nck)` for a block. Any of them is `0` when
/// the corresponding node dimension is `< 2` (the block has no cells in
/// that direction — e.g. a 2D block has `nck == 0`).
fn cell_dims(b: &Block) -> (usize, usize, usize) {
    (
        b.imax.saturating_sub(1),
        b.jmax.saturating_sub(1),
        b.kmax.saturating_sub(1),
    )
}

/// Compute a per-cell scalar field over a block, cell-indexed
/// `(k * ncj + j) * nci + i`. Returns an empty `Vec` for a block with no
/// cells (any cell dimension `0`).
fn cell_field<F: Fn(&Block, usize, usize, usize) -> Float>(
    b: &Block,
    f: F,
) -> Vec<Float> {
    let (nci, ncj, nck) = cell_dims(b);
    if nci == 0 || ncj == 0 || nck == 0 {
        return Vec::new();
    }
    let mut field = vec![0.0 as Float; nci * ncj * nck];
    for k in 0..nck {
        for j in 0..ncj {
            for i in 0..nci {
                field[(k * ncj + j) * nci + i] = f(b, i, j, k);
            }
        }
    }
    field
}

// =============================================================================
// Block handedness — left-handed / negative-Jacobian detection + fix
// =============================================================================

/// Handedness of a block, from the sign of its median signed cell volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    /// Median signed volume `> 0` — the normal, FV-ready orientation.
    RightHanded,
    /// Median signed volume `< 0` — the block's `(i,j,k)` indexing is
    /// mirror-flipped; [`make_right_handed`] fixes it by flipping one axis.
    LeftHanded,
    /// Median signed volume `≈ 0`, or the block has no cells — flipping
    /// cannot help; the per-cell negative/degenerate-volume checks flag it.
    Degenerate,
}

/// Classify a block's handedness from the **median** signed cell volume
/// (median is robust to a handful of locally-bad cells). A left-handed
/// block has essentially every cell at negative signed volume.
pub fn block_handedness(b: &Block) -> Handedness {
    let mut sv = cell_field(b, cell_signed_volume);
    if sv.is_empty() {
        return Handedness::Degenerate;
    }
    sv.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
    let median = median_sorted(&sv);
    // Scale the "≈ 0" tolerance by the typical cell size so it works for
    // meshes in any units.
    let typical = sv[sv.len() - 1].abs().max(sv[0].abs());
    let eps = typical * 1e-12 + Float::MIN_POSITIVE;
    if median > eps {
        Handedness::RightHanded
    } else if median < -eps {
        Handedness::LeftHanded
    } else {
        Handedness::Degenerate
    }
}

/// If `block` is left-handed, return a right-handed copy (one structured
/// axis reversed) and `Some(flipped_axis)`; otherwise return a clone and
/// `None`. Flipping one axis is a reflection — it negates every cell's
/// signed volume, so a left-handed block becomes right-handed. The
/// geometry (cells, volumes, faces) is unchanged — only the `(i,j,k)`
/// traversal direction — so this is physics-neutral relabeling.
///
/// A `Degenerate` block is returned unchanged (`None`) — flipping cannot
/// help; [`run_all`]'s negative/degenerate-volume checks report it.
pub fn make_right_handed(block: &Block) -> (Block, Option<usize>) {
    if block_handedness(block) != Handedness::LeftHanded {
        return (block.clone(), None);
    }
    // Flip the first structured axis that actually has cells.
    let axis = if block.imax > 1 {
        0
    } else if block.jmax > 1 {
        1
    } else {
        2
    };
    let mut x = block.x.clone();
    let mut y = block.y.clone();
    let mut z = block.z.clone();
    crate::block_analysis::flip_block_axis(
        &mut x,
        &mut y,
        &mut z,
        (block.imax, block.jmax, block.kmax),
        axis,
    );
    (
        Block::new(block.imax, block.jmax, block.kmax, x, y, z),
        Some(axis),
    )
}

// =============================================================================
// Thresholds — port of tgs-py-grc `thresholds.py` (core fields only)
// =============================================================================

/// CFD-readiness thresholds for the core quality checks. Three named
/// presets — [`Thresholds::STRICT`], [`Thresholds::STANDARD`],
/// [`Thresholds::RELAXED`] — ported from `tgs-py-grc/.../thresholds.py`.
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// Per-cell skewness (deg): the 99th-percentile cell limit.
    pub skew_p99_deg: Float,
    /// Per-cell skewness (deg): the absolute-worst cell limit.
    pub skew_max_deg: Float,
    /// Minimum interior-edge orthogonality angle (deg); `90°` = perfect.
    pub min_orthogonality_deg: Float,
    /// Maximum aspect ratio for interior cells.
    pub max_ar_interior: Float,
    /// Maximum aspect ratio for wall first-cells (BL legitimately high).
    pub max_ar_wall: Float,
    /// Minimum cell volume relative to the block median — catches
    /// near-degenerate cells that survive a sign check.
    pub min_cell_volume_ratio: Float,
    /// Cell layers dropped from each axis-endpoint before computing the
    /// skewness percentiles (excludes wall first-cells from the stats).
    pub boundary_drop: usize,
}

impl Thresholds {
    /// Ship-quality CFD mesh bar.
    pub const STRICT: Thresholds = Thresholds {
        skew_p99_deg: 50.0,
        skew_max_deg: 60.0,
        min_orthogonality_deg: 30.0,
        max_ar_interior: 100_000.0,
        max_ar_wall: 200_000.0,
        min_cell_volume_ratio: 1e-5,
        boundary_drop: 2,
    };
    /// Day-to-day production bar — the default.
    ///
    /// skew_max / min_orthogonality (2026-07): tightened 90°→75° / 10°→15°
    /// in lockstep with tgs-py-grc's thresholds.py. skew_max=90° was
    /// vacuous (90° IS a degenerate cell). Anchor case: a cascade OH
    /// mesh's inlet-extension↔O-grid corner cell (skew 77.6°, min-ortho
    /// 12.38°) passed this preset silently, then blew up k-omega
    /// (omega/mu_t → NaN by iter 3) on BOTH GlennHT-Fortran (F64) and
    /// glennht-gpu. Solver-killing cells must at least WARN in the
    /// mesh-quality battery (they stay Warn severity — the solver still
    /// runs; see glennht-gpu mesh_diagnostics policy).
    pub const STANDARD: Thresholds = Thresholds {
        skew_p99_deg: 80.0,
        skew_max_deg: 75.0,
        min_orthogonality_deg: 15.0,
        max_ar_interior: 5_000.0,
        max_ar_wall: 100_000.0,
        min_cell_volume_ratio: 1e-6,
        boundary_drop: 2,
    };
    /// Prototype / exploratory bar — still flags genuinely broken meshes.
    pub const RELAXED: Thresholds = Thresholds {
        skew_p99_deg: 87.0,
        skew_max_deg: 90.0,
        min_orthogonality_deg: 1.0,
        max_ar_interior: 30_000.0,
        max_ar_wall: Float::INFINITY,
        min_cell_volume_ratio: 1e-9,
        boundary_drop: 2,
    };

    /// Resolve a case-insensitive preset name (`"strict"` / `"standard"`
    /// / `"relaxed"`). Unknown names fall back to `STANDARD`.
    pub fn from_preset_name(name: &str) -> Thresholds {
        match name.to_ascii_uppercase().as_str() {
            "STRICT" => Self::STRICT,
            "RELAXED" => Self::RELAXED,
            _ => Self::STANDARD,
        }
    }
}

impl Default for Thresholds {
    fn default() -> Self {
        Self::STANDARD
    }
}

// =============================================================================
// Violation model — port of tgs-py-grc `Violation`
// =============================================================================

/// Severity of a quality violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Advisory — the solver runs, accuracy may suffer.
    Warn,
    /// Fatal — breaks the finite-volume discretization (negative /
    /// degenerate cell volume).
    Error,
}

/// Where a violating cell lives — for pointing the user straight at it.
#[derive(Debug, Clone, Copy)]
pub struct CellLocation {
    /// Index of the block in the mesh's block list.
    pub block: usize,
    /// Structured cell index within the block.
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Cell centroid `(x, y, z)` — for locating the cell in space.
    pub centroid: [Float; 3],
}

/// A single mesh-quality threshold breach.
#[derive(Debug, Clone)]
pub struct Violation {
    /// Short check name — `"negative_volume"`, `"skewness"`,
    /// `"aspect_ratio"`, `"orthogonality"`, `"min_cell_volume"`.
    pub check: &'static str,
    /// Severity (`Error` for negative/degenerate volume, else `Warn`).
    pub severity: Severity,
    /// The observed value that breached the threshold.
    pub actual: Float,
    /// The threshold it breached.
    pub threshold: Float,
    /// The worst cell's location, when the check is per-cell.
    pub location: Option<CellLocation>,
    /// Human-readable one-liner.
    pub message: String,
}

// =============================================================================
// Statistics helpers — port of `_crop_boundary` / `_stats_dropped`
// =============================================================================

/// Median of an **already-sorted** slice, matching `np.median`: the
/// middle element for an odd count, the average of the two middle
/// elements for an even count. (`sorted[len/2]` alone would take the
/// upper-middle element on even-length inputs, diverging from NumPy.)
fn median_sorted(sorted: &[Float]) -> Float {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Linear-interpolation percentile (matches NumPy's default), `p` in `[0, 1]`.
fn percentile(values: &[Float], p: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = p * (v.len() - 1) as Float;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (pos - lo as Float) * (v[hi] - v[lo])
    }
}

/// `(p99, max, argmax_cell_index)` of a cell-indexed field, computed over
/// the boundary-cropped interior (mirrors `_stats_dropped`). The returned
/// index is in the *original* (uncropped) cell numbering.
///
/// Cropping is **all-or-nothing**, matching the Python `_crop_boundary`:
/// if any axis is too thin to drop `2*drop` layers, NO axis is cropped
/// (the reference `return arr`s the whole array the moment one axis
/// fails). Cropping each axis independently would change which cells the
/// p99 / max are taken over.
fn stats_dropped(
    field: &[Float],
    nci: usize,
    ncj: usize,
    nck: usize,
    drop: usize,
) -> (Float, Float, usize) {
    if field.is_empty() {
        return (0.0, 0.0, 0);
    }
    let can_crop =
        drop > 0 && nci > 2 * drop && ncj > 2 * drop && nck > 2 * drop;
    let span = |n: usize| -> (usize, usize) {
        if can_crop {
            (drop, n - drop)
        } else {
            (0, n)
        }
    };
    let (i0, i1) = span(nci);
    let (j0, j1) = span(ncj);
    let (k0, k1) = span(nck);

    let mut cropped: Vec<Float> = Vec::new();
    let mut max_val = Float::NEG_INFINITY;
    let mut max_idx = 0usize;
    for k in k0..k1 {
        for j in j0..j1 {
            for i in i0..i1 {
                let idx = (k * ncj + j) * nci + i;
                let v = field[idx];
                cropped.push(v);
                if v > max_val {
                    max_val = v;
                    max_idx = idx;
                }
            }
        }
    }
    if cropped.is_empty() {
        // Crop emptied everything — fall back to the full field.
        let (mut mv, mut mi) = (Float::NEG_INFINITY, 0usize);
        for (idx, &v) in field.iter().enumerate() {
            if v > mv {
                mv = v;
                mi = idx;
            }
        }
        return (percentile(field, 0.99), mv, mi);
    }
    (percentile(&cropped, 0.99), max_val, max_idx)
}

/// Decompose a flat cell index back into `(i, j, k)`.
fn cell_ijk(idx: usize, nci: usize, ncj: usize) -> (usize, usize, usize) {
    let i = idx % nci;
    let j = (idx / nci) % ncj;
    let k = idx / (nci * ncj);
    (i, j, k)
}

// =============================================================================
// Report
// =============================================================================

/// The full mesh-quality report for a multi-block mesh.
#[derive(Debug, Clone)]
pub struct MeshQualityReport {
    /// Threshold preset name used (`"STANDARD"` etc.).
    pub preset: String,
    /// Per-block handedness (index `b` = block `b`).
    pub handedness: Vec<Handedness>,
    /// Every threshold breach found, across all blocks.
    pub violations: Vec<Violation>,
}

impl MeshQualityReport {
    /// Number of `Error`-severity violations (negative/degenerate volume).
    pub fn n_error(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count()
    }

    /// Number of `Warn`-severity violations (skewness / AR / orthogonality).
    pub fn n_warn(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Warn)
            .count()
    }

    /// True when there are no `Error` violations — the mesh is
    /// FV-discretizable (warnings allowed).
    pub fn passes(&self) -> bool {
        self.n_error() == 0
    }

    /// Human-readable multi-line report — handedness summary, the
    /// violation counts, then each violation with its location.
    pub fn format_report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Mesh quality ({} preset): {} block(s), {} error(s), {} warning(s)\n",
            self.preset,
            self.handedness.len(),
            self.n_error(),
            self.n_warn(),
        ));
        let left: Vec<usize> = self
            .handedness
            .iter()
            .enumerate()
            .filter(|(_, h)| **h == Handedness::LeftHanded)
            .map(|(b, _)| b)
            .collect();
        let degen: Vec<usize> = self
            .handedness
            .iter()
            .enumerate()
            .filter(|(_, h)| **h == Handedness::Degenerate)
            .map(|(b, _)| b)
            .collect();
        if !left.is_empty() {
            s.push_str(&format!(
                "  left-handed blocks (need flipping): {left:?}\n"
            ));
        }
        if !degen.is_empty() {
            s.push_str(&format!("  degenerate blocks: {degen:?}\n"));
        }
        if left.is_empty() && degen.is_empty() {
            s.push_str("  all blocks right-handed\n");
        }
        for v in &self.violations {
            let sev = match v.severity {
                Severity::Error => "ERROR",
                Severity::Warn => "warn ",
            };
            match &v.location {
                Some(loc) => s.push_str(&format!(
                    "  [{sev}] {} — block {} cell ({},{},{}) at \
                     ({:.4},{:.4},{:.4}): {}\n",
                    v.check,
                    loc.block,
                    loc.i,
                    loc.j,
                    loc.k,
                    loc.centroid[0],
                    loc.centroid[1],
                    loc.centroid[2],
                    v.message,
                )),
                None => {
                    s.push_str(&format!("  [{sev}] {}: {}\n", v.check, v.message))
                }
            }
        }
        s
    }
}

// =============================================================================
// Checks — port of tgs-py-grc `checks_3d.py` (core checks) + negative volume
// =============================================================================

/// Max per-cell negative-volume violations listed individually before
/// the report collapses the rest into a "+N more" line.
const MAX_LISTED_NEGATIVE: usize = 64;

/// Run the full quality battery on a multi-block mesh.
///
/// Per block (blocks with fewer than 2 nodes on any axis are skipped —
/// they have no cells): handedness classification, then the per-cell
/// checks — negative signed volume, degenerate (min-volume / median),
/// equiangle skewness, minimum orthogonality, aspect ratio. Negative /
/// degenerate volume → `Error`; skewness / orthogonality / aspect ratio
/// → `Warn`.
///
/// `run_all` does not mutate the blocks and never panics — it returns the
/// report and the caller decides what is fatal. Apply [`make_right_handed`]
/// to fix left-handed blocks *before* calling this so the handedness
/// column reads clean.
pub fn run_all(blocks: &[Block], t: &Thresholds, preset_name: &str) -> MeshQualityReport {
    let mut handedness = Vec::with_capacity(blocks.len());
    let mut violations: Vec<Violation> = Vec::new();

    for (bi, b) in blocks.iter().enumerate() {
        handedness.push(block_handedness(b));

        let (nci, ncj, nck) = cell_dims(b);
        if nci == 0 || ncj == 0 || nck == 0 {
            continue; // no cells — nothing to score
        }

        // --- negative signed volume (per cell) ---
        //
        // Two DIFFERENT defects hide behind "signed volume <= 0", and
        // conflating them makes a legitimate reference grid unusable:
        //
        //   INVERTED   — the cell is genuinely turned inside out. The
        //                divergence-theorem volume is negative. Fatal:
        //                it breaks the finite-volume discretization.
        //   DEGENERATE — `cell_signed_volume` is <= 0 but the cell still has
        //                real positive volume. NOT fatal: the grid is valid
        //                and the reference solver ran on it. Two distinct
        //                causes, and BOTH must be tolerated:
        //                  (a) collapsed/pinched grid line — the anchor
        //                      corner has a zero-length edge, so the triple
        //                      product is exactly 0 (O-grid pinch lines).
        //                  (b) corner warp on a high-aspect-ratio cell —
        //                      `cell_signed_volume` is a ONE-CORNER triple
        //                      product, exact only on a parallelepiped. On a
        //                      warped hex it measures the anchor corner's
        //                      Jacobian, which can go slightly negative while
        //                      the cell's true volume is comfortably positive.
        //                      Wall-resolved boundary layers produce exactly
        //                      this: 0.5 um first cells at AR ~1e5.
        //
        // We therefore adjudicate with `cell_volume_divergence` (the
        // operator the solver actually integrates with) and only call it
        // an error when that says the cell is inverted.
        //
        // CORRECTED 2026-09-18: this block previously required
        // `cell_has_collapsed_edge()` — an EXACT bit-equality test on two of
        // the eight corners — before it would accept a positive-volume cell,
        // which silently narrowed the rule to cause (a) and contradicted the
        // policy stated immediately above. Cause (b) has no bit-identical
        // corner pair, so wall cells with genuinely positive volume were
        // escalated to Error. Found on CMC009 at reduce_factor 1: 32 cells of
        // 18,923,520, all in the j-max wall layer of 10 blade-surface blocks,
        // every one with `cell_volume_divergence > 0` (3.2e-16 .. 1e-14, the
        // expected size of a 0.5 um wall cell) and aspect ratios ~1.2e5. The
        // same mesh passes at reduce_factor 2 only because merging two wall
        // layers halves the aspect ratio and the corner Jacobian stays
        // positive — i.e. coarsening was masking a metric artifact, not a
        // mesh defect. The cells remain reported, at Warn.
        let signed = cell_field(b, cell_signed_volume);
        let mut neg_count = 0usize;
        let mut degen_count = 0usize;
        let mut degen_first: Option<(usize, usize, usize, Float)> = None;
        for (idx, &sv) in signed.iter().enumerate() {
            if sv > 0.0 {
                continue;
            }
            let (i, j, k) = cell_ijk(idx, nci, ncj);
            let vd = cell_volume_divergence(b, i, j, k);
            if vd > 0.0 {
                // Real positive volume by the operator the solver integrates
                // with — report, don't abort. Cause (a) or (b) above; we do
                // not care which, because neither is unusable geometry.
                degen_count += 1;
                if degen_first.is_none() {
                    degen_first = Some((i, j, k, vd));
                }
                continue;
            }
            neg_count += 1;
            if neg_count <= MAX_LISTED_NEGATIVE {
                violations.push(Violation {
                    check: "negative_volume",
                    severity: Severity::Error,
                    actual: sv,
                    threshold: 0.0,
                    location: Some(CellLocation {
                        block: bi,
                        i,
                        j,
                        k,
                        centroid: cell_centroid(b, i, j, k),
                    }),
                    message: format!(
                        "signed cell volume {sv:.3e} <= 0 and divergence-theorem \
                         volume {vd:.3e} (inverted cell)"
                    ),
                });
            }
        }
        if neg_count > MAX_LISTED_NEGATIVE {
            violations.push(Violation {
                check: "negative_volume",
                severity: Severity::Error,
                actual: neg_count as Float,
                threshold: 0.0,
                location: None,
                message: format!(
                    "block {bi}: {neg_count} cells with non-positive signed volume \
                     ({} more not listed individually)",
                    neg_count - MAX_LISTED_NEGATIVE
                ),
            });
        }
        if degen_count > 0 {
            let (i, j, k, vd) = degen_first.unwrap_or((0, 0, 0, 0.0));
            violations.push(Violation {
                check: "degenerate_cell",
                severity: Severity::Warn,
                actual: degen_count as Float,
                threshold: 0.0,
                location: Some(CellLocation {
                    block: bi,
                    i,
                    j,
                    k,
                    centroid: cell_centroid(b, i, j, k),
                }),
                message: format!(
                    "block {bi}: {degen_count} cell(s) on a COLLAPSED GRID LINE \
                     (coincident corner nodes). These are not inverted — the \
                     divergence-theorem volume is positive (e.g. {vd:.3e} at the \
                     first such cell) — so the discretization is well posed. But \
                     such cells are orders of magnitude smaller than their \
                     neighbours and will throttle an explicit local time step; \
                     merge them into a neighbour rather than advancing them as \
                     independent control volumes."
                ),
            });
        }

        // --- degenerate: min |volume| relative to the block median ---
        //
        // Scored on the divergence-theorem volume for the same reason as
        // above: the corner-anchored triple product reads exactly 0 on a
        // collapsed grid line, which would peg this ratio at 0 and fire a
        // spurious Error on a valid grid. A cell that is genuinely tiny
        // (rather than merely corner-degenerate) still trips this check —
        // it is a real stiffness hazard — but at Warn severity when the
        // cell is only degenerate, so the run is not blocked.
        let true_vol = cell_field(b, cell_volume_divergence);
        // Renamed 2026-09-18 from `all_degenerate_are_collapsed`. The old name
        // described the collapsed-edge test that used to gate the block above;
        // what the condition actually encodes — and all the site below needs —
        // is "some cells tripped the signed-volume test, and none of them is
        // genuinely inverted."
        let nothing_is_inverted = degen_count > 0 && neg_count == 0;
        let mut absvol: Vec<Float> = true_vol.iter().map(|v| v.abs()).collect();
        absvol.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        let median = median_sorted(&absvol);
        if median <= 0.0 {
            violations.push(Violation {
                check: "min_cell_volume",
                severity: Severity::Error,
                actual: 0.0,
                threshold: 1.0,
                location: None,
                message: format!(
                    "block {bi}: median cell volume is non-positive — block is degenerate"
                ),
            });
        } else {
            // argmin over the original (unsorted) true-volume magnitudes
            let mut vmin = Float::INFINITY;
            let mut vmin_idx = 0usize;
            for (idx, &sv) in true_vol.iter().enumerate() {
                let a = sv.abs();
                if a < vmin {
                    vmin = a;
                    vmin_idx = idx;
                }
            }
            let ratio = vmin / median;
            if ratio < t.min_cell_volume_ratio {
                let (i, j, k) = cell_ijk(vmin_idx, nci, ncj);
                // If nothing in this block is actually inverted, a merely
                // small cell is the known-degenerate case already reported
                // above — advisory, not fatal.
                //
                // CORRECTED 2026-09-18, same defect as the negative-volume
                // block above: this also required `cell_has_collapsed_edge()`,
                // which restricted the exemption to O-grid pinch lines and
                // escalated a warped-but-valid high-aspect-ratio wall cell to
                // Error. That conjunct was redundant as well as wrong —
                // `nothing_is_inverted` already establishes that no cell in
                // this block has non-positive divergence-theorem volume, which
                // is the only thing that makes a small cell unusable.
                let sev = if nothing_is_inverted {
                    Severity::Warn
                } else {
                    Severity::Error
                };
                violations.push(Violation {
                    check: "min_cell_volume",
                    severity: sev,
                    actual: ratio,
                    threshold: t.min_cell_volume_ratio,
                    location: Some(CellLocation {
                        block: bi,
                        i,
                        j,
                        k,
                        centroid: cell_centroid(b, i, j, k),
                    }),
                    message: format!(
                        "min cell volume / median = {ratio:.2e} < {:.0e} (near-degenerate)",
                        t.min_cell_volume_ratio
                    ),
                });
            }
        }

        // --- skewness (p99 + max), orthogonality ---
        let skew = cell_field(b, cell_skewness);
        let (skew_p99, skew_max, skew_max_idx) =
            stats_dropped(&skew, nci, ncj, nck, t.boundary_drop);
        if skew_p99 > t.skew_p99_deg {
            violations.push(Violation {
                check: "skewness",
                severity: Severity::Warn,
                actual: skew_p99,
                threshold: t.skew_p99_deg,
                location: None,
                message: format!(
                    "skewness p99 = {skew_p99:.1}° > {:.1}°",
                    t.skew_p99_deg
                ),
            });
        }
        if skew_max > t.skew_max_deg {
            let (i, j, k) = cell_ijk(skew_max_idx, nci, ncj);
            violations.push(Violation {
                check: "skewness",
                severity: Severity::Warn,
                actual: skew_max,
                threshold: t.skew_max_deg,
                location: Some(CellLocation {
                    block: bi,
                    i,
                    j,
                    k,
                    centroid: cell_centroid(b, i, j, k),
                }),
                message: format!("skewness {skew_max:.1}° > {:.1}°", t.skew_max_deg),
            });
        }
        // orthogonality: global min angle = 90 - global max skewness
        let global_skew_max =
            skew.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        let ortho_min = 90.0 - global_skew_max;
        if ortho_min < t.min_orthogonality_deg {
            violations.push(Violation {
                check: "orthogonality",
                severity: Severity::Warn,
                actual: ortho_min,
                threshold: t.min_orthogonality_deg,
                location: None,
                message: format!(
                    "min orthogonality angle {ortho_min:.1}° < {:.1}°",
                    t.min_orthogonality_deg
                ),
            });
        }

        // --- aspect ratio: interior vs wall first-cell ---
        let ar = cell_field(b, cell_aspect_ratio);
        // interior = strip one cell layer from each axis end (when possible)
        let interior_ok = nci > 2 && ncj > 2 && nck > 2;
        let mut interior_max = Float::NEG_INFINITY;
        let mut interior_idx = 0usize;
        let mut wall_max = Float::NEG_INFINITY;
        let mut wall_idx = 0usize;
        for k in 0..nck {
            for j in 0..ncj {
                for i in 0..nci {
                    let idx = (k * ncj + j) * nci + i;
                    let v = ar[idx];
                    // A cell on a collapsed grid line has no aspect ratio
                    // (`cell_aspect_ratio` returns INFINITY for it). Skip it
                    // here so ONE condition is not reported twice under two
                    // names: the degenerate-cell check above already names
                    // those cells, and letting them set the max here turned a
                    // legitimate O-grid construction into a spurious
                    // aspect-ratio violation that trained readers to dismiss
                    // the whole check.
                    if !v.is_finite() {
                        continue;
                    }
                    if v > wall_max {
                        wall_max = v;
                        wall_idx = idx;
                    }
                    let is_interior = !interior_ok
                        || (i > 0
                            && i < nci - 1
                            && j > 0
                            && j < ncj - 1
                            && k > 0
                            && k < nck - 1);
                    if is_interior && v > interior_max {
                        interior_max = v;
                        interior_idx = idx;
                    }
                }
            }
        }
        if interior_max > t.max_ar_interior {
            let (i, j, k) = cell_ijk(interior_idx, nci, ncj);
            violations.push(Violation {
                check: "aspect_ratio",
                severity: Severity::Warn,
                actual: interior_max,
                threshold: t.max_ar_interior,
                location: Some(CellLocation {
                    block: bi,
                    i,
                    j,
                    k,
                    centroid: cell_centroid(b, i, j, k),
                }),
                message: format!(
                    "interior aspect ratio {interior_max:.0} > {:.0}",
                    t.max_ar_interior
                ),
            });
        }
        if wall_max > t.max_ar_wall {
            let (i, j, k) = cell_ijk(wall_idx, nci, ncj);
            violations.push(Violation {
                check: "aspect_ratio",
                severity: Severity::Warn,
                actual: wall_max,
                threshold: t.max_ar_wall,
                location: Some(CellLocation {
                    block: bi,
                    i,
                    j,
                    k,
                    centroid: cell_centroid(b, i, j, k),
                }),
                message: format!(
                    "wall aspect ratio {wall_max:.0} > {:.0}",
                    t.max_ar_wall
                ),
            });
        }
    }

    MeshQualityReport {
        preset: preset_name.to_string(),
        handedness,
        violations,
    }
}

// =============================================================================
// Element-type inventory — ADS / Code Leo `ELEMTYPE` + `ADVOLAREA` analogue
// =============================================================================
//
// A structured hex cell that sits on an O-grid pinch line collapses to a
// lower element: the NASA Rotor 35 blade-tip seam has 960 cells with two
// coincident node-pairs (a wedge). Code Leo's load log reports exactly that
// — `1509440 HEX + 960 PRISM` — via its `ELEMTYPE` (type census) and
// `ADVOLAREA` (per-element volume) passes. This block reproduces that census
// so a glennht-gpu load echoes the reference solver's own printout. It is a
// DIAGNOSTIC: it changes no solver state (we already integrate the collapsed
// hex correctly — the divergence-theorem volume is exact for a wedge).

/// The standard element a structured hex cell collapses to, by DISTINCT
/// corner-node count. Mirrors Code Leo's `ELEMTYPE` categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 8 distinct nodes — a normal hexahedron.
    Hex,
    /// 6 — one edge collapsed (a wedge; two coincident node-pairs).
    Prism,
    /// 5 — one face collapsed toward a point.
    Pyramid,
    /// 4 — a tetrahedron.
    Tet,
    /// 7, or `< 4` — a non-standard partial collapse.
    Other,
}

impl ElementType {
    /// Classify by the number of distinct corner nodes.
    pub fn from_distinct_nodes(n: usize) -> ElementType {
        match n {
            8 => ElementType::Hex,
            6 => ElementType::Prism,
            5 => ElementType::Pyramid,
            4 => ElementType::Tet,
            _ => ElementType::Other,
        }
    }
    /// Upper-case type name (`"HEX"`, `"PRISM"`, …).
    pub fn name(self) -> &'static str {
        match self {
            ElementType::Hex => "HEX",
            ElementType::Prism => "PRISM",
            ElementType::Pyramid => "PYRAMID",
            ElementType::Tet => "TET",
            ElementType::Other => "OTHER",
        }
    }
}

/// Number of DISTINCT corner nodes of cell `(i,j,k)` (exact equality — the
/// same bit-identical test [`cell_has_collapsed_edge`] uses; O-grid pinch
/// nodes are bit-identical, so a tolerance would sweep in merely-close nodes).
pub fn cell_distinct_node_count(b: &Block, i: usize, j: usize, k: usize) -> usize {
    let mut n = [[0.0 as Float; 3]; 8];
    let mut m = 0;
    for &dk in &[0usize, 1] {
        for &dj in &[0usize, 1] {
            for &di in &[0usize, 1] {
                let (x, y, z) = b.xyz(i + di, j + dj, k + dk);
                n[m] = [x, y, z];
                m += 1;
            }
        }
    }
    let mut count = 0usize;
    for a in 0..8 {
        let mut dup = false;
        for c in 0..a {
            if n[a] == n[c] {
                dup = true;
                break;
            }
        }
        if !dup {
            count += 1;
        }
    }
    count
}

/// Per-block element-type tally + minimum divergence-theorem cell volume.
#[derive(Debug, Clone)]
pub struct BlockElementSummary {
    pub n_hex: usize,
    pub n_prism: usize,
    pub n_pyramid: usize,
    pub n_tet: usize,
    pub n_other: usize,
    /// Minimum divergence-theorem cell volume in the block.
    pub min_volume: Float,
    /// The `(i,j,k)` of the minimum-volume cell.
    pub min_volume_cell: (usize, usize, usize),
}

/// Whole-mesh element census — the analogue of Code Leo's `ELEMTYPE`
/// (type counts) + `ADVOLAREA` (per-element volume) load passes.
#[derive(Debug, Clone)]
pub struct ElementInventory {
    pub per_block: Vec<BlockElementSummary>,
    pub n_hex: usize,
    pub n_prism: usize,
    pub n_pyramid: usize,
    pub n_tet: usize,
    pub n_other: usize,
    pub total: usize,
    pub global_min_volume: Float,
    /// `(block, i, j, k)` of the global-minimum-volume cell.
    pub global_min_cell: (usize, usize, usize, usize),
}

/// Classify every cell of every block by collapsed-node topology and tally
/// per-block volumes. A cheap second load-time pass, independent of
/// [`run_all`].
pub fn element_type_inventory(blocks: &[Block]) -> ElementInventory {
    let mut per_block = Vec::with_capacity(blocks.len());
    let (mut t_hex, mut t_prism, mut t_pyr, mut t_tet, mut t_other) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut g_min = Float::INFINITY;
    let mut g_cell = (0usize, 0usize, 0usize, 0usize);

    for (bi, b) in blocks.iter().enumerate() {
        let (nci, ncj, nck) = cell_dims(b);
        let mut s = BlockElementSummary {
            n_hex: 0,
            n_prism: 0,
            n_pyramid: 0,
            n_tet: 0,
            n_other: 0,
            min_volume: Float::INFINITY,
            min_volume_cell: (0, 0, 0),
        };
        if nci == 0 || ncj == 0 || nck == 0 {
            per_block.push(s);
            continue;
        }
        for k in 0..nck {
            for j in 0..ncj {
                for i in 0..nci {
                    match ElementType::from_distinct_nodes(
                        cell_distinct_node_count(b, i, j, k),
                    ) {
                        ElementType::Hex => s.n_hex += 1,
                        ElementType::Prism => s.n_prism += 1,
                        ElementType::Pyramid => s.n_pyramid += 1,
                        ElementType::Tet => s.n_tet += 1,
                        ElementType::Other => s.n_other += 1,
                    }
                    let v = cell_volume_divergence(b, i, j, k);
                    if v < s.min_volume {
                        s.min_volume = v;
                        s.min_volume_cell = (i, j, k);
                    }
                }
            }
        }
        if s.min_volume < g_min {
            g_min = s.min_volume;
            g_cell =
                (bi, s.min_volume_cell.0, s.min_volume_cell.1, s.min_volume_cell.2);
        }
        t_hex += s.n_hex;
        t_prism += s.n_prism;
        t_pyr += s.n_pyramid;
        t_tet += s.n_tet;
        t_other += s.n_other;
        per_block.push(s);
    }

    ElementInventory {
        per_block,
        n_hex: t_hex,
        n_prism: t_prism,
        n_pyramid: t_pyr,
        n_tet: t_tet,
        n_other: t_other,
        total: t_hex + t_prism + t_pyr + t_tet + t_other,
        global_min_volume: if g_min.is_finite() { g_min } else { 0.0 },
        global_min_cell: g_cell,
    }
}

impl ElementInventory {
    /// Code Leo-style census string (`ELEMTYPE` counts + `ADVOLAREA` min-vol),
    /// so a glennht-gpu load echoes the reference solver's own printout.
    pub fn format_ads_style(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Element census (ELEMTYPE analogue) — {} elements:\n",
            self.total
        ));
        s.push_str(&format!("  {:>10} ELEMENTS OF HEX     TYPE\n", self.n_hex));
        s.push_str(&format!("  {:>10} ELEMENTS OF PRISM   TYPE\n", self.n_prism));
        s.push_str(&format!(
            "  {:>10} ELEMENTS OF PYRAMID TYPE\n",
            self.n_pyramid
        ));
        s.push_str(&format!("  {:>10} ELEMENTS OF TET     TYPE\n", self.n_tet));
        if self.n_other > 0 {
            s.push_str(&format!(
                "  {:>10} ELEMENTS OF OTHER   TYPE (non-standard partial collapse)\n",
                self.n_other
            ));
        }
        let (gb, gi, gj, gk) = self.global_min_cell;
        s.push_str(&format!(
            "Volume (ADVOLAREA analogue): global MIN VOLUME {:.6e} at block {} cell ({},{},{})\n",
            self.global_min_volume, gb, gi, gj, gk
        ));
        for (bi, blk) in self.per_block.iter().enumerate() {
            let (i, j, k) = blk.min_volume_cell;
            let other = if blk.n_other > 0 {
                format!(", {} other", blk.n_other)
            } else {
                String::new()
            };
            s.push_str(&format!(
                "  block {:>2}: MIN VOLUME {:.4e} at ({},{},{})  [{} hex, {} prism, {} pyr, {} tet{}]\n",
                bi, blk.min_volume, i, j, k,
                blk.n_hex, blk.n_prism, blk.n_pyramid, blk.n_tet, other
            ));
        }
        s
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed unit-cube block of `n×n×n` nodes (n-1 cells/axis).
    fn unit_cube(n: usize) -> Block {
        let mut x = vec![0.0 as Float; n * n * n];
        let mut y = x.clone();
        let mut z = x.clone();
        let h = 1.0 / (n - 1) as Float;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = (k * n + j) * n + i;
                    x[idx] = i as Float * h;
                    y[idx] = j as Float * h;
                    z[idx] = k as Float * h;
                }
            }
        }
        Block::new(n, n, n, x, y, z)
    }

    /// A unit cube with one grid line COLLAPSED: the `i=0` and `i=1`
    /// nodes are made bit-identical along the `j=0` plane, reproducing an
    /// O-grid pinched onto a camber line (the NASA Rotor 35 tip block).
    /// The affected cells keep real positive volume but their
    /// corner-anchored triple product is exactly zero.
    fn collapsed_line_block(n: usize) -> Block {
        let b = unit_cube(n);
        let (mut x, mut y, mut z) = (b.x.clone(), b.y.clone(), b.z.clone());
        for k in 0..n {
            let src = (k * n) * n; // (i=0, j=0, k)
            let dst = src + 1; // (i=1, j=0, k)
            x[dst] = x[src];
            y[dst] = y[src];
            z[dst] = z[src];
        }
        Block::new(n, n, n, x, y, z)
    }

    /// A collapsed cell has NO aspect ratio, and must not be given a finite
    /// one — nor allowed to set the block's reported aspect-ratio maximum.
    ///
    /// Regression for a real misreport. `cell_aspect_ratio` used to floor its
    /// denominator at `1e-30`, so a cell with a legitimate 13 µm edge and a
    /// collapsed one returned `1.3e25`. On rotor35 (0.5 µm first cell) that
    /// surfaced as `wall aspect ratio 13045882770553017259261952 > 100000`:
    /// a precise-looking number that is pure arithmetic, twenty orders of
    /// magnitude past anything the geometry can produce, reported as an
    /// ASPECT-RATIO fault when the real and separately-reported condition was
    /// a collapsed grid line. One condition, named twice, the second time
    /// wrongly — which taught readers to dismiss the check entirely.
    ///
    /// The old floor made this test impossible to write as an equality, so
    /// note what the assertions below actually pin: the value is NOT finite
    /// (the `1e-30` floor always produced a finite one), and the violation
    /// list carries no `aspect_ratio` entry sourced from the pinched cell.
    #[test]
    fn a_collapsed_cell_has_no_aspect_ratio_and_does_not_set_the_maximum() {
        let b = collapsed_line_block(5);

        // The pinched cell: undefined, not a large number.
        let ar = cell_aspect_ratio(&b, 0, 0, 0);
        assert!(
            !ar.is_finite(),
            "a zero-length edge has no aspect ratio; got the finite value {ar:e}, \
             which is the 1e-30-floor artefact this test exists to refuse"
        );
        assert!(cell_has_collapsed_edge(&b, 0, 0, 0));

        // A healthy cell in the same block still gets a real, finite ratio —
        // so the guard above cannot be passing by disabling the metric.
        let healthy = cell_aspect_ratio(&b, 2, 2, 2);
        assert!(
            healthy.is_finite() && healthy >= 1.0,
            "healthy cell must still report a finite ratio >= 1, got {healthy:e}"
        );

        // And the collapsed cell must not become the block's reported max.
        let report = run_all(&[b], &Thresholds::STANDARD, "STANDARD");
        for v in report.violations.iter().filter(|v| v.check == "aspect_ratio") {
            assert!(
                v.actual.is_finite(),
                "an aspect_ratio violation was raised with a non-finite value \
                 ({:e}) — the collapsed cell set the maximum",
                v.actual
            );
        }
    }

    #[test]
    fn collapsed_line_is_degenerate_not_inverted() {
        let b = collapsed_line_block(5);
        // The pinched cell: triple product is exactly 0 ...
        assert_eq!(cell_signed_volume(&b, 0, 0, 0), 0.0);
        // ... but it has REAL positive volume and a detectable collapse.
        assert!(cell_volume_divergence(&b, 0, 0, 0) > 0.0);
        assert!(cell_has_collapsed_edge(&b, 0, 0, 0));
        // A healthy cell elsewhere is untouched and not flagged.
        assert!(cell_signed_volume(&b, 2, 2, 2) > 0.0);
        assert!(!cell_has_collapsed_edge(&b, 2, 2, 2));

        let report = run_all(&[b], &Thresholds::STANDARD, "STANDARD");
        // The collapse must NOT be reported as an inverted cell ...
        assert!(
            !report
                .violations
                .iter()
                .any(|v| v.check == "negative_volume"),
            "a collapsed grid line must not be classified as an inverted cell"
        );
        // ... it must be reported as a degenerate cell, at Warn severity ...
        let degen: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.check == "degenerate_cell")
            .collect();
        assert_eq!(degen.len(), 1, "expected one degenerate_cell violation");
        assert!(matches!(degen[0].severity, Severity::Warn));
        // ... and nothing in the block may be fatal.
        assert!(
            !report
                .violations
                .iter()
                .any(|v| matches!(v.severity, Severity::Error)),
            "a valid grid with a collapsed line must not produce a fatal verdict"
        );
    }

    #[test]
    fn genuinely_inverted_cell_is_still_fatal() {
        // Regression guard: the degenerate carve-out must not become a
        // blanket amnesty. Mirror one node so a cell turns inside out.
        let b = unit_cube(5);
        let (mut x, y, z) = (b.x.clone(), b.y.clone(), b.z.clone());
        x[0] = 10.0; // drag (0,0,0) far past the opposite face
        let b = Block::new(5, 5, 5, x, y, z);
        assert!(cell_signed_volume(&b, 0, 0, 0) < 0.0);
        assert!(!cell_has_collapsed_edge(&b, 0, 0, 0));
        let report = run_all(&[b], &Thresholds::STANDARD, "STANDARD");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.check == "negative_volume"
                    && matches!(v.severity, Severity::Error)),
            "a genuinely inverted cell must remain a fatal negative_volume error"
        );
    }

    #[test]
    fn unit_cube_is_right_handed_and_clean() {
        let b = unit_cube(5);
        assert_eq!(block_handedness(&b), Handedness::RightHanded);
        // Perfect cube: skewness ~0, aspect ratio ~1, no violations.
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    assert!(cell_signed_volume(&b, i, j, k) > 0.0);
                    assert!((cell_aspect_ratio(&b, i, j, k) - 1.0).abs() < 1e-4);
                    assert!(cell_skewness(&b, i, j, k).abs() < 1e-3);
                }
            }
        }
        let report = run_all(&[b], &Thresholds::STANDARD, "STANDARD");
        assert_eq!(report.handedness, vec![Handedness::RightHanded]);
        assert!(report.passes());
        assert_eq!(report.n_error(), 0);
        assert_eq!(report.n_warn(), 0);
    }

    #[test]
    fn left_handed_block_is_detected_and_fixed() {
        // Mirror the cube along i → left-handed.
        let cube = unit_cube(5);
        let dims = (cube.imax, cube.jmax, cube.kmax);
        let (mut x, mut y, mut z) =
            (cube.x.clone(), cube.y.clone(), cube.z.clone());
        crate::block_analysis::flip_block_axis(&mut x, &mut y, &mut z, dims, 0);
        let lh = Block::new(cube.imax, cube.jmax, cube.kmax, x, y, z);
        assert_eq!(block_handedness(&lh), Handedness::LeftHanded);

        let (fixed, flipped) = make_right_handed(&lh);
        assert_eq!(flipped, Some(0));
        assert_eq!(block_handedness(&fixed), Handedness::RightHanded);
        // Fixing a left-handed copy of the cube recovers the original cube.
        let report = run_all(&[fixed], &Thresholds::STANDARD, "STANDARD");
        assert!(report.passes());

        // make_right_handed on an already-right-handed block is a no-op.
        let (same, flip2) = make_right_handed(&cube);
        assert_eq!(flip2, None);
        assert_eq!(block_handedness(&same), Handedness::RightHanded);
    }

    #[test]
    fn collapsed_cell_is_flagged_with_location() {
        // Take a clean cube and collapse one cell by snapping a node.
        let mut b = unit_cube(5);
        // Move node (2,2,2) onto (1,2,2): cell (1,1,1) loses i-extent.
        let src = b.idx(1, 2, 2);
        let dst = b.idx(2, 2, 2);
        b.x[dst] = b.x[src];
        b.y[dst] = b.y[src];
        b.z[dst] = b.z[src];
        let report = run_all(&[b], &Thresholds::STANDARD, "STANDARD");
        // BEHAVIOUR CHANGE (degenerate-cell classification): snapping a
        // node produces coincident corners — a COLLAPSED cell, not an
        // INVERTED one. Its divergence-theorem volume is still positive,
        // so the finite-volume discretization is well posed and this is
        // no longer fatal. It must still be flagged, and still carry a
        // location so the user can find it; that is what this test guards.
        // Genuine inversion remains fatal — see
        // `genuinely_inverted_cell_is_still_fatal`.
        assert!(
            report.violations.iter().any(|v| v.check == "degenerate_cell"
                && v.severity == Severity::Warn
                && v.location.is_some()),
            "a collapsed cell must be flagged (with a location) as degenerate"
        );
        assert_eq!(
            report.n_error(),
            0,
            "a collapsed-but-positive-volume cell must not be fatal"
        );
    }

    #[test]
    fn percentile_linear_interpolation() {
        let v = [0.0 as Float, 1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&v, 0.0) - 0.0).abs() < 1e-6);
        assert!((percentile(&v, 1.0) - 4.0).abs() < 1e-6);
        assert!((percentile(&v, 0.5) - 2.0).abs() < 1e-6);
    }
}
