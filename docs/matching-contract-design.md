# Face-matching contract: target design (not yet built)

Status: design note, 2026-09-12. Recorded so the analysis survives; the API
described here is **not** implemented. What *was* landed alongside this note
is the minimal set of correctness fixes it motivated (see "What landed
instead" at the end).

## Why this note exists

Two defects in the current matching API cannot be fixed inside its shape:

1. **An ambiguous patch enters the exterior-face list silently and becomes a
   wall.** Every discovery function returns `(matches, outer_faces)`. A face
   that *almost* matched — nodes within a few tolerances of a partner, or two
   candidate partners that both fit — is not a match, so it goes to
   `outer_faces`. Downstream, an outer face is a wall boundary. A periodic
   seam that missed by 2× tolerance therefore turns into a solid wall with no
   signal to the caller beyond a face count that is off by two.

2. **A tuple cannot express partial success or ambiguity.** "Enough nearby
   nodes were found" and "there is exactly one admissible correspondence" are
   different claims, and only the second is what a solver needs. The tuple has
   no place for "this region has two competing donors", "this pair fails at
   node (4,2,2) by 3.1× tolerance", or "the declared precision cannot
   distinguish these two surfaces".

The design below makes the library answer the question it is actually asked:

> Under this declared geometric accuracy and periodic transformation, is
> there a unique, complete, admissible correspondence between these patches?

## Why the first fix attempt was withdrawn

The first attempt replaced `translational_periodicity`'s
`ADAPTIVE_TOL_FLOOR = 1e-4` with
`max(0.03 * local_spacing, 32 * f32_quantum * coordinate_extent)`. Review
found three problems, each general enough to shape the design:

* **A spacing fraction can exceed the physical feature it must not bridge.**
  On a production compressor deck the tip-clearance gap is 1.25e-4 to
  1.90e-4 m; at 5 mm spacing, `0.03 h = 150 µm` > a 125 µm gap. A blade tip
  would be declared coincident with the casing. Spacing measures *resolution*
  and *ambiguity*; it says nothing about how far apart two copies of one node
  may legitimately be.
* **Rotation does not magnify coordinate error.** `||R dx|| = ||dx||` for an
  exact rotation. The real extra error comes from *evaluating* the transform
  and from *uncertainty in its parameters* (angle uncertainty contributes
  `r |dθ|`), and independent worst cases **add** — they do not combine by
  `max`.
* **"Rotation preserves radius, so tip and casing cannot be confused" is
  invalid.** Matching accepts a *nonzero* distance, so radius separation
  excludes a pair only when it exceeds the tolerance plus margins, and wrong
  candidates can share a radius anyway.

There is no universal constant that distinguishes an intended seam mismatch
from a physical gap of the same size. That distinction needs accuracy
assumptions, connectivity metadata, or geometric separation constraints —
all of which have to come from the caller.

## (a) One tolerance policy, shared by every stage

For a proposed node correspondence `i <-> j` with
`d_ij = ||T(x_i) - x_j||_2`:

```
tau_ij = delta_geom + delta_input_i + delta_input_j + delta_transform_i + delta_arithmetic_ij
```

All are distances in the coordinates' current units:

| term | meaning | source |
|---|---|---|
| `delta_geom` | explicitly permitted mismatch between independently generated representations of one intended interface | caller declaration |
| `delta_input` | coordinate uncertainty from storage/export precision, per node from its own magnitude | `InputPrecision` |
| `delta_transform` | uncertainty in the declared angle, axis, origin or translation, per source node | `TransformUncertainty` |
| `delta_arithmetic` | round-off evaluating `T` and the comparison in the crate's float type | `K · eps · (|x_i| + |T x_i| + |x_j| + |offset|)` |

It is a worst-case **additive** budget, conditional on the input assumptions
holding. The recommended default has **no spacing term**.

`MatchPolicy` (sketch):

```rust
pub enum InputPrecision {
    Exact,
    F32 { scale_since_storage: Float },     // quantised in the stored unit,
    F64 { scale_since_storage: Float },     // then scaled (e.g. 0.0254 in→m)
    Decimal { significant_digits: u32, scale_since_storage: Float },
    Absolute(Float),
    Composite(Vec<InputPrecision>),         // stages add
}
pub enum Transform { Identity, Translation([Float;3]), Rotation { axis, angle_rad, origin } }
pub struct TransformUncertainty { angle_rad, axis_tilt_rad, origin, translation }
pub struct MatchPolicy {
    geometric_tolerance: Float,
    input_precision: InputPrecision,
    transform_uncertainty: TransformUncertainty,
    arithmetic_epsilon: Float,
    min_surface_separation: Option<Float>,  // caller-asserted; never overridden
}
```

Requirements the policy must meet:

* Carry the source precision or a caller-specified bound. **Never silently
  assume every coordinate originated as `f32`** — reading `f32` data into
  `f64` retains its quantisation; genuinely `f64` or ASCII input does not
  have `f32` uncertainty. `scale_since_storage` exists because the
  quantisation happened in the *stored* unit: an f32 coordinate of 40 in
  converted to metres carries `0.0254 · ulp(40)/2`, not `ulp(1.016)/2` (a
  factor ~40 difference).
* `policy.scaled(k)` scales every distance allowance; angular terms are
  unchanged. Then `tolerance(k·x, k·y) == k · tolerance(x, y)`.
* Reject invalid tolerances (negative, NaN, zero separation) and non-finite
  geometry up front (`InvalidInput`).
* A collapsed/degenerate patch is *insufficient information*, never an
  invented spacing (the old `median_inplane_spacing` returned `1.0` for a
  face with fewer than two points — fixed in the landed code to return
  `None`).
* Every default uncertainty estimate is an **assumption reported with the
  result** (`policy.describe()` in every report), not a proven bound.

Transform-uncertainty bounds, for the record:
angle `r_perp · δθ`; axis tilt `4 sin(θ/2) · δψ · ||p − o||` (from
`Q R Qᵀ − R = E R + R Eᵀ + O(E²)`, `E` skew); origin `2 sin(θ/2) · δo`
(from `(I − R)(o' − o)`); translation `δt`. Arithmetic: `8 · eps ·` (sum of
operand magnitudes) covers a 3-FMA rotation, the subtraction and the norm.

### Where the policy applies

| stage | required behaviour |
|---|---|
| candidate generation | expand spatial bounds conservatively by the applicable uncertainty bounds (`query_radius` = upper bound of `tau` over all targets) |
| pruning | reject only when a **lower bound** on separation exceeds the permitted discrepancy |
| orientation | enumerate compatible structured mappings; corners **propose**, never certify |
| final validation | check **every** corresponding node against the same policy, on the **original** grid |
| conflict resolution | accept only correspondence consistent with patch ownership and uniqueness |

Spatial bins stay useful, but a query must visit **every bin the tolerance
ball intersects** and then evaluate actual distances. Bin equality must not
define coincidence (the old `round(x / tol)` keys rejected two points 0.9 tol
apart that straddled a bin edge — fixed in the landed code).

If tolerances overlap competing correspondences, **do not take whichever
candidate is encountered first**. Structured constraints may resolve it;
otherwise report ambiguity. A caller-specified `min_surface_separation` must
never be overridden by an automatically enlarged tolerance: any node pair
whose evaluated `tau` reaches it is *inadequate precision*, not a match.

## (b) Minimal correct face validation

For conformal structured connectivity:

1. identify two logically rectangular patch ranges (one constant index each);
2. check compatible dimensions under index reversal / axis exchange;
3. construct the structured index mapping (one of the 8 permutations);
4. check **every node**, interior included, under that mapping;
5. bijective coverage follows from (2)+(3); adjoining-cell sides/orientation
   follow from the mapping;
6. reject conflicting ownership of interface face *elements* (cells); shared
   patch-**edge** nodes are legitimate.

For partial-face matches, validate the **entire claimed rectangular
subpatch**. A percentage of matching nodes is candidate evidence, not proof.

Different node counts or genuinely non-conformal interfaces need a
**distinct representation** and interpolation/conservation treatment — never
label them conformal because their corners agree (cf. CGNS's separate
structured one-to-one vs general connectivity).

Reduced meshes accelerate **discovery only**; every proposed match is
validated on the original full-resolution nodes before it is returned.

This part is implemented: `src/correspondence.rs` (`certify_correspondence`,
`certify_permutation`, `certify_face_match`) is what the landed fixes use.

## (c) Surfacing uncertainty: the report API

Additive, preserving the existing tuple functions:

```rust
pub struct MatchReport<M> {
    pub matches: Vec<M>,             // certified node-for-node on the full grid
    pub unmatched: Vec<FaceRecord>,  // no candidate at all (ordinary exterior faces)
    pub unresolved: Vec<MatchIssue>, // evidence found but not certifiable — NEVER in `unmatched`
    pub assumptions: String,         // policy.describe() + transform
}

pub struct MatchIssue {
    pub reason: IssueReason,         // Ambiguity | InadequatePrecision | UnsupportedDegeneracy
                                     // | IncompatibleCorrespondence | OwnershipConflict
    pub patch: Patch,                // block + inclusive index range
    pub competing: Vec<Patch>,       // competing donors / proposed partner
    pub orientations: Vec<u8>,       // mappings that fit (ambiguity) or the best tried
    pub transform: String,
    pub tolerance_assumptions: String,
    pub worst: Option<NodeDiscrepancy>, // largest distance/tolerance and its indices
    pub detail: String,
}

pub fn analyze_connectivity(blocks: &[Block], policy: &MatchPolicy)
    -> Result<MatchReport<FaceMatch>, InvalidInput>;
pub fn analyze_periodicity(blocks, faces: &[FaceRecord], transform: &Transform, policy)
    -> Result<MatchReport<FaceMatch>, InvalidInput>;
pub fn validate_proposed_matches(blocks, proposed: &[FaceMatch], outer: &[FaceRecord], transform, policy)
    -> Result<MatchReport<FaceMatch>, InvalidInput>;   // for legacy discovery output

impl<M> MatchReport<M> {
    pub fn into_strict(self) -> Result<(Vec<M>, Vec<FaceRecord>), UnresolvedMatches>;
}
impl MatchReport<FaceMatch> {
    pub fn require_matched(&self, required: &[FaceRecord]) -> Result<(), MissingMatches>;
}
```

Semantics that matter:

* **An ambiguous patch never enters `unmatched`.** Cells claimed by a demoted
  (conflicting) candidate are marked unresolved and excluded from the
  rectangular decomposition that produces `unmatched`.
* `into_strict` refuses any report with `unresolved` entries.
* `require_matched` is for faces that *must* be periodic or *must* be
  interfaces: a required face is satisfied only when all its cells are
  covered by certified matches. Unmatched ordinary exterior faces are not
  automatically errors.
* The existing tuple APIs stay, explicitly marked **legacy**, because a tuple
  cannot honestly communicate partial success and ambiguity.

### Discovery algorithm (as prototyped)

* Put every node of every candidate face in a spatial hash whose bin is at
  least the largest `query_radius` (so a query touches ≤ 27 bins) and at
  least `max|coord| · 2⁻³⁰` (so `coord / bin` fits `i64`).
* For each source node, visit the bins the ball around `T(x_i)` intersects,
  compare actual `d_ij` against actual `tau_ij`, and tally hits per
  (source face, target face).
* Per face pair, the hits' bounding rectangle on each side is the proposed
  sub-patch (edge/point contact — extent 1 — is skipped as legitimate). The
  sub-patches are certified node-for-node; failure becomes a `MatchIssue`
  with the worst node.
* Ownership: per face, a cell grid; every certified candidate claims its
  cells; any cell with two claimants demotes **all** claimants to
  `Ambiguity` issues (order-independent — never "first wins").
* **Near-miss strips** (the one use of spacing): for each certified
  candidate, the row of nodes just outside it on each side is checked; if
  those nodes lie closer to their would-be partners than half the local
  spacing yet outside tolerance, the interface evidently continues beyond
  what the tolerance admits, and that strip is reported unresolved instead of
  becoming a wall.
* No reduced mesh anywhere; discovery is `O(boundary nodes)`.

The tip-gap scenario under this design: with `delta_geom` below the gap the
three faces (blade tip, gap block bottom/top, casing) pair correctly; with
`delta_geom` above the gap the extra pairs certify too, every affected cell
gets two claimants, all four candidates are reported as `Ambiguity`, and
`into_strict` refuses — nothing is silently wrong, and the report names the
competing donors. With `min_surface_separation` asserted below the
tolerance, the same pairs are reported as `InadequatePrecision` instead.

## Migration the consuming project (not-cfd) would need

Today not-cfd calls `connectivity_fast(blocks)`,
`rotated_periodicity(blocks, matches, outer, angle_deg, 'x', false)`, and the
verifiers with `verify_tol = 5.08e-8` m.

* **Now available (landed):** `connectivity_fast_with_tol(blocks, tol)`,
  `rotated_periodicity_with_tol(…, tol)`,
  `translational_periodicity_with_tols(…, &TranslationalTolerances)`. The
  no-tolerance entry points are unchanged in behaviour apart from the
  correctness fixes. `full_face_match{,_transformed}` now take the two blocks
  (every node is checked); the corner-only proposal is `corner_match{,_transformed}`.
* **Under the report design:** `verify_tol = 5.08e-8` becomes
  `MatchPolicy::from_f32_storage(0.0254).with_geometric_tolerance(…)` — the
  f32-in-inches quantisation is then modelled per node instead of folded
  into one constant, and the geometric allowance is stated on its own. The
  load pipeline would call `analyze_connectivity`, pass `report.unmatched`
  to `analyze_periodicity`, call `require_matched` on the faces the YAML
  declares periodic, and `into_strict` before building the flat mesh.
  `face_matches_to_dict` must not be applied to report output: its
  first/last-point diagonal derivation scales `MatchPoint`s by the mesh GCD,
  and report matches carry full-resolution points.

## What landed instead (2026-09-12)

Scoped down by the owner to targeted fixes for a shared library:

1. `full_face_match{,_transformed}` certify **every node** under exactly one
   structured mapping (`src/correspondence.rs`); corner agreement only
   proposes (`corner_match{,_transformed}`). Partial matches from
   `get_face_intersection` certify the whole claimed sub-patch on both sides.
2. All "shared node" tests use a proximity grid that visits every bin the
   tolerance ball intersects and compares actual distances
   (`geometry::PointGrid3/2`); rounded-bin keys are gone.
3. `connectivity_fast`, `rotated_periodicity`, `rotational_periodicity` and
   `translational_periodicity` re-certify reduced-grid proposals on the
   full-resolution nodes; failures are demoted to outer faces **with a
   stderr warning** — the tuple's only channel, which is exactly the
   limitation this note records.
4. Every hardcoded tolerance is a parameter with the old value as default.
5. `median_inplane_spacing` returns `None` instead of `1.0`.
