# Changelog

## 0.1.16

### Fixed

- `full_face_match` / `full_face_match_transformed` certified a face match
  from its **four corners only**, despite the name. They now take the two
  parent blocks and certify **every node** — interior nodes included — under
  exactly one of the eight structured index mappings, with the dimensions
  required to agree under that mapping (so coverage is bijective). Corner
  agreement is a proposal, available as `corner_match` /
  `corner_match_transformed`, and is used only to reject early and order the
  search. The certification lives in the new `correspondence` module
  (`certify_correspondence`, `certify_permutation`, `certify_face_match`,
  `Patch`).

  Partial-face matches from `get_face_intersection` (connectivity Phases 2–3,
  rotational Phase 2/3 split matches, translational Phases 2–3) now certify
  the **entire claimed rectangular sub-patch** on both sides the same way. A
  non-conformal overlap whose corners and edges agree but whose interior
  nodes do not is no longer reported as an interface.

- "Shared node" tests compared **rounded coordinate bins**
  (`round(x / tol)`), so two nodes closer than `tol` that straddled a bin
  boundary were wrongly counted as distinct — in `Face::touches_by_nodes`,
  `Face::shared_point_fraction`, and `translational_periodicity`'s
  orthogonal pre-check and Phase-3 footprint/3-D intersection. All of them
  now use a proximity grid (`geometry::PointGrid3` / `PointGrid2`) that
  visits every bin the tolerance ball intersects and compares actual
  Euclidean distances. Bin equality no longer defines coincidence.

- `connectivity_fast`, `rotated_periodicity` (with `reduce_mesh`),
  `rotational_periodicity` and `translational_periodicity` discovered matches
  on a GCD-reduced grid and scaled the indices back **without re-checking
  them on the original nodes**. Every reduced-grid proposal is now
  re-certified node-for-node at full resolution (under the recorded
  orientation where one was recorded; under either rotation direction, or
  the pair's own centroid shift, for periodic pairs). A proposal that fails
  is demoted to outer faces with a warning on stderr — the tuple return has
  no other channel; see `docs/matching-contract-design.md` for the report
  API that would carry it properly.

- `translational_periodicity`'s `median_inplane_spacing` returned a
  fabricated `1.0` for a face with fewer than two points. It now returns
  `None`, and a pair for which no tolerance can be derived is skipped
  instead of being matched at an invented tolerance.

### Added

- Tolerances that were hardcoded are now parameters, with the old values as
  defaults so existing callers are unchanged:
  `connectivity_fast_with_tol(blocks, tol)`;
  `rotated_periodicity_with_tol(…, tol)` and
  `rotational_periodicity_with_tol(…, tol)` (default
  `rotational_periodicity::DEFAULT_MATCH_TOL = 1e-4`);
  `translational_periodicity_with_tols(…, &TranslationalTolerances)` (fields
  `bounding_face_tol = 1e-6`, `full_face_tol = 1e-6`,
  `adaptive_floor = 1e-4`, `adaptive_spacing_fraction = 0.03`);
  `get_outer_faces_with_tol(block, tol)` (default
  `block_face_functions::DEFAULT_TOL = 1e-8`); and
  `Face::match_indices_with_tol(other, tol)` (default
  `block_face_functions::VERTEX_MATCH_TOL = 1e-6`). The constants are now
  public.

- `docs/matching-contract-design.md`: the target design for a tolerance
  policy and a `MatchReport` API that can express ambiguity and partial
  success, and why it is wanted. Not implemented.

### Changed (breaking)

- `full_face_match(face_a, face_b, tol)` is now
  `full_face_match(face_a, block_a, face_b, block_b, tol)`, and
  `full_face_match_transformed` likewise takes the two blocks. The old
  corner-only behaviour is `corner_match` / `corner_match_transformed`.

## 0.1.15

### Fixed

- `connectivity` / `connectivity_fast` no longer use a fixed `1e-6` node
  matching tolerance. Coordinate storage (binary `f32`, or ASCII with a fixed
  number of significant digits) loses precision in proportion to coordinate
  magnitude, so on meshes whose coordinates are large the two stored copies of
  a shared interface node differ by far more than `1e-6` and the interface was
  silently reported as two outer faces. The tolerance is now derived from the
  mesh by `connectivity::adaptive_tolerance`: a `1e-6`-relative storage-noise
  estimate, floored at the historical `1e-6` (so nothing that matches today can
  stop matching) and capped at a quarter of the shortest distance between two
  corners of any cell in the mesh (so it cannot start matching faces that are
  genuinely apart, and cannot pair a node with a neighbour of its true partner
  even on strongly sheared cells, whose short face diagonal is much shorter
  than an edge). Meshes with coordinates of magnitude ≤ 1 get bit-identical
  behaviour.

  `connectivity_fast` uses the tolerance derived from the full-resolution
  blocks, not from the GCD-reduced grid it matches on, so both entry points use
  one tolerance for a given mesh. A reduced-grid tolerance would have a ceiling
  up to `gcd` times looser for no gain in recall, and `connectivity_fast`
  scales its matches back to full resolution without re-verifying them (on
  `VSPT_ASCII.xyz`, GCD 4, it would have been `3.82e-5` against `5.72e-6`).

  Verified no-op on the real 593-block CMC009 mesh (`RANS_009_refined2.p3d`,
  20.77M nodes): `max|coord|` 11 would ask for `1.1e-5`, but the mesh's own
  `2.37e-7` finest cell corner spacing (a wall-normal edge) clamps the result
  back to exactly `1e-6`; the corner-spacing pass itself takes about 50 ms
  there. Also a no-op on `VSPT_ASCII.xyz`: both entry points derive `5.72e-6`
  and report the same 2 interfaces / 11 outer faces as the fixed `1e-6` did.

### Added

- `connectivity::adaptive_tolerance(&blocks) -> Float`, `connectivity::TOL_FLOOR`
  and `connectivity::connectivity_with_tol(&blocks, tol)` (all re-exported at
  the crate root) — the derived tolerance, its floor, and an explicit-tolerance
  entry point. Existing `connectivity`/`connectivity_fast` signatures are
  unchanged.

## v0.1.9

### Fixed

- `face_matches_to_dict` now uses MatchPoint data with GCD scaling for Phase 2/3
  partial matches, preventing synthetic bounding-box corners from producing
  incorrect diagonal correspondence.
- Removed Phase 1 fast path in old `verify_periodicity` (in
  `rotational_periodicity.rs`) that bypassed corner correction for full-face
  matches.

### Added

- `verification.rs` module with permutation-matrix-based `verify_connectivity`
  and `verify_periodicity`. These extract canonical 2D grids and try all 8
  permutation matrices to determine the correct orientation.
- `serialization.rs` module with two JSON output formats:
  - Default (`lo`/`hi`): ascending bounds + `permutation_index` (0-7).
  - Diagonal (`lb`/`ub`): GlennHT-compatible format with direction encoded in
    bounds (in-plane: `permutation_index=-1`, cross-plane: actual index).

### Removed

- Dead `linear_real_transform` function (was unused).
- Made 4 internal-only functions private in `rotational_periodicity.rs`:
  `count_rotated_corners_on_face`, `periodicity_check_with_points`,
  `faces_support_direction`, `faces_support_any`.
