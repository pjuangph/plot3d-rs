# Changelog

## Unreleased

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
