# Changelog

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
