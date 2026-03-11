use plot3d::{connectivity_fast, read_plot3d_ascii, verify_connectivity};

#[test]
fn test_cross_plane_connectivity() {
    let path = "tests/data/cross_plane_pair.p3d";
    let blocks = read_plot3d_ascii(path).expect("Failed to read cross_plane_pair.p3d");
    assert_eq!(blocks.len(), 2, "Expected 2 blocks");

    let (matches, _outer) = connectivity_fast(&blocks);
    assert_eq!(matches.len(), 1, "Expected exactly 1 face match for cross-plane pair");

    // Verify all matched points have the same coordinates (within tolerance)
    let (verified, cross_plane) = verify_connectivity(&blocks, &matches, 1e-6);
    let total_verified = verified.len() + cross_plane.len();
    assert_eq!(total_verified, 1, "The single match should be verified");
}
