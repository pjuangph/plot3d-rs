use plot3d::write::{BinaryFormat, FloatPrecision};
use plot3d::Endian;
use plot3d::{
    combine_nxnxn_cubes_mixed_pairs, connectivity_fast, read_plot3d_ascii, write_plot3d, Block,
};

#[test]
fn merge_block_test() {
    let ascii_path = "weld_ascii.xyz";
    let blocks = read_plot3d_ascii(ascii_path).unwrap();
    // Capture the source bounds so we can confirm merged/exported geometry stays consistent.
    let original_bounds = calc_bounds(&blocks);

    let (face_matches, _outer_faces) = connectivity_fast(&blocks);

    // Collect merged blocks from the connectivity-driven cube grouping.
    let merged = combine_nxnxn_cubes_mixed_pairs(&blocks, &face_matches, 3, None);

    let merged_blocks: Vec<_> = merged.into_iter().map(|(block, _ids)| block).collect();

    let (_face_matches_merged, _outer_faces_merged) = connectivity_fast(&merged_blocks);

    let output_path = "weld_ascii_3x3x3_out.xyz";
    // Persist merged geometry to disk so we can verify the round-trip readback.
    write_plot3d(
        output_path,
        &merged_blocks,
        false,
        BinaryFormat::Raw,
        FloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    let exported_blocks = read_plot3d_ascii(output_path).unwrap();

    assert_eq!(merged_blocks.len(), exported_blocks.len());
    assert_blocks_match(&merged_blocks, &exported_blocks, 1e-8);

    let merged_bounds = calc_bounds(&merged_blocks);
    let exported_bounds = calc_bounds(&exported_blocks);

    // Bounds should be unchanged across the original, merged, and exported versions.
    assert_bounds_close(&merged_bounds, &original_bounds, 1e-8);
    assert_bounds_close(&exported_bounds, &original_bounds, 1e-8);
}

fn assert_blocks_match(expected: &[Block], actual: &[Block], tol: f64) {
    assert_eq!(expected.len(), actual.len());
    for (lhs, rhs) in expected.iter().zip(actual.iter()) {
        assert_eq!(lhs.imax, rhs.imax);
        assert_eq!(lhs.jmax, rhs.jmax);
        assert_eq!(lhs.kmax, rhs.kmax);
        assert_vec_close(lhs.x_slice(), rhs.x_slice(), tol);
        assert_vec_close(lhs.y_slice(), rhs.y_slice(), tol);
        assert_vec_close(lhs.z_slice(), rhs.z_slice(), tol);
    }
}

fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "value mismatch at index {idx}: {a} vs {b} (tol {tol})"
        );
    }
}

fn calc_bounds(blocks: &[Block]) -> [[f64; 2]; 3] {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for block in blocks {
        for &x in block.x_slice() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        for &y in block.y_slice() {
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        for &z in block.z_slice() {
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
    }

    [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
}

fn assert_bounds_close(lhs: &[[f64; 2]; 3], rhs: &[[f64; 2]; 3], tol: f64) {
    for (axis_idx, (lhs_axis, rhs_axis)) in lhs.iter().zip(rhs.iter()).enumerate() {
        for (bound_idx, (a, b)) in lhs_axis.iter().zip(rhs_axis.iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "bound mismatch axis {axis_idx} bound {bound_idx}: {a} vs {b} (tol {tol})"
            );
        }
    }
}
