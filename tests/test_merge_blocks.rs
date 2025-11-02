use std::path::Path;

use plot3d::{
    block_face_functions,
    combine_nxnxn_cubes_mixed_pairs,
    connectivity_fast,
    read_plot3d_ascii,
    read_plot3d_binary,
    write_plot3d,
    BinaryFormat as ReadBinaryFormat,
    FloatPrecision as ReadFloatPrecision,
    Block,
    Endian,
};
use plot3d::write::{BinaryFormat as WriteBinaryFormat, FloatPrecision as WriteFloatPrecision};

#[test]
fn merge_block_test() {
    let ascii_path = "weld_ascii.xyz";
    let binary_path = "weld_binary.xyz";

    if !Path::new(binary_path).exists() {
        let ascii_blocks = read_plot3d_ascii(ascii_path).unwrap();
        write_plot3d(
            binary_path,
            &ascii_blocks,
            false,
            WriteBinaryFormat::Raw,
            WriteFloatPrecision::F32,
            Endian::Little,
        )
        .unwrap();
    }

    let blocks = read_plot3d_binary(
        binary_path,
        ReadBinaryFormat::Raw,
        ReadFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    let original_bounds = calc_bounds(&blocks);

    let gcd = blocks
        .iter()
        .map(|b| gcd_three(b.imax - 1, b.jmax - 1, b.kmax - 1))
        .min()
        .unwrap_or(1)
        .max(1);

    let reduced_blocks = if gcd > 1 {
        block_face_functions::reduce_blocks(&blocks, gcd)
    } else {
        blocks.clone()
    };

    let reduced_path = "weld_binary_reduced.xyz";
    write_plot3d(
        reduced_path,
        &reduced_blocks,
        false,
        WriteBinaryFormat::Raw,
        WriteFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    let (face_matches, _outer_faces) = connectivity_fast(&reduced_blocks);

    let merged = combine_nxnxn_cubes_mixed_pairs(&reduced_blocks, &face_matches, 3, None);

    let merged_blocks: Vec<_> = merged.into_iter().map(|(block, _ids)| block).collect();

    let (_face_matches_merged, _outer_faces_merged) = connectivity_fast(&merged_blocks);

    let output_path = "weld_binary_reduced_3x3x3_out.xyz";
    write_plot3d(
        output_path,
        &merged_blocks,
        false,
        WriteBinaryFormat::Raw,
        WriteFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    let exported_blocks = read_plot3d_binary(
        output_path,
        ReadBinaryFormat::Raw,
        ReadFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    assert_eq!(merged_blocks.len(), exported_blocks.len());
    assert_blocks_match(&merged_blocks, &exported_blocks, 1e-8);

    let merged_bounds = calc_bounds(&merged_blocks);
    let exported_bounds = calc_bounds(&exported_blocks);

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

fn gcd_two(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn gcd_three(a: usize, b: usize, c: usize) -> usize {
    gcd_two(gcd_two(a, b), c)
}
