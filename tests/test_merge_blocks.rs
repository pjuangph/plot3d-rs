use std::path::Path;

use plot3d::write::{BinaryFormat as WriteBinaryFormat, FloatPrecision as WriteFloatPrecision};
use plot3d::{
    block_face_functions, combine_nxnxn_cubes_mixed_pairs, connectivity_fast, read_plot3d_ascii,
    read_plot3d_binary, write_plot3d, BinaryFormat as ReadBinaryFormat, Block, Endian,
    FloatPrecision as ReadFloatPrecision,
};

#[test]
fn merge_block_test() {
    let ascii_path = Path::new("weld_ascii.xyz");
    let reduced_binary_path = Path::new("weld_binary_reduced.xyzb");
    let merged_binary_path = Path::new("weld_binary_reduced_3x3x3_out.xyzb");

    let reduced_blocks = if reduced_binary_path.exists() {
        read_plot3d_binary(
            reduced_binary_path
                .to_str()
                .expect("valid reduced binary path"),
            ReadBinaryFormat::Raw,
            ReadFloatPrecision::F32,
            Endian::Little,
        )
        .unwrap()
    } else {
        let ascii_blocks =
            read_plot3d_ascii(ascii_path.to_str().expect("valid ascii path")).unwrap();
        let gcd = ascii_blocks
            .iter()
            .map(|b| gcd_three(b.imax - 1, b.jmax - 1, b.kmax - 1))
            .min()
            .unwrap_or(1)
            .max(1);
        let reduced = if gcd > 1 {
            block_face_functions::reduce_blocks(&ascii_blocks, gcd)
        } else {
            ascii_blocks
        };
        write_plot3d(
            reduced_binary_path
                .to_str()
                .expect("valid reduced binary path"),
            &reduced,
            true,
            WriteBinaryFormat::Raw,
            WriteFloatPrecision::F32,
            Endian::Little,
        )
        .unwrap();
        reduced
    };

    let reference_bounds = calc_bounds(&reduced_blocks);

    let (face_matches, _outer_faces) = connectivity_fast(&reduced_blocks);
    let merged = combine_nxnxn_cubes_mixed_pairs(&reduced_blocks, &face_matches, 3, None);
    let merged_blocks: Vec<_> = merged.into_iter().map(|(block, _ids)| block).collect();

    write_plot3d(
        merged_binary_path
            .to_str()
            .expect("valid merged binary path"),
        &merged_blocks,
        true,
        WriteBinaryFormat::Raw,
        WriteFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    let merged_blocks_disk = read_plot3d_binary(
        merged_binary_path
            .to_str()
            .expect("valid merged binary path"),
        ReadBinaryFormat::Raw,
        ReadFloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    assert_eq!(merged_blocks.len(), merged_blocks_disk.len());
    assert_blocks_match(&merged_blocks, &merged_blocks_disk, 1e-8);

    let merged_bounds = calc_bounds(&merged_blocks);
    let merged_disk_bounds = calc_bounds(&merged_blocks_disk);

    assert_bounds_close(&merged_bounds, &reference_bounds, 1e-8);
    assert_bounds_close(&merged_disk_bounds, &reference_bounds, 1e-8);
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
        let delta = (a - b).abs();
        if delta <= tol {
            continue;
        }
        let a32 = *a as f32;
        let b32 = *b as f32;
        let delta32 = (a32 - b32).abs();
        assert!(
            delta32 == 0.0,
            "value mismatch at index {idx}: {a} vs {b} (tol {tol}); f32 delta {delta32}"
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
            let delta = (a - b).abs();
            if delta <= tol {
                continue;
            }
            let a32 = *a as f32;
            let b32 = *b as f32;
            if ((a32 - b32).abs() as f64) <= tol {
                continue;
            }
            panic!(
                "bound mismatch axis {axis_idx} bound {bound_idx}: {a} vs {b} (tol {tol}); f32 delta {}",
                (a32 - b32).abs()
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
