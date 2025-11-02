use std::vec;

use plot3d::{
    block, connectivity_fast, create_rotation_matrix, read_plot3d_ascii, read_plot3d_binary,
    rotate_block_with_matrix, rotated_periodicity, translational_periodicity, FaceMatchPrinter,
    FaceRecordTraits,
};

#[test]
fn rotational_periodicity_test() {
    let url = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/VSPT_ASCII.xyz";
    let ascii_path = "VSPT_ASCII.xyz";
    if !std::path::Path::new(ascii_path).exists() {
        let bytes = reqwest::blocking::get(url).unwrap().bytes().unwrap();
        std::fs::write(ascii_path, &bytes).unwrap();
    }

    let blocks = read_plot3d_ascii(ascii_path).unwrap();

    let number_of_blades = 55;
    let rotation_angle = 360.0 / number_of_blades as f64;
    let copies: usize = 3;

    let mut rotated_blocks = Vec::with_capacity(blocks.len() * copies);
    rotated_blocks.extend(blocks.iter().cloned());

    for copy_idx in 1..copies {
        let angle_rad = (rotation_angle * copy_idx as f64).to_radians();
        let rotation_matrix = create_rotation_matrix(angle_rad, 'x');
        for block in &blocks {
            rotated_blocks.push(rotate_block_with_matrix(block, rotation_matrix));
        }
    }

    assert_eq!(rotated_blocks.len(), blocks.len() * copies);

    let (face_matches, outer_faces) = connectivity_fast(&blocks);

    // face_matches.print();

    let (periodic_faces, outer_faces_rotated) = rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle,
        'x',
        true,
    );
    println!("Printing Periodic faces");
    periodic_faces.print();
    outer_faces_rotated.print();
}

#[test]
fn translational_periodicity_test() {
    let url = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/iso65_64blocks.xyz";
    let file_path = "iso65_64blocks.xyz";
    if !std::path::Path::new(file_path).exists() {
        let bytes = reqwest::blocking::get(url).unwrap().bytes().unwrap();
        std::fs::write(file_path, &bytes).unwrap();
    }

    let blocks = read_plot3d_binary(
        file_path,
        plot3d::BinaryFormat::Raw,
        plot3d::FloatPrecision::F32,
        plot3d::Endian::Little,
    )
    .unwrap();

    let (face_matches, outer_faces) = connectivity_fast(&blocks);

    let mut remaining_faces = outer_faces.clone();

    let (x_periodicity, rest) =
        translational_periodicity(&blocks, &remaining_faces, None, "x", None, 0.5, 1, 1, 1);
    remaining_faces = rest;

    let (y_periodicity, rest) =
        translational_periodicity(&blocks, &remaining_faces, None, "y", None, 0.5, 1, 1, 1);
    remaining_faces = rest;

    let (z_periodicity, rest) =
        translational_periodicity(&blocks, &remaining_faces, None, "z", None, 0.5, 1, 1, 1);
    remaining_faces = rest;
    let mut face_matches_all = face_matches;
    face_matches_all.extend(x_periodicity);
    face_matches_all.extend(y_periodicity);
    face_matches_all.extend(z_periodicity);
}
