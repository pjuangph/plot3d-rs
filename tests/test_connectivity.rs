use plot3d::{block_face_functions, connectivity_fast, read_plot3d_ascii, FaceRecord};
#[test]
fn test_connectivity() {
    // download mesh
    let url = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/VSPT_ASCII.xyz";
    let ascii_path = "VSPT_ASCII.xyz";
    if !std::path::Path::new(ascii_path).exists() {
        let bytes = reqwest::blocking::get(url).unwrap().bytes().unwrap();
        std::fs::write(ascii_path, &bytes).unwrap();
    }

    // read ASCII
    let blocks = read_plot3d_ascii(ascii_path).unwrap();
    assert!(blocks.len() == 2);
    let (i, j, k) = (8, 8, 8);

    println!("Block 0 printing at {i}, {j}, {k}");
    blocks[0].print_xyz(i, j, k);

    let factor = 4;
    let reduced_blocks = block_face_functions::reduce_blocks(&blocks, factor);
    println!("Reduced block 0 printing at {i}, {j}, {k}");
    reduced_blocks[0].print_xyz(i / factor, j / factor, k / factor);

    // verify all nodes align between original and reduced block
    let original = &blocks[0];
    let reduced = &reduced_blocks[0];
    for (ri, i_idx) in (0..original.imax).step_by(factor).enumerate() {
        for (rj, j_idx) in (0..original.jmax).step_by(factor).enumerate() {
            for (rk, k_idx) in (0..original.kmax).step_by(factor).enumerate() {
                let (ox, oy, oz) = original.xyz(i_idx, j_idx, k_idx);
                let (rx, ry, rz) = reduced.xyz(ri, rj, rk);
                assert!((ox - rx).abs() < 1e-9);
                assert!((oy - ry).abs() < 1e-9);
                assert!((oz - rz).abs() < 1e-9);
            }
        }
    }

    // // Find Connectivity
    let (matches, outer_faces) = connectivity_fast(&blocks);

    fn face_summary(
        face: &FaceRecord,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        Option<usize>,
    ) {
        (
            face.block_index,
            face.imin,
            face.jmin,
            face.kmin,
            face.imax,
            face.jmax,
            face.kmax,
            face.id,
        )
    }

    let matches_summary: Vec<_> = matches
        .iter()
        .map(|m| {
            (
                face_summary(&m.block1),
                face_summary(&m.block2),
                m.points.len(),
            )
        })
        .collect();

    let expected_matches = vec![
        (
            (0, 128, 0, 32, 256, 100, 32, None),
            (1, 40, 0, 0, 168, 100, 0, None),
            858,
        ),
        (
            (0, 0, 0, 0, 0, 100, 32, None),
            (0, 256, 0, 0, 256, 100, 32, None),
            0,
        ),
    ];
    assert_eq!(matches_summary, expected_matches);

    let outer_faces_summary: Vec<_> = outer_faces.iter().map(face_summary).collect();
    let expected_outer_faces = vec![
        (0, 0, 0, 0, 256, 0, 32, Some(1)),
        (0, 0, 100, 0, 256, 100, 32, Some(2)),
        (0, 0, 0, 0, 256, 100, 0, Some(3)),
        (0, 0, 0, 32, 128, 100, 32, Some(4)),
        (1, 0, 0, 0, 0, 100, 52, Some(5)),
        (1, 268, 0, 0, 268, 100, 52, Some(6)),
        (1, 0, 0, 0, 268, 0, 52, Some(7)),
        (1, 0, 100, 0, 268, 100, 52, Some(8)),
        (1, 0, 0, 52, 268, 100, 52, Some(9)),
        (1, 0, 0, 0, 40, 100, 0, Some(10)),
        (1, 168, 0, 0, 268, 100, 0, Some(11)),
    ];
    assert_eq!(outer_faces_summary, expected_outer_faces);
}
