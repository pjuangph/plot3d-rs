use plot3d::{block_face_functions, read_plot3d_ascii, connectivity_fast, FaceMatchPrinter, FaceRecordTraits};
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
    matches.print();
    outer_faces.print();
}
