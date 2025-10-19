use plot3d::{block_face_functions, read_plot3d_ascii};
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

    let reduced_blocks = block_face_functions::reduce_blocks(&blocks, 4);
    println!("Reduced block 0 printing at {i}, {j}, {k}");
    reduced_blocks[0].print_xyz(i / 4, j / 4, k / 4);

    // // Find Connectivity
    // let (matches, outer_faces) = connectivity_fast(&blocks);
    // matches.print();
    // outer_faces.print();

    // println!("Done!");
}
