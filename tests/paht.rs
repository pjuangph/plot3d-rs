use plot3d::{
    read_plot3d_ascii, read_plot3d_binary, write_plot3d, BinaryFormat, Block, Endian,
    FloatPrecision,
};

#[test]
fn read_write_fortran_binary_roundtrip() {
    // download once
    let url = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/VSPT_ASCII.xyz";
    let ascii_path = "VSPT_ASCII.xyz";
    if !std::path::Path::new(ascii_path).exists() {
        let bytes = reqwest::blocking::get(url).unwrap().bytes().unwrap();
        std::fs::write(ascii_path, &bytes).unwrap();
    }

    // read ASCII
    let blocks = read_plot3d_ascii(ascii_path).unwrap();
    assert!(blocks.len() >= 1);

    // quick shape sanity
    for b in &blocks {
        assert_eq!(b.x.len(), b.imax * b.jmax * b.kmax);
        assert_eq!(b.y.len(), b.imax * b.jmax * b.kmax);
        assert_eq!(b.z.len(), b.imax * b.jmax * b.kmax);
    }

    // write Fortran-record binary (Float32 LE)
    let bin_path = "VSPT_BINARY.xyzb";
    write_plot3d(
        bin_path,
        &blocks,
        true,
        plot3d::write::BinaryFormat::Fortran,
        plot3d::write::FloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    assert!(std::path::Path::new(bin_path).exists());
    let size = std::fs::metadata(bin_path).unwrap().len();
    assert!(size > 0);

    // read it back
    let round = read_plot3d_binary(
        bin_path,
        BinaryFormat::Fortran,
        FloatPrecision::F32,
        Endian::Little,
    )
    .unwrap();

    assert_eq!(round.len(), blocks.len());
    for (a, b) in blocks.iter().zip(round.iter()) {
        assert_eq!(a.imax, b.imax);
        assert_eq!(a.jmax, b.jmax);
        assert_eq!(a.kmax, b.kmax);
        assert_eq!(a.x.len(), b.x.len());
        assert_eq!(a.y.len(), b.y.len());
        assert_eq!(a.z.len(), b.z.len());
    }
}
