use plot3d::utils::write_fortran_record;
use plot3d::{
    read_plot3d_ascii, read_plot3d_binary, write_plot3d, Block, BinaryFormat, Endian, Float,
    FloatPrecision,
};

#[test]
fn test_read_write_roundtrip() {
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

// ---------------------------------------------------------------------------
// Fortran-unformatted record-layout coverage.
//
// The standard PLOT3D convention writes X, Y, Z as a single concatenated
// record per block. Older plot3d-rs / Plot3D_utilities releases wrote them
// as three separate records. `write_fortran` now emits the concatenated
// layout; `read_binary_fortran` auto-detects and accepts both. These tests
// are hermetic (no network) — the S3 round-trip above can be flaky offline.
// ---------------------------------------------------------------------------

/// Two blocks of different shapes. Coordinates are distinct per axis and
/// per block, so a mis-split (X/Y/Z swap) or block mix-up is caught. All
/// values are small integers — exact in both f32 and f64.
fn sample_blocks() -> Vec<Block> {
    let mk = |imax: usize, jmax: usize, kmax: usize, base: Float| -> Block {
        let n = imax * jmax * kmax;
        let x: Vec<Float> = (0..n).map(|i| base + i as Float).collect();
        let y: Vec<Float> = (0..n).map(|i| base + 100.0 + i as Float).collect();
        let z: Vec<Float> = (0..n).map(|i| base + 200.0 + i as Float).collect();
        Block::new(imax, jmax, kmax, x, y, z)
    };
    vec![mk(3, 4, 2, 0.0), mk(5, 2, 3, 1000.0)]
}

fn assert_blocks_eq(a: &[Block], b: &[Block]) {
    assert_eq!(a.len(), b.len());
    for (ba, bb) in a.iter().zip(b.iter()) {
        assert_eq!((ba.imax, ba.jmax, ba.kmax), (bb.imax, bb.jmax, bb.kmax));
        assert_eq!(ba.x, bb.x);
        assert_eq!(ba.y, bb.y);
        assert_eq!(ba.z, bb.z);
    }
}

/// Encode coordinates as little-endian f64 bytes — for hand-crafting
/// legacy-layout Fortran files in the tests below.
fn encode_f64_le(v: &[Float]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&(x as f64).to_le_bytes());
    }
    out
}

#[test]
fn test_fortran_roundtrip_concatenated() {
    // write (concatenated layout) -> read -> exact equality, across both
    // endians and both precisions.
    let blocks = sample_blocks();
    for (elabel, endian) in [("le", Endian::Little), ("be", Endian::Big)] {
        for (plabel, precision) in [("f32", FloatPrecision::F32), ("f64", FloatPrecision::F64)] {
            let path = std::env::temp_dir()
                .join(format!("plot3d_fortran_rt_{elabel}_{plabel}.xyzb"));
            let path = path.to_str().unwrap();
            write_plot3d(path, &blocks, true, BinaryFormat::Fortran, precision, endian).unwrap();
            let round =
                read_plot3d_binary(path, BinaryFormat::Fortran, precision, endian).unwrap();
            assert_blocks_eq(&blocks, &round);
            let _ = std::fs::remove_file(path);
        }
    }
}

#[test]
fn test_fortran_reader_accepts_legacy_three_record_layout() {
    // Hand-craft a Fortran-unformatted file in the OLD three-records-per-
    // block layout (one record each for X, Y, Z) and confirm the reader
    // still picks it up — backward compatibility for files the previous
    // plot3d-rs writer produced.
    let blocks = sample_blocks();
    let endian = Endian::Little;
    let mut buf: Vec<u8> = Vec::new();

    let mut nb = [0u8; 4];
    Endian::write_u32(&mut nb, blocks.len() as u32, endian);
    write_fortran_record(&mut buf, &nb, endian).unwrap();

    for b in &blocks {
        let mut rec = [0u8; 12];
        Endian::write_u32(&mut rec[0..4], b.imax as u32, endian);
        Endian::write_u32(&mut rec[4..8], b.jmax as u32, endian);
        Endian::write_u32(&mut rec[8..12], b.kmax as u32, endian);
        write_fortran_record(&mut buf, &rec, endian).unwrap();
    }
    // THREE records per block — the legacy layout.
    for b in &blocks {
        write_fortran_record(&mut buf, &encode_f64_le(&b.x), endian).unwrap();
        write_fortran_record(&mut buf, &encode_f64_le(&b.y), endian).unwrap();
        write_fortran_record(&mut buf, &encode_f64_le(&b.z), endian).unwrap();
    }

    let path = std::env::temp_dir().join("plot3d_fortran_legacy3.xyzb");
    let path = path.to_str().unwrap();
    std::fs::write(path, &buf).unwrap();
    let round =
        read_plot3d_binary(path, BinaryFormat::Fortran, FloatPrecision::F64, endian).unwrap();
    assert_blocks_eq(&blocks, &round);
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_fortran_reader_rejects_bad_record_size() {
    // A payload record that is neither `npts` nor `3*npts` reals must
    // produce an error, not silently-wrong geometry.
    let endian = Endian::Little;
    let (imax, jmax, kmax) = (3usize, 4usize, 2usize);
    let npts = imax * jmax * kmax;
    let mut buf: Vec<u8> = Vec::new();

    let mut nb = [0u8; 4];
    Endian::write_u32(&mut nb, 1, endian);
    write_fortran_record(&mut buf, &nb, endian).unwrap();

    let mut rec = [0u8; 12];
    Endian::write_u32(&mut rec[0..4], imax as u32, endian);
    Endian::write_u32(&mut rec[4..8], jmax as u32, endian);
    Endian::write_u32(&mut rec[8..12], kmax as u32, endian);
    write_fortran_record(&mut buf, &rec, endian).unwrap();

    // A payload record of 2*npts reals — matches neither layout.
    let bad = vec![0.0 as Float; 2 * npts];
    write_fortran_record(&mut buf, &encode_f64_le(&bad), endian).unwrap();

    let path = std::env::temp_dir().join("plot3d_fortran_badsize.xyzb");
    let path = path.to_str().unwrap();
    std::fs::write(path, &buf).unwrap();
    let result = read_plot3d_binary(path, BinaryFormat::Fortran, FloatPrecision::F64, endian);
    assert!(result.is_err(), "bad Fortran record size must be rejected");
    let _ = std::fs::remove_file(path);
}
