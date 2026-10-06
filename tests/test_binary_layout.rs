//! Binary and ASCII Plot3D read/write coverage.
//!
//! Writer layout (Fortran-sequential, `BinaryFormat::Fortran`), every record
//! framed as `[len: u32][payload][len: u32]` in the file's endianness:
//!   1. nblocks record:          1 x u32
//!   2. dims, one record/block:  3 x u32 (imax, jmax, kmax), nblocks records
//!   3. coordinates, one record/block: X then Y then Z concatenated,
//!      3 * npts reals (f64 or f32 per `FloatPrecision`)
//!
//! Reader tolerance matrix (dims layout x coordinate layout):
//!   per-block dims  + concatenated XYZ   accepted (writer's own layout)
//!   per-block dims  + three X/Y/Z records accepted
//!   all-blocks dims + concatenated XYZ   rejected
//!   all-blocks dims + three X/Y/Z records rejected (layout of the Python
//!                                         `plot3d` package, fortran=True)

use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};

use plot3d::utils::write_fortran_record;
use plot3d::{
    read_plot3d_ascii, read_plot3d_binary, write_plot3d, BinaryFormat, Block, Endian, Float,
    FloatPrecision,
};

const LE: Endian = Endian::Little;

/// Non-cube blocks with non-trivial (non-integer, irrational-ish) values so
/// that any precision loss or axis mix-up shows up.
fn multi_blocks() -> Vec<Block> {
    let mk = |imax: usize, jmax: usize, kmax: usize, seed: Float| -> Block {
        let n = imax * jmax * kmax;
        let f = |off: Float, i: usize| {
            ((i as Float + seed) * 0.37 + off).sin() * 1.234_567_890_123 + off
        };
        let x = (0..n).map(|i| f(0.1, i)).collect();
        let y = (0..n).map(|i| f(10.3, i)).collect();
        let z = (0..n).map(|i| f(-20.7, i)).collect();
        Block::new(imax, jmax, kmax, x, y, z)
    };
    vec![
        mk(5, 3, 2, 1.0),
        mk(2, 7, 4, 2.0),
        mk(6, 2, 3, 3.0),
        mk(3, 3, 9, 4.0),
    ]
}

fn single_block() -> Vec<Block> {
    multi_blocks().into_iter().take(1).collect()
}

fn tmp(name: &str) -> (tempfile::TempDir, String) {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join(name).to_str().unwrap().to_string();
    (d, p)
}

fn assert_blocks_eq(a: &[Block], b: &[Block]) {
    assert_eq!(a.len(), b.len());
    for (ba, bb) in a.iter().zip(b) {
        assert_eq!((ba.imax, ba.jmax, ba.kmax), (bb.imax, bb.jmax, bb.kmax));
        assert_eq!(ba.x, bb.x);
        assert_eq!(ba.y, bb.y);
        assert_eq!(ba.z, bb.z);
    }
}

fn u32_rec(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_rec(vals: &[&[Float]]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| v.iter().flat_map(|x| (*x as f64).to_le_bytes()))
        .collect()
}

fn rec(buf: &mut Vec<u8>, payload: &[u8]) {
    write_fortran_record(buf, payload, LE).unwrap();
}

/// Build a little-endian f64 Fortran file in an arbitrary layout.
fn build(blocks: &[Block], dims_one_record: bool, xyz_three_records: bool) -> Vec<u8> {
    let mut buf = Vec::new();
    rec(&mut buf, &u32_rec(&[blocks.len() as u32]));
    if dims_one_record {
        let all: Vec<u32> = blocks
            .iter()
            .flat_map(|b| [b.imax as u32, b.jmax as u32, b.kmax as u32])
            .collect();
        rec(&mut buf, &u32_rec(&all));
    } else {
        for b in blocks {
            rec(
                &mut buf,
                &u32_rec(&[b.imax as u32, b.jmax as u32, b.kmax as u32]),
            );
        }
    }
    for b in blocks {
        if xyz_three_records {
            rec(&mut buf, &f64_rec(&[&b.x]));
            rec(&mut buf, &f64_rec(&[&b.y]));
            rec(&mut buf, &f64_rec(&[&b.z]));
        } else {
            rec(&mut buf, &f64_rec(&[&b.x, &b.y, &b.z]));
        }
    }
    buf
}

fn read_f64_le(path: &str) -> std::io::Result<Vec<Block>> {
    read_plot3d_binary(path, BinaryFormat::Fortran, FloatPrecision::F64, LE)
}

fn write_bin(path: &str, blocks: &[Block]) {
    write_plot3d(
        path,
        blocks,
        true,
        BinaryFormat::Fortran,
        FloatPrecision::F64,
        LE,
    )
    .unwrap();
}

#[test]
fn binary_f64_roundtrip_multi_and_single_block_exact() {
    for blocks in [multi_blocks(), single_block()] {
        let (_d, p) = tmp("rt.xyzb");
        write_bin(&p, &blocks);
        assert_blocks_eq(&blocks, &read_f64_le(&p).unwrap());
    }
}

#[test]
fn raw_binary_f64_roundtrip_exact() {
    let blocks = multi_blocks();
    let (_d, p) = tmp("raw.xyzb");
    write_plot3d(
        &p,
        &blocks,
        true,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        LE,
    )
    .unwrap();
    let back = read_plot3d_binary(&p, BinaryFormat::Raw, FloatPrecision::F64, LE).unwrap();
    assert_blocks_eq(&blocks, &back);
}

#[test]
fn ascii_f64_roundtrip_within_printed_precision() {
    for blocks in [multi_blocks(), single_block()] {
        let (_d, p) = tmp("rt.xyz");
        write_plot3d(
            &p,
            &blocks,
            false,
            BinaryFormat::Fortran,
            FloatPrecision::F64,
            LE,
        )
        .unwrap();
        let back = read_plot3d_ascii(&p).unwrap();
        assert_eq!(back.len(), blocks.len());
        for (a, b) in blocks.iter().zip(&back) {
            assert_eq!((a.imax, a.jmax, a.kmax), (b.imax, b.jmax, b.kmax));
            for (va, vb) in [(&a.x, &b.x), (&a.y, &b.y), (&a.z, &b.z)] {
                for (p, q) in va.iter().zip(vb) {
                    // "{:23.15}" prints 15 decimals.
                    assert!((p - q).abs() <= 1e-14, "{p} vs {q}");
                }
            }
        }
    }
}

#[test]
fn file_sizes_match_expected_layout() {
    let blocks = multi_blocks();
    let nb = blocks.len();
    let npts: usize = blocks.iter().map(|b| b.imax * b.jmax * b.kmax).sum();

    let (_d, pb) = tmp("sz.xyzb");
    write_bin(&pb, &blocks);
    // 8 bytes of markers per record: nblocks + nb dims + nb coordinate records.
    let expected = 8 * (1 + 2 * nb) + 4 + 12 * nb + 3 * 8 * npts;
    assert_eq!(std::fs::metadata(&pb).unwrap().len() as usize, expected);

    let (_d2, pr) = tmp("sz_raw.xyzb");
    write_plot3d(
        &pr,
        &blocks,
        true,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        LE,
    )
    .unwrap();
    assert_eq!(
        std::fs::metadata(&pr).unwrap().len() as usize,
        4 + 12 * nb + 3 * 8 * npts
    );

    let (_d3, pa) = tmp("sz.xyz");
    write_plot3d(
        &pa,
        &blocks,
        false,
        BinaryFormat::Fortran,
        FloatPrecision::F64,
        LE,
    )
    .unwrap();
    let ascii = std::fs::metadata(&pa).unwrap().len() as usize;
    assert!(
        ascii > expected,
        "ASCII ({ascii}) should exceed binary ({expected})"
    );

    let single = single_block();
    let (_d4, ps) = tmp("sz1.xyzb");
    write_bin(&ps, &single);
    let n1 = single[0].imax * single[0].jmax * single[0].kmax;
    assert_eq!(
        std::fs::metadata(&ps).unwrap().len() as usize,
        8 * 3 + 4 + 12 + 24 * n1
    );
}

#[test]
fn writer_emits_exact_record_structure() {
    let blocks = multi_blocks();
    let (_d, p) = tmp("layout.xyzb");
    write_bin(&p, &blocks);
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes, build(&blocks, false, false));

    // Walk the records explicitly so a mismatch names the structure.
    let mut cur = &bytes[..];
    let mut records: Vec<Vec<u8>> = Vec::new();
    while !cur.is_empty() {
        let r = plot3d::utils::read_fortran_record(&mut cur, LE).unwrap();
        records.push(r);
    }
    assert_eq!(records.len(), 1 + 2 * blocks.len());
    assert_eq!(records[0], u32_rec(&[blocks.len() as u32]));
    for (i, b) in blocks.iter().enumerate() {
        assert_eq!(
            records[1 + i],
            u32_rec(&[b.imax as u32, b.jmax as u32, b.kmax as u32])
        );
        let n = b.imax * b.jmax * b.kmax;
        let coords = &records[1 + blocks.len() + i];
        assert_eq!(coords.len(), 3 * n * 8);
        assert_eq!(coords[..], f64_rec(&[&b.x, &b.y, &b.z])[..]);
    }
}

#[test]
fn writer_big_endian_markers() {
    let blocks = single_block();
    let (_d, p) = tmp("be.xyzb");
    write_plot3d(
        &p,
        &blocks,
        true,
        BinaryFormat::Fortran,
        FloatPrecision::F64,
        Endian::Big,
    )
    .unwrap();
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(&bytes[..12], &[0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 4]);
    let back =
        read_plot3d_binary(&p, BinaryFormat::Fortran, FloatPrecision::F64, Endian::Big).unwrap();
    assert_blocks_eq(&blocks, &back);
}

#[test]
fn reader_layout_matrix() {
    let blocks = multi_blocks();
    let mut outcomes = Vec::new();
    for (dims_one, xyz_three) in [(false, false), (false, true), (true, false), (true, true)] {
        let (_d, p) = tmp("matrix.xyzb");
        std::fs::write(&p, build(&blocks, dims_one, xyz_three)).unwrap();
        let res = catch_unwind(AssertUnwindSafe(|| read_f64_le(&p)));
        let ok = matches!(&res, Ok(Ok(b)) if {
            assert_blocks_eq(&blocks, b);
            true
        });
        outcomes.push((
            (dims_one, xyz_three),
            ok,
            format!("{:?}", res.map(|r| r.map(|b| b.len()))),
        ));
    }
    for (k, ok, msg) in &outcomes {
        eprintln!(
            "dims_one_record={} xyz_three_records={} -> accepted={ok} ({msg})",
            k.0, k.1
        );
    }
    assert!(outcomes[0].1, "per-block dims + concatenated XYZ");
    assert!(outcomes[1].1, "per-block dims + three XYZ records");
}

#[test]
#[ignore = "reader supports only one dims record per block; single all-blocks dims record is unsupported"]
fn reader_accepts_all_blocks_dims_record_concatenated_xyz() {
    let blocks = multi_blocks();
    let (_d, p) = tmp("a.xyzb");
    std::fs::write(&p, build(&blocks, true, false)).unwrap();
    assert_blocks_eq(&blocks, &read_f64_le(&p).unwrap());
}

#[test]
#[ignore = "reader supports only one dims record per block; Python plot3d package layout (all-blocks dims + 3 XYZ records) is unsupported"]
fn reader_accepts_python_plot3d_package_layout() {
    let blocks = multi_blocks();
    let (_d, p) = tmp("b.xyzb");
    std::fs::write(&p, build(&blocks, true, true)).unwrap();
    assert_blocks_eq(&blocks, &read_f64_le(&p).unwrap());
}

/// Run a read on raw bytes; it must return (not panic) and must be an error.
fn assert_read_errors(bytes: &[u8], what: &str) {
    let (_d, p) = tmp("bad.xyzb");
    std::fs::write(&p, bytes).unwrap();
    let res = catch_unwind(AssertUnwindSafe(|| read_f64_le(&p)));
    match res {
        Ok(Err(_)) => {}
        Ok(Ok(_)) => panic!("{what}: expected an error, got blocks"),
        Err(_) => panic!("{what}: reader panicked"),
    }
}

#[test]
fn truncated_files_error_without_panic() {
    let blocks = multi_blocks();
    let full = build(&blocks, false, false);
    for cut in [
        0,
        2,
        4,
        7,
        11,
        20,
        full.len() / 2,
        full.len() - 5,
        full.len() - 1,
    ] {
        assert_read_errors(&full[..cut], &format!("truncated at {cut}"));
    }
}

#[test]
fn wrong_record_sizes_error_without_panic() {
    // Trailing marker disagrees with leading marker.
    let mut buf = build(&single_block(), false, false);
    let last = buf.len() - 4;
    buf[last..].copy_from_slice(&1u32.to_le_bytes());
    assert_read_errors(&buf, "mismatched trailing marker");

    // Dims record too short.
    let mut buf = Vec::new();
    rec(&mut buf, &u32_rec(&[1]));
    rec(&mut buf, &u32_rec(&[2, 2]));
    assert_read_errors(&buf, "short dims record");

    // Empty nblocks record.
    let mut buf = Vec::new();
    rec(&mut buf, &[]);
    assert_read_errors(&buf, "empty nblocks record");

    // Payload record of 2*npts reals.
    let b = &single_block()[0];
    let mut buf = Vec::new();
    rec(&mut buf, &u32_rec(&[1]));
    rec(
        &mut buf,
        &u32_rec(&[b.imax as u32, b.jmax as u32, b.kmax as u32]),
    );
    rec(&mut buf, &f64_rec(&[&b.x, &b.y]));
    assert_read_errors(&buf, "payload of 2*npts");

    // Three-record layout with a short Z record.
    let mut buf = Vec::new();
    rec(&mut buf, &u32_rec(&[1]));
    rec(
        &mut buf,
        &u32_rec(&[b.imax as u32, b.jmax as u32, b.kmax as u32]),
    );
    rec(&mut buf, &f64_rec(&[&b.x]));
    rec(&mut buf, &f64_rec(&[&b.y]));
    rec(&mut buf, &f64_rec(&[&b.z[..1]]));
    assert_read_errors(&buf, "short Z record");
}

#[test]
fn non_fortran_files_error_without_panic() {
    // ASCII text read as binary.
    assert_read_errors(b"2\n3 3 3\n4 4 4\n0.0 1.0 2.0\n", "ASCII as binary");

    // Raw (headerless-marker) binary read as Fortran.
    let (_d, p) = tmp("raw.xyzb");
    write_plot3d(
        &p,
        &multi_blocks(),
        true,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        LE,
    )
    .unwrap();
    let bytes = std::fs::read(&p).unwrap();
    assert_read_errors(&bytes, "raw binary as Fortran");

    // Garbage bytes.
    assert_read_errors(&[0xFF; 64], "0xFF garbage");
}

#[test]
fn ascii_reader_errors_without_panic() {
    let (_d, p) = tmp("bad.xyz");
    let mut f = std::fs::File::create(&p).unwrap();
    write!(f, "1\n2 2 2\n0 1 2 3\n").unwrap();
    drop(f);
    assert!(read_plot3d_ascii(&p).is_err());
    std::fs::write(&p, "abc").unwrap();
    assert!(read_plot3d_ascii(&p).is_err());
    assert!(read_plot3d_ascii("/nonexistent/path.xyz").is_err());
}
