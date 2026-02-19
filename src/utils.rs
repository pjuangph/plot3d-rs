use byteorder::{BigEndian, ByteOrder, LittleEndian};
use std::io::{self, Read, Write};

use crate::block::Block;
use crate::Float;

// ---------------------------------------------------------------------------
// GCD helpers
// ---------------------------------------------------------------------------

/// Greatest common divisor of two integers.
pub(crate) fn gcd_two(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Greatest common divisor of three integers.
pub(crate) fn gcd_three(a: usize, b: usize, c: usize) -> usize {
    gcd_two(gcd_two(a, b), c)
}

/// Compute the minimum GCD of `(imax-1, jmax-1, kmax-1)` across all blocks.
///
/// Returns at least 1.
pub fn compute_min_gcd(blocks: &[Block]) -> usize {
    blocks
        .iter()
        .map(|b| {
            gcd_three(
                b.imax.saturating_sub(1),
                b.jmax.saturating_sub(1),
                b.kmax.saturating_sub(1),
            )
        })
        .filter(|&g| g > 0)
        .min()
        .unwrap_or(1)
        .max(1)
}

// ---------------------------------------------------------------------------
// 3-D vector helpers (used across multiple modules)
// ---------------------------------------------------------------------------

/// Component-wise subtraction.
#[inline]
pub(crate) fn sub3(a: [Float; 3], b: [Float; 3]) -> [Float; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product.
#[inline]
pub(crate) fn cross3(a: [Float; 3], b: [Float; 3]) -> [Float; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product.
#[inline]
pub(crate) fn dot3(a: [Float; 3], b: [Float; 3]) -> Float {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm.
#[inline]
pub(crate) fn vec_norm3(a: [Float; 3]) -> Float {
    dot3(a, a).sqrt()
}

/// Euclidean distance between two 3-D points.
#[inline]
pub(crate) fn distance3(a: [Float; 3], b: [Float; 3]) -> Float {
    vec_norm3(sub3(a, b))
}

/// Apply a 3×3 rotation matrix to a point.
#[inline]
pub(crate) fn apply_rotation(p: [Float; 3], rot: [[Float; 3]; 3]) -> [Float; 3] {
    [
        rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2],
        rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2],
        rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2],
    ]
}

// ---------------------------------------------------------------------------
// FaceKey – shared identifier type for faces
// ---------------------------------------------------------------------------

/// Compact identifier for a face: `(block_index, imin, jmin, kmin, imax, jmax, kmax)`.
pub type FaceKey = (usize, usize, usize, usize, usize, usize, usize);
#[derive(Copy, Clone, Debug)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    pub fn is_host_little() -> bool {
        // stable host endianness check
        cfg!(target_endian = "little")
    }
    pub fn read_u32(buf: &[u8], e: Endian) -> u32 {
        match e {
            Endian::Little => LittleEndian::read_u32(buf),
            Endian::Big => BigEndian::read_u32(buf),
        }
    }
    pub fn write_u32(buf: &mut [u8], v: u32, e: Endian) {
        match e {
            Endian::Little => LittleEndian::write_u32(buf, v),
            Endian::Big => BigEndian::write_u32(buf, v),
        }
    }
    pub fn read_f32_slice(buf: &[u8], e: Endian) -> Vec<f32> {
        let mut out = vec![0f32; buf.len() / 4];
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            let u = Self::read_u32(chunk, e);
            out[i] = f32::from_bits(u);
        }
        out
    }
    pub fn write_f32_slice(v: &[f32], e: Endian) -> Vec<u8> {
        let mut out = vec![0u8; v.len() * 4];
        for (i, f) in v.iter().enumerate() {
            let mut b = [0u8; 4];
            Self::write_u32(&mut b, f.to_bits(), e);
            out[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
        out
    }
    pub fn read_f64_slice(buf: &[u8], e: Endian) -> Vec<f64> {
        let mut out = vec![0f64; buf.len() / 8];
        for (i, chunk) in buf.chunks_exact(8).enumerate() {
            let top = Self::read_u32(&chunk[0..4], e) as u64;
            let bot = Self::read_u32(&chunk[4..8], e) as u64;
            // join two u32 as u64 with endianness already respected
            let bits = (top << 32) | bot;
            out[i] = f64::from_bits(bits);
        }
        out
    }
    pub fn write_f64_slice(v: &[f64], e: Endian) -> Vec<u8> {
        let mut out = vec![0u8; v.len() * 8];
        for (i, f) in v.iter().enumerate() {
            let bits = f.to_bits();
            let top = (bits >> 32) as u32;
            let bot = (bits & 0xFFFF_FFFF) as u32;
            let mut a = [0u8; 4];
            let mut b = [0u8; 4];
            Self::write_u32(&mut a, top, e);
            Self::write_u32(&mut b, bot, e);
            out[i * 8..i * 8 + 4].copy_from_slice(&a);
            out[i * 8 + 4..i * 8 + 8].copy_from_slice(&b);
        }
        out
    }
}

// Fortran unformatted record helpers: [len:u32] payload [len:u32]
pub fn write_fortran_record<W: Write>(w: &mut W, payload: &[u8], endian: Endian) -> io::Result<()> {
    let mut lenb = [0u8; 4];
    Endian::write_u32(&mut lenb, payload.len() as u32, endian);
    w.write_all(&lenb)?;
    w.write_all(payload)?;
    w.write_all(&lenb)?;
    Ok(())
}
// BORROW the reader
pub fn read_fortran_record<R: Read>(r: &mut R, endian: Endian) -> io::Result<Vec<u8>> {
    let mut lenb = [0u8; 4];
    r.read_exact(&mut lenb)?;
    let len = Endian::read_u32(&lenb, endian) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    r.read_exact(&mut lenb)?;
    let len2 = Endian::read_u32(&lenb, endian) as usize;
    if len != len2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Fortran record length mismatch",
        ));
    }
    Ok(buf)
}
