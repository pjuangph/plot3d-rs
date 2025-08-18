use std::io::{self, Read, Write};
use byteorder::{ByteOrder, LittleEndian, BigEndian};
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
            Endian::Big    => BigEndian::read_u32(buf),
        }
    }
    pub fn write_u32(buf: &mut [u8], v: u32, e: Endian) {
        match e {
            Endian::Little => LittleEndian::write_u32(buf, v),
            Endian::Big    => BigEndian::write_u32(buf, v),
        }
    }
    pub fn read_f32_slice(buf: &[u8], e: Endian) -> Vec<f32> {
        let mut out = vec![0f32; buf.len()/4];
        for (i, chunk) in buf.chunks_exact(4).enumerate() {
            let u = Self::read_u32(chunk, e);
            out[i] = f32::from_bits(u);
        }
        out
    }
    pub fn write_f32_slice(v: &[f32], e: Endian) -> Vec<u8> {
        let mut out = vec![0u8; v.len()*4];
        for (i, f) in v.iter().enumerate() {
            let mut b = [0u8; 4];
            Self::write_u32(&mut b, f.to_bits(), e);
            out[i*4..i*4+4].copy_from_slice(&b);
        }
        out
    }
    pub fn read_f64_slice(buf: &[u8], e: Endian) -> Vec<f64> {
        let mut out = vec![0f64; buf.len()/8];
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
        let mut out = vec![0u8; v.len()*8];
        for (i, f) in v.iter().enumerate() {
            let bits = f.to_bits();
            let top = (bits >> 32) as u32;
            let bot = (bits & 0xFFFF_FFFF) as u32;
            let mut a = [0u8; 4];
            let mut b = [0u8; 4];
            Self::write_u32(&mut a, top, e);
            Self::write_u32(&mut b, bot, e);
            out[i*8..i*8+4].copy_from_slice(&a);
            out[i*8+4..i*8+8].copy_from_slice(&b);
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
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Fortran record length mismatch"));
    }
    Ok(buf)
}