use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use crate::block::Block;
use crate::utils::{self, read_fortran_record, Endian};

#[derive(Copy, Clone, Debug)]
pub enum BinaryFormat {
    Fortran,
    Raw,
}

#[derive(Copy, Clone, Debug)]
pub enum FloatPrecision {
    F32,
    F64,
}

// If you want to re-export Endian from this module, do it from the original path:
pub use crate::utils::Endian as EndianOrder;

pub fn read_plot3d_ascii(path: &str) -> io::Result<Vec<Block>> {
    let f = File::open(path)?;
    let mut rdr = BufReader::new(f);

    // first non-empty line = nblocks
    let mut first = String::new();
    loop {
        first.clear();
        let n = rdr.read_line(&mut first)?;
        if n == 0 {
            return Err(ioerr("empty file"));
        }
        if !first.trim().is_empty() {
            break;
        }
    }
    let nblocks: usize = first
        .split_whitespace()
        .next()
        .ok_or_else(|| ioerr("bad nblocks"))?
        .parse()
        .map_err(|_| ioerr("bad nblocks"))?;

    // dims
    let mut dims = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        let mut line = String::new();
        loop {
            line.clear();
            rdr.read_line(&mut line)?;
            if !line.trim().is_empty() {
                break;
            }
        }
        let mut it = line.split_whitespace();
        let imax: usize = it
            .next()
            .ok_or_else(|| ioerr("bad dims"))?
            .parse()
            .map_err(|_| ioerr("bad dims"))?;
        let jmax: usize = it
            .next()
            .ok_or_else(|| ioerr("bad dims"))?
            .parse()
            .map_err(|_| ioerr("bad dims"))?;
        let kmax: usize = it
            .next()
            .ok_or_else(|| ioerr("bad dims"))?
            .parse()
            .map_err(|_| ioerr("bad dims"))?;
        dims.push((imax, jmax, kmax));
    }

    // read N floats from ASCII
    fn read_n(rdr: &mut BufReader<File>, n: usize) -> io::Result<Vec<f64>> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut line = String::new();
            let cnt = rdr.read_line(&mut line)?;
            if cnt == 0 {
                break;
            }
            for t in line.split_whitespace() {
                if out.len() == n {
                    break;
                }
                out.push(t.parse::<f64>().map_err(|_| ioerr("bad float"))?);
            }
        }
        if out.len() != n {
            return Err(ioerr("unexpected EOF in payload"));
        }
        Ok(out)
    }

    // payload
    let mut blocks = Vec::with_capacity(nblocks);
    for (imax, jmax, kmax) in dims {
        let n = imax * jmax * kmax;
        let x = read_n(&mut rdr, n)?;
        let y = read_n(&mut rdr, n)?;
        let z = read_n(&mut rdr, n)?;
        blocks.push(Block::new(imax, jmax, kmax, x, y, z));
    }
    Ok(blocks)
}

pub fn read_plot3d_binary(
    path: &str,
    format: BinaryFormat,
    precision: FloatPrecision,
    endian: Endian,
) -> io::Result<Vec<Block>> {
    let mut f = File::open(path)?;
    match format {
        BinaryFormat::Raw => read_binary_raw(&mut f, precision, endian),
        BinaryFormat::Fortran => read_binary_fortran(&mut f, precision, endian),
    }
}

fn read_binary_raw(
    r: &mut impl Read,
    precision: FloatPrecision,
    endian: Endian,
) -> io::Result<Vec<Block>> {
    use byteorder::{BigEndian, LittleEndian, ReadBytesExt};

    // header
    let nblocks = match endian {
        Endian::Little => r.read_u32::<LittleEndian>()?,
        Endian::Big => r.read_u32::<BigEndian>()?,
    } as usize;

    let mut dims = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        let imax = match endian {
            Endian::Little => r.read_u32::<LittleEndian>()?,
            Endian::Big => r.read_u32::<BigEndian>()?,
        } as usize;
        let jmax = match endian {
            Endian::Little => r.read_u32::<LittleEndian>()?,
            Endian::Big => r.read_u32::<BigEndian>()?,
        } as usize;
        let kmax = match endian {
            Endian::Little => r.read_u32::<LittleEndian>()?,
            Endian::Big => r.read_u32::<BigEndian>()?,
        } as usize;
        dims.push((imax, jmax, kmax));
    }

    // payload
    let mut blocks = Vec::with_capacity(nblocks);
    for (imax, jmax, kmax) in dims {
        let n = imax * jmax * kmax;
        let x = read_vec_num(r, n, precision, endian)?;
        let y = read_vec_num(r, n, precision, endian)?;
        let z = read_vec_num(r, n, precision, endian)?;
        blocks.push(Block::new(imax, jmax, kmax, x, y, z));
    }
    Ok(blocks)
}

fn read_binary_fortran(
    r: &mut impl Read,
    precision: FloatPrecision,
    endian: Endian,
) -> io::Result<Vec<Block>> {
    // nblocks record
    let nb_rec = read_fortran_record(r, endian)?;
    if nb_rec.len() < 4 {
        return Err(ioerr("short nblocks record"));
    }
    let nblocks = utils::Endian::read_u32(&nb_rec[..4], endian) as usize;

    // dims per block (one record per block)
    let mut dims = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        let rec = read_fortran_record(r, endian)?;
        if rec.len() < 12 {
            return Err(ioerr("short dims record"));
        }
        let imax = utils::Endian::read_u32(&rec[0..4], endian) as usize;
        let jmax = utils::Endian::read_u32(&rec[4..8], endian) as usize;
        let kmax = utils::Endian::read_u32(&rec[8..12], endian) as usize;
        dims.push((imax, jmax, kmax));
    }

    // payload records: X, Y, Z for each block
    let mut blocks = Vec::with_capacity(nblocks);
    for (imax, jmax, kmax) in dims {
        let n = imax * jmax * kmax;

        let xr = read_fortran_record(r, endian)?;
        let x = match precision {
            FloatPrecision::F32 => utils::Endian::read_f32_slice(&xr, endian)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            FloatPrecision::F64 => utils::Endian::read_f64_slice(&xr, endian),
        };
        if x.len() != n {
            return Err(ioerr("X size mismatch"));
        }

        let yr = read_fortran_record(r, endian)?;
        let y = match precision {
            FloatPrecision::F32 => utils::Endian::read_f32_slice(&yr, endian)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            FloatPrecision::F64 => utils::Endian::read_f64_slice(&yr, endian),
        };
        if y.len() != n {
            return Err(ioerr("Y size mismatch"));
        }

        let zr = read_fortran_record(r, endian)?;
        let z = match precision {
            FloatPrecision::F32 => utils::Endian::read_f32_slice(&zr, endian)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            FloatPrecision::F64 => utils::Endian::read_f64_slice(&zr, endian),
        };
        if z.len() != n {
            return Err(ioerr("Z size mismatch"));
        }

        blocks.push(Block::new(imax, jmax, kmax, x, y, z));
    }

    Ok(blocks)
}

fn read_vec_num(
    r: &mut impl Read,
    n: usize,
    precision: FloatPrecision,
    endian: Endian,
) -> io::Result<Vec<f64>> {
    use byteorder::{BigEndian, LittleEndian, ReadBytesExt};

    let mut out = Vec::with_capacity(n);
    match (precision, endian) {
        (FloatPrecision::F32, Endian::Little) => {
            for _ in 0..n {
                out.push(r.read_f32::<LittleEndian>()? as f64);
            }
        }
        (FloatPrecision::F32, Endian::Big) => {
            for _ in 0..n {
                out.push(r.read_f32::<BigEndian>()? as f64);
            }
        }
        (FloatPrecision::F64, Endian::Little) => {
            for _ in 0..n {
                out.push(r.read_f64::<LittleEndian>()?);
            }
        }
        (FloatPrecision::F64, Endian::Big) => {
            for _ in 0..n {
                out.push(r.read_f64::<BigEndian>()?);
            }
        }
    }
    Ok(out)
}

fn ioerr(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}
