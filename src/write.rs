use std::fs::File;
use std::io::{Write, BufWriter};
use crate::block::Block;
use crate::utils::{self, Endian, write_fortran_record};

#[derive(Copy, Clone, Debug)]
pub enum BinaryFormat { Fortran, Raw }

#[derive(Copy, Clone, Debug)]
pub enum FloatPrecision { F32, F64 }

pub fn write_plot3d(
    path: &str,
    blocks: &[Block],
    binary: bool,
    format: BinaryFormat,
    precision: FloatPrecision,
    endian: Endian,
) -> std::io::Result<()> {
    if binary {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        match format {
            BinaryFormat::Raw => write_raw(&mut w, blocks, precision, endian),
            BinaryFormat::Fortran => write_fortran(&mut w, blocks, precision, endian),
        }
    } else {
        write_ascii(path, blocks)
    }
}

fn write_ascii(path: &str, blocks: &[Block]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "{}", blocks.len())?;
    for b in blocks {
        writeln!(w, "{} {} {}", b.imax, b.jmax, b.kmax)?;
    }
    for b in blocks {
        write_var_ascii(&mut w, &b.x)?;
        write_var_ascii(&mut w, &b.y)?;
        write_var_ascii(&mut w, &b.z)?;
    }
    Ok(())
}

fn write_var_ascii(w: &mut impl Write, v: &[f64]) -> std::io::Result<()> {
    let mut col = 0usize;
    for val in v {
        write!(w, "{:.8} ", val)?;
        col += 1;
        if col % 6 == 0 { writeln!(w)?; }
    }
    if col % 6 != 0 { writeln!(w)?; }
    Ok(())
}

fn write_raw(mut w: &mut impl Write, blocks: &[Block], precision: FloatPrecision, endian: Endian) -> std::io::Result<()> {
    use byteorder::{WriteBytesExt, LittleEndian, BigEndian};

    // header
    match endian {
        Endian::Little => w.write_u32::<LittleEndian>(blocks.len() as u32)?,
        Endian::Big    => w.write_u32::<BigEndian>(blocks.len() as u32)?,
    }
    for b in blocks {
        match endian {
            Endian::Little => {
                w.write_u32::<LittleEndian>(b.imax as u32)?;
                w.write_u32::<LittleEndian>(b.jmax as u32)?;
                w.write_u32::<LittleEndian>(b.kmax as u32)?;
            }
            Endian::Big => {
                w.write_u32::<BigEndian>(b.imax as u32)?;
                w.write_u32::<BigEndian>(b.jmax as u32)?;
                w.write_u32::<BigEndian>(b.kmax as u32)?;
            }
        }
    }

    // payload
    for b in blocks {
        write_vec_num(&mut w, &b.x, precision, endian)?;
        write_vec_num(&mut w, &b.y, precision, endian)?;
        write_vec_num(&mut w, &b.z, precision, endian)?;
    }
    Ok(())
}

fn write_fortran(mut w: &mut impl Write, blocks: &[Block], precision: FloatPrecision, endian: Endian) -> std::io::Result<()> {
    // header: [nblocks] as a record
    let mut nb = [0u8; 4];
    utils::Endian::write_u32(&mut nb, blocks.len() as u32, endian);
    write_fortran_record(&mut w, &nb, endian)?;

    // dims as one record per block
    for b in blocks {
        let mut rec = [0u8; 12];
        utils::Endian::write_u32(&mut rec[0..4],  b.imax as u32, endian);
        utils::Endian::write_u32(&mut rec[4..8],  b.jmax as u32, endian);
        utils::Endian::write_u32(&mut rec[8..12], b.kmax as u32, endian);
        write_fortran_record(&mut w, &rec, endian)?;
    }

    // payload: 3 records per block: X, Y, Z
    for b in blocks {
        match precision {
            FloatPrecision::F32 => {
                let xb = utils::Endian::write_f32_slice(&b.x.iter().map(|v| *v as f32).collect::<Vec<f32>>(), endian);
                write_fortran_record(&mut w, &xb, endian)?;
                let yb = utils::Endian::write_f32_slice(&b.y.iter().map(|v| *v as f32).collect::<Vec<f32>>(), endian);
                write_fortran_record(&mut w, &yb, endian)?;
                let zb = utils::Endian::write_f32_slice(&b.z.iter().map(|v| *v as f32).collect::<Vec<f32>>(), endian);
                write_fortran_record(&mut w, &zb, endian)?;
            }
            FloatPrecision::F64 => {
                let xb = utils::Endian::write_f64_slice(&b.x, endian);
                write_fortran_record(&mut w, &xb, endian)?;
                let yb = utils::Endian::write_f64_slice(&b.y, endian);
                write_fortran_record(&mut w, &yb, endian)?;
                let zb = utils::Endian::write_f64_slice(&b.z, endian);
                write_fortran_record(&mut w, &zb, endian)?;
            }
        }
    }

    Ok(())
}

fn write_vec_num(mut w: &mut impl Write, v: &[f64], precision: FloatPrecision, endian: Endian) -> std::io::Result<()> {
    use byteorder::{WriteBytesExt, LittleEndian, BigEndian};
    match (precision, endian) {
        (FloatPrecision::F32, Endian::Little) => for &f in v { w.write_f32::<LittleEndian>(f as f32)?; },
        (FloatPrecision::F32, Endian::Big)    => for &f in v { w.write_f32::<BigEndian>(f as f32)?; },
        (FloatPrecision::F64, Endian::Little) => for &f in v { w.write_f64::<LittleEndian>(f)?; },
        (FloatPrecision::F64, Endian::Big)    => for &f in v { w.write_f64::<BigEndian>(f)?; },
    }
    Ok(())
}
