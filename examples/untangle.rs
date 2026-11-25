//! Untangle folded grid lines in a Plot3D mesh.
//!
//! Reads the mesh, runs [`plot3d::untangle_mesh`], prints the per-block
//! report (every reversed face and the folded edge it was localised to,
//! every moved node, the re-check), and writes `<stem>_untangled.<ext>` and
//! `<stem>_untangled_report.txt` next to the input, in the input's format.
//! On a refusal only the report is written. Refuses to overwrite an existing
//! output. The written mesh is read back and every node compared with the
//! repaired mesh, so the report states what the write itself changed.
//!
//! Usage:
//!   cargo run --release --example untangle -- <mesh> \
//!       [--binary] [--raw] [--f32] [--big-endian]
//!
//! Without `--binary` the input is read as ASCII. Binary defaults to Fortran
//! records, f64, little-endian.

use std::path::{Path, PathBuf};

use plot3d::{
    read_plot3d_ascii, read_plot3d_binary, untangle_mesh, write_plot3d, BinaryFormat, Block,
    Endian, FloatPrecision,
};

fn output_paths(input: &Path) -> (PathBuf, PathBuf) {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mesh".to_string());
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    let mesh = match input.extension() {
        Some(ext) => dir.join(format!("{stem}_untangled.{}", ext.to_string_lossy())),
        None => dir.join(format!("{stem}_untangled")),
    };
    let report = dir.join(format!("{stem}_untangled_report.txt"));
    (mesh, report)
}

/// Largest coordinate difference between two meshes of the same shape.
fn max_node_difference(a: &[Block], b: &[Block]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("block count {} vs {}", a.len(), b.len()));
    }
    let mut worst = 0.0f64;
    for (bi, (p, q)) in a.iter().zip(b).enumerate() {
        if (p.imax, p.jmax, p.kmax) != (q.imax, q.jmax, q.kmax) {
            return Err(format!("block {bi} dimensions differ"));
        }
        for (u, v) in [(&p.x, &q.x), (&p.y, &q.y), (&p.z, &q.z)] {
            for (s, t) in u.iter().zip(v.iter()) {
                worst = worst.max((*s as f64 - *t as f64).abs());
            }
        }
    }
    Ok(worst)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(input) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: untangle <mesh> [--binary] [--raw] [--f32] [--big-endian]");
        std::process::exit(2);
    };
    let flag = |f: &str| args.iter().any(|a| a == f);
    let binary = flag("--binary");
    let format = if flag("--raw") {
        BinaryFormat::Raw
    } else {
        BinaryFormat::Fortran
    };
    let precision = if flag("--f32") {
        FloatPrecision::F32
    } else {
        FloatPrecision::F64
    };
    let endian = if flag("--big-endian") {
        Endian::Big
    } else {
        Endian::Little
    };

    let input_path = Path::new(input);
    let (out_mesh, out_report) = output_paths(input_path);
    for p in [&out_mesh, &out_report] {
        if p.exists() {
            eprintln!("refusing to overwrite existing {}", p.display());
            std::process::exit(1);
        }
    }

    let t0 = std::time::Instant::now();
    let blocks = if binary {
        read_plot3d_binary(input, format, precision, endian)?
    } else {
        read_plot3d_ascii(input)?
    };
    println!(
        "read {} blocks from {} ({:.1}s)",
        blocks.len(),
        input,
        t0.elapsed().as_secs_f64()
    );

    let t1 = std::time::Instant::now();
    let (repaired, report) = match untangle_mesh(&blocks) {
        Ok(r) => r,
        Err(e) => {
            // Refused: nothing is written but the report, which says why.
            let text = format!("input:  {input}\noutput: none (refused)\n\n{e}");
            std::fs::write(&out_report, &text)?;
            eprint!("{e}");
            eprintln!("wrote {}", out_report.display());
            std::process::exit(1);
        }
    };
    println!("untangle ({:.1}s)", t1.elapsed().as_secs_f64());
    print!("{report}");

    let out_str = out_mesh.to_string_lossy().into_owned();
    write_plot3d(&out_str, &repaired, binary, format, precision, endian)?;
    let reread = if binary {
        read_plot3d_binary(&out_str, format, precision, endian)?
    } else {
        read_plot3d_ascii(&out_str)?
    };
    let write_error = max_node_difference(&repaired, &reread)?;

    let mut text = format!("input:  {input}\noutput: {out_str}\n");
    text.push_str(&format!(
        "write round trip: max |written - repaired| over all nodes = {write_error:e}\n\n"
    ));
    text.push_str(&report.format_report());
    std::fs::write(&out_report, text)?;
    println!("wrote {}", out_mesh.display());
    println!("wrote {}", out_report.display());
    println!("write round trip: max |written - repaired| = {write_error:e}");
    Ok(())
}
