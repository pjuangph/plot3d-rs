//! Cross-validation driver: discover connectivity + translational
//! periodicity on a mesh and dump canonical JSON for comparison with
//! Plot3D_utilities (Python).
//!
//! glennht-gpu uses this crate's verifiers at load time; if this
//! discovery disagrees with the Python tooling on the same mesh, the
//! GPU may reject interfaces the Python-side exporter (tgs-py)
//! believes are fine. See compressor-loss-model
//! `campaigns/cascade/test/compare_connectivity.py` for the diff step.
//!
//! Usage:
//!   cargo run --release --example cascade_connectivity -- \
//!       <mesh.ascii.xyz> <axis: x|y|z> [out.json]

use plot3d::{connectivity_fast, read_plot3d_ascii, translational_periodicity, FaceMatch};

fn canon(m: &FaceMatch) -> serde_json::Value {
    // Canonical form matching the Python side: per-side block index +
    // per-axis sorted lo/hi corners; sides ordered (block, lo) ascending.
    let side = |b: &plot3d::FaceRecord| {
        let lo = [b.il.min(b.ih), b.jl.min(b.jh), b.kl.min(b.kh)];
        let hi = [b.il.max(b.ih), b.jl.max(b.jh), b.kl.max(b.kh)];
        (b.block_index, lo, hi)
    };
    let s1 = side(&m.block1);
    let s2 = side(&m.block2);
    let (a, b) = if (s2.0, s2.1) < (s1.0, s1.1) { (s2, s1) } else { (s1, s2) };
    serde_json::json!({
        "b1": a.0, "lo1": a.1, "hi1": a.2,
        "b2": b.0, "lo2": b.1, "hi2": b.2,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cascade_connectivity <mesh.ascii.xyz> <axis> [out.json]");
        std::process::exit(2);
    }
    let mesh_path = &args[1];
    let axis = &args[2];
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "connectivity_rust.json".to_string());

    let blocks = read_plot3d_ascii(mesh_path)?;
    println!(
        "blocks: {:?}",
        blocks.iter().map(|b| (b.imax, b.jmax, b.kmax)).collect::<Vec<_>>()
    );

    let t0 = std::time::Instant::now();
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!(
        "connectivity_fast: {} matches, {} outer faces  ({:.1}s)",
        face_matches.len(),
        outer_faces.len(),
        t0.elapsed().as_secs_f64()
    );

    let t1 = std::time::Instant::now();
    // Signature: (blocks, outer_faces, delta, direction, node_tol_xyz,
    //             min_shared_frac, min_shared_abs, stride_u, stride_v)
    // Defaults mirror the Python side: delta auto, adaptive tol,
    // min_shared_frac=0.02, min_shared_abs=4, stride 1.
    let (periodic, remaining_outer) =
        translational_periodicity(&blocks, &outer_faces, None, axis, None, 0.02, 4, 1, 1);
    println!(
        "translational_periodicity({}): {} periodic pairs, {} outer left  ({:.1}s)",
        axis,
        periodic.len(),
        remaining_outer.len(),
        t1.elapsed().as_secs_f64()
    );

    let mut fm: Vec<serde_json::Value> = face_matches.iter().map(canon).collect();
    fm.sort_by_key(|v| v.to_string());
    let mut pm: Vec<serde_json::Value> = periodic.iter().map(canon).collect();
    pm.sort_by_key(|v| v.to_string());

    let result = serde_json::json!({
        "tool": "plot3d-rs",
        "mesh": mesh_path,
        "face_matches": fm,
        "periodic_matches": pm,
        "n_outer_after": remaining_outer.len(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
    println!("wrote {}", out_path);
    Ok(())
}
