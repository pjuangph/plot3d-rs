//! One-time helper: generates the subset indices file needed by
//! test_connectivity_subset_diagnostics.
//!
//! Run once:  cargo test --release --test test_generate_subset -- --nocapture

mod common;

use std::collections::HashSet;

use plot3d::{
    connectivity_fast, read_plot3d_binary, write_plot3d, BinaryFormat, Endian, FloatPrecision,
};

use common::{build_computed_keys, build_json_keys, GridProConnectivity};

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";
const JSON_FILE: &str =
    "/Users/pjuangph/Documents/GitHub/plot3d_troubleshoot/gridpro_connectivity.json";
const SUBSET_OUTPUT: &str = "/tmp/connectivity_debug_subset.p3d";
const SUBSET_INDICES: &str = "/tmp/connectivity_debug_subset_indices.json";

#[test]
fn generate_subset() {
    println!("\n=== Generating subset files ===\n");

    println!("Loading JSON...");
    let json_str = std::fs::read_to_string(JSON_FILE).expect("read JSON");
    let json_data: GridProConnectivity = serde_json::from_str(&json_str).expect("parse JSON");
    println!("  JSON face_matches: {}", json_data.face_matches.len());

    println!("Loading mesh (optimized reader)...");
    let start = std::time::Instant::now();
    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("read mesh");
    println!(
        "  Loaded {} blocks in {:.1}s",
        blocks.len(),
        start.elapsed().as_secs_f64()
    );

    println!("Running connectivity_fast...");
    let start = std::time::Instant::now();
    let (computed, _) = connectivity_fast(&blocks);
    println!(
        "  {} matches in {:.1}s",
        computed.len(),
        start.elapsed().as_secs_f64()
    );

    // Build key sets
    let json_keys = build_json_keys(&json_data.face_matches);
    let comp_keys = build_computed_keys(&computed);

    let missing: Vec<_> = json_keys.difference(&comp_keys).collect();
    let extra: Vec<_> = comp_keys.difference(&json_keys).collect();
    println!(
        "  Exact: {}, Missing: {}, Extra: {}",
        json_keys.intersection(&comp_keys).count(),
        missing.len(),
        extra.len()
    );

    // Collect mismatch block indices
    let mut indices: HashSet<usize> = HashSet::new();
    for k in &missing {
        indices.insert(k.face_a.block_index);
        indices.insert(k.face_b.block_index);
    }
    for k in &extra {
        indices.insert(k.face_a.block_index);
        indices.insert(k.face_b.block_index);
    }
    let mut sorted: Vec<usize> = indices.into_iter().collect();
    sorted.sort();
    println!("  Mismatch blocks: {} / {}", sorted.len(), blocks.len());

    // Write subset mesh
    let subset: Vec<_> = sorted.iter().map(|&i| blocks[i].clone()).collect();
    write_plot3d(
        SUBSET_OUTPUT,
        &subset,
        true,
        plot3d::write::BinaryFormat::Raw,
        plot3d::write::FloatPrecision::F64,
        Endian::Little,
    )
    .expect("write subset");
    println!("  Wrote {} blocks to {}", subset.len(), SUBSET_OUTPUT);

    // Write indices file
    let json = serde_json::to_string(&sorted).unwrap();
    std::fs::write(SUBSET_INDICES, &json).expect("write indices");
    println!("  Wrote indices to {}", SUBSET_INDICES);

    println!("\nDone! Now run: cargo test --release --test test_connectivity_subset_diagnostics -- --nocapture");
}
