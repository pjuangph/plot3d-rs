//! Full pipeline test: connectivity + rotational periodicity on the grid_packed mesh.
//!
//! Outputs a `tecplot.json` file containing face_matches, periodic faces,
//! and non-connected faces for visualization.
//!
//! Usage (release build required for large meshes):
//!   cargo test --release --test test_full_pipeline -- --nocapture

use std::collections::HashMap;

use plot3d::{
    connectivity_fast, face_matches_to_dict, read_plot3d_binary, rotated_periodicity,
    verify_periodicity, BinaryFormat, Endian, FaceMatch, FaceRecord, Float, FloatPrecision,
};
use serde_json::{json, Value};

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

const OUTPUT_DIR: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/";

const NBLADES: usize = 44;
const ROTATION_AXIS: char = 'x';

/// Convert a FaceRecord to the tecplot JSON format with UPPERCASE keys.
fn face_record_to_json(rec: &FaceRecord) -> Value {
    let mut obj = json!({
        "block_index": rec.block_index,
        "IMIN": rec.il,
        "JMIN": rec.jl,
        "KMIN": rec.kl,
        "IMAX": rec.ih,
        "JMAX": rec.jh,
        "KMAX": rec.kh,
    });
    if let Some(id) = rec.id {
        obj["id"] = json!(id);
    }
    obj
}

/// Convert a FaceMatch to the tecplot JSON format.
fn face_match_to_json(fm: &FaceMatch) -> Value {
    let mut obj = json!({
        "block1": face_record_to_json(&fm.block1),
        "block2": face_record_to_json(&fm.block2),
    });
    if let Some(ref orient) = fm.orientation {
        obj["orientation"] = json!({
            "u_reversed": orient.u_reversed,
            "v_reversed": orient.v_reversed,
            "swapped": orient.swapped,
        });
    }
    obj
}

#[test]
fn full_pipeline() {
    let rotation_angle_deg: Float = 360.0 / NBLADES as Float;
    let rotation_angle_rad: Float = rotation_angle_deg.to_radians();

    // ── Read mesh ──────────────────────────────────────────────────────
    println!("\nReading mesh...");
    let blocks = read_plot3d_binary(
        MESH_FILE,
        BinaryFormat::Raw,
        FloatPrecision::F64,
        Endian::Little,
    )
    .expect("Failed to read mesh file");
    println!("  {} blocks loaded", blocks.len());

    // ── Connectivity ───────────────────────────────────────────────────
    println!("\nRunning connectivity_fast...");
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!("  {} face matches", face_matches.len());
    println!("  {} non-connected faces", outer_faces.len());

    // Validate connectivity
    let validated_matches = face_matches_to_dict(&blocks, &face_matches);
    println!("  {} validated face matches", validated_matches.len());

    // ── Rotational periodicity ─────────────────────────────────────────
    println!(
        "\nRunning rotated_periodicity (nblades={}, angle={:.4} deg, axis='{}')...",
        NBLADES, rotation_angle_deg, ROTATION_AXIS
    );
    let (periodic_faces, remaining_outer) = rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle_deg,
        ROTATION_AXIS,
        true, // reduce_mesh (GCD)
    );
    println!("  {} periodic face pairs", periodic_faces.len());
    println!("  {} remaining non-connected faces", remaining_outer.len());

    // Verify periodicity
    println!("\nVerifying periodic matches...");
    let (verified, mismatched) = verify_periodicity(
        &blocks,
        &periodic_faces,
        rotation_angle_rad,
        ROTATION_AXIS,
        1e-4,
    );
    println!(
        "  {} verified, {} mismatched",
        verified.len(),
        mismatched.len()
    );

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n=== SUMMARY ===");
    println!("  Blocks:              {}", blocks.len());
    println!("  Connectivity:        {}", validated_matches.len());
    println!("  Periodic:            {}", periodic_faces.len());
    println!("  Periodic verified:   {}", verified.len());
    println!("  Periodic mismatched: {}", mismatched.len());
    println!("  Non-connected:       {}", remaining_outer.len());

    // ── Write tecplot.json ─────────────────────────────────────────────
    println!("\nWriting tecplot.json...");

    let connectivity_json: Vec<Value> = validated_matches.iter().map(face_match_to_json).collect();
    let periodic_json: Vec<Value> = periodic_faces.iter().map(face_match_to_json).collect();
    let non_connected_json: Vec<Value> = remaining_outer.iter().map(face_record_to_json).collect();

    let tecplot = json!({
        "connectivity_face_matches": connectivity_json,
        "periodic_faces": periodic_json,
        "non_connected_faces": non_connected_json,
    });

    let output_path = format!("{}tecplot.json", OUTPUT_DIR);
    let file = std::fs::File::create(&output_path).expect("Failed to create tecplot.json");
    serde_json::to_writer_pretty(file, &tecplot).expect("Failed to write tecplot.json");
    println!("  Written to: {}", output_path);

    // ── Print some stats about non-connected faces ──────────────────────
    let mut blocks_with_remaining: HashMap<usize, usize> = HashMap::new();
    for r in &remaining_outer {
        *blocks_with_remaining.entry(r.block_index).or_insert(0) += 1;
    }
    println!(
        "\n  {} blocks have remaining non-connected faces",
        blocks_with_remaining.len()
    );

    println!("\nDone.");
}
