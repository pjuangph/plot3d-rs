//! Full pipeline test: connectivity + rotational periodicity on the grid_packed mesh.
//!
//! Usage (release build required for large meshes):
//!   cargo test --release --test test_full_pipeline -- --nocapture

use plot3d::{
    connectivity_fast, create_face_from_diagonals, read_plot3d_binary, rotated_periodicity,
    to_radius, to_theta, BinaryFormat, Endian, Float, FloatPrecision,
};

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

const NBLADES: usize = 44;
const ROTATION_AXIS: char = 'x';

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
        true, // reduce_mesh
    );
    println!("  {} periodic face pairs", periodic_faces.len());
    println!("  {} remaining non-connected faces", remaining_outer.len());

    // ── Check known pair (Block 3741 / Block 4554) ─────────────────────
    println!("\n=== KNOWN PAIR CHECK (Block 3741 / Block 4554) ===");
    let mut found_known = false;
    for pp in &periodic_faces {
        let b1 = pp.block1.block_index;
        let b2 = pp.block2.block_index;
        if (b1 == 3741 && b2 == 4554) || (b1 == 4554 && b2 == 3741) {
            println!(
                "  FOUND: block1={} ({},{},{},{},{},{}) <-> block2={} ({},{},{},{},{},{})",
                b1, pp.block1.il, pp.block1.jl, pp.block1.kl,
                pp.block1.ih, pp.block1.jh, pp.block1.kh,
                b2, pp.block2.il, pp.block2.jl, pp.block2.kl,
                pp.block2.ih, pp.block2.jh, pp.block2.kh,
            );
            found_known = true;
        }
    }
    if !found_known {
        println!("  NOT FOUND in periodic results!");
        // Check if it's in remaining outer
        let in_remaining_a = remaining_outer.iter().any(|r| r.block_index == 3741);
        let in_remaining_b = remaining_outer.iter().any(|r| r.block_index == 4554);
        println!("  Block 3741 in remaining: {}", in_remaining_a);
        println!("  Block 4554 in remaining: {}", in_remaining_b);
    }

    // ── Summary ────────────────────────────────────────────────────────
    println!("\n=== SUMMARY ===");
    println!("  Blocks:              {}", blocks.len());
    println!("  Connectivity:        {}", face_matches.len());
    println!("  Periodic:            {}", periodic_faces.len());
    println!("  Non-connected:       {}", remaining_outer.len());

    // ── Analyze remaining non-connected faces ──────────────────────────
    println!("\n=== ANALYZING REMAINING NON-CONNECTED FACES ===");

    // Compute theta centroid for each remaining face
    struct FaceInfo {
        idx: usize,
        block_index: usize,
        theta_centroid: Float,
        radius_min: Float,
        radius_max: Float,
        axial_min: Float,
        axial_max: Float,
        const_type: i8,
        imin: usize,
        jmin: usize,
        kmin: usize,
        imax: usize,
        jmax: usize,
        kmax: usize,
    }

    let mut face_infos: Vec<FaceInfo> = Vec::with_capacity(remaining_outer.len());

    for (idx, rec) in remaining_outer.iter().enumerate() {
        let bi = rec.block_index;
        if bi >= blocks.len() {
            continue;
        }
        let block = &blocks[bi];
        // Validate indices
        if rec.i_hi() >= block.imax || rec.j_hi() >= block.jmax || rec.k_hi() >= block.kmax {
            continue;
        }
        let face = create_face_from_diagonals(
            block, rec.i_lo(), rec.j_lo(), rec.k_lo(), rec.i_hi(), rec.j_hi(), rec.k_hi(),
        );
        let verts = face.vertices();
        if verts.is_empty() {
            continue;
        }

        let thetas: Vec<Float> = verts
            .iter()
            .map(|v| to_theta(v[0], v[1], v[2], ROTATION_AXIS))
            .collect();
        let radii: Vec<Float> = verts
            .iter()
            .map(|v| to_radius(v[0], v[1], v[2], ROTATION_AXIS))
            .collect();
        let axials: Vec<Float> = verts.iter().map(|v| v[0]).collect(); // x-axis for 'x' rotation

        let theta_mean = thetas.iter().sum::<Float>() / thetas.len() as Float;
        let r_min = radii.iter().cloned().fold(Float::INFINITY, Float::min);
        let r_max = radii.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
        let ax_min = axials.iter().cloned().fold(Float::INFINITY, Float::min);
        let ax_max = axials.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

        face_infos.push(FaceInfo {
            idx,
            block_index: bi,
            theta_centroid: theta_mean,
            radius_min: r_min,
            radius_max: r_max,
            axial_min: ax_min,
            axial_max: ax_max,
            const_type: face.const_type(),
            imin: rec.il,
            jmin: rec.jl,
            kmin: rec.kl,
            imax: rec.ih,
            jmax: rec.jh,
            kmax: rec.kh,
        });
    }

    println!("  Analyzed {} faces", face_infos.len());

    // Count potential periodic pairs by theta proximity
    let theta_tol: Float = 0.05; // ~3 deg tolerance for centroid matching
    let mut potential_pairs = 0usize;
    let mut sample_pairs: Vec<(usize, usize)> = Vec::new();

    for i in 0..face_infos.len() {
        let fi = &face_infos[i];
        for j in (i + 1)..face_infos.len() {
            let fj = &face_infos[j];

            // Check theta separation ≈ rotation angle
            let theta_diff = (fi.theta_centroid - fj.theta_centroid).abs();
            if (theta_diff - rotation_angle_rad).abs() > theta_tol {
                continue;
            }

            // Check radial overlap
            if fi.radius_max < fj.radius_min || fj.radius_max < fi.radius_min {
                continue;
            }

            // Check axial overlap
            if fi.axial_max < fj.axial_min || fj.axial_max < fi.axial_min {
                continue;
            }

            // Both must be planar
            if fi.const_type == -1 || fj.const_type == -1 {
                continue;
            }

            potential_pairs += 1;
            if sample_pairs.len() < 20 {
                sample_pairs.push((i, j));
            }
        }
    }

    println!(
        "  Potential periodic pairs (theta ≈ angle, overlap): {}",
        potential_pairs
    );

    // Print sample pairs for investigation
    if !sample_pairs.is_empty() {
        println!("\n  Sample missed pairs (first 20):");
        for (i, j) in &sample_pairs {
            let fi = &face_infos[*i];
            let fj = &face_infos[*j];
            let theta_diff = (fi.theta_centroid - fj.theta_centroid).abs();
            println!(
                "    block {}({},{},{},{},{},{}) ct={} <-> block {}({},{},{},{},{},{}) ct={}  theta_diff={:.4}rad={:.2}deg",
                fi.block_index, fi.imin, fi.jmin, fi.kmin, fi.imax, fi.jmax, fi.kmax, fi.const_type,
                fj.block_index, fj.imin, fj.jmin, fj.kmin, fj.imax, fj.jmax, fj.kmax, fj.const_type,
                theta_diff, theta_diff.to_degrees()
            );
        }
    }

    // Count by const_type
    let mut ct_counts = [0usize; 4]; // 0=I, 1=J, 2=K, 3=none
    for fi in &face_infos {
        match fi.const_type {
            0 => ct_counts[0] += 1,
            1 => ct_counts[1] += 1,
            2 => ct_counts[2] += 1,
            _ => ct_counts[3] += 1,
        }
    }
    println!("\n  Remaining faces by const_type:");
    println!("    I-constant: {}", ct_counts[0]);
    println!("    J-constant: {}", ct_counts[1]);
    println!("    K-constant: {}", ct_counts[2]);
    println!("    Non-planar: {}", ct_counts[3]);

    println!();
}
