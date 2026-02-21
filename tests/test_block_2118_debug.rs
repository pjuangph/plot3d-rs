//! Diagnostic test for block 2118 connectivity investigation.
//!
//! Usage: cargo test --release --test test_block_2118_debug -- --nocapture

use plot3d::{
    connectivity_fast, create_face_from_diagonals, get_face_intersection, get_outer_faces,
    read_plot3d_binary, BinaryFormat, Endian, FaceMatch, FaceRecord, FloatPrecision,
};

const MESH_FILE: &str =
    "/Users/pjuangph/Library/CloudStorage/OneDrive-NASA/share/ali/grid_packed/grid_packed_binary.tmp.tmp.p3d";

const TARGET_BLOCK: usize = 2118;

#[test]
fn debug_block_2118() {
    println!("\n=== Block 2118 Diagnostic ===\n");

    // ── Load mesh ──
    println!("Loading mesh...");
    let blocks = read_plot3d_binary(MESH_FILE, BinaryFormat::Raw, FloatPrecision::F64, Endian::Little)
        .expect("read mesh");
    println!("  {} blocks loaded", blocks.len());

    let b = &blocks[TARGET_BLOCK];
    println!("\nBlock {} dimensions: {}x{}x{}", TARGET_BLOCK, b.imax, b.jmax, b.kmax);

    // Print corner coordinates
    println!("\nCorner coordinates:");
    for &i in &[0, b.imax - 1] {
        for &j in &[0, b.jmax - 1] {
            for &k in &[0, b.kmax - 1] {
                let (x, y, z) = b.xyz(i, j, k);
                println!("  ({},{},{}) -> ({:.6}, {:.6}, {:.6})", i, j, k, x, y, z);
            }
        }
    }

    // ── Get outer faces for block 2118 ──
    let (all_faces, _) = get_outer_faces(b);
    println!("\nAll 6 outer faces of block {}:", TARGET_BLOCK);
    for (idx, face) in all_faces.iter().enumerate() {
        println!(
            "  [{}] i={}..{}, j={}..{}, k={}..{}",
            idx, face.imin(), face.imax(), face.jmin(), face.jmax(), face.kmin(), face.kmax()
        );
    }

    // ── Run connectivity ──
    println!("\nRunning connectivity_fast...");
    let (face_matches, outer_faces) = connectivity_fast(&blocks);
    println!("  {} face matches, {} outer faces", face_matches.len(), outer_faces.len());

    // ── Filter matches involving block 2118 ──
    let my_matches: Vec<&FaceMatch> = face_matches
        .iter()
        .filter(|fm| fm.block1.block_index == TARGET_BLOCK || fm.block2.block_index == TARGET_BLOCK)
        .collect();
    println!("\nConnectivity matches involving block {}:", TARGET_BLOCK);
    for fm in &my_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;
        println!(
            "  block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]  ({} pts)",
            b1.block_index, b1.i_lo(), b1.j_lo(), b1.k_lo(), b1.i_hi(), b1.j_hi(), b1.k_hi(),
            b2.block_index, b2.i_lo(), b2.j_lo(), b2.k_lo(), b2.i_hi(), b2.j_hi(), b2.k_hi(),
            fm.points.len()
        );
    }

    // ── Filter remaining outer faces for block 2118 ──
    let my_outer: Vec<&FaceRecord> = outer_faces
        .iter()
        .filter(|f| f.block_index == TARGET_BLOCK)
        .collect();
    println!("\nRemaining outer faces for block {}:", TARGET_BLOCK);
    for f in &my_outer {
        println!(
            "  [{},{},{}->{},{},{}]",
            f.i_lo(), f.j_lo(), f.k_lo(), f.i_hi(), f.j_hi(), f.k_hi()
        );
        // Print corner coordinates of this face
        let (x1, y1, z1) = blocks[TARGET_BLOCK].xyz(f.i_lo(), f.j_lo(), f.k_lo());
        let (x2, y2, z2) = blocks[TARGET_BLOCK].xyz(f.i_hi(), f.j_hi(), f.k_hi());
        println!("    corner1: ({:.6}, {:.6}, {:.6})", x1, y1, z1);
        println!("    corner2: ({:.6}, {:.6}, {:.6})", x2, y2, z2);
    }

    // ── Find blocks with overlapping AABBs ──
    println!("\nSearching for AABB-overlapping blocks...");
    let target = &blocks[TARGET_BLOCK];
    let (txn, txx, tyn, tyx, tzn, tzx) = block_aabb(target);
    println!("  Block {} AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]",
        TARGET_BLOCK, txn, txx, tyn, tyx, tzn, tzx);

    let tol = 1e-4;
    let mut overlapping: Vec<usize> = Vec::new();
    for (bi, blk) in blocks.iter().enumerate() {
        if bi == TARGET_BLOCK {
            continue;
        }
        let (xn, xx, yn, yx, zn, zx) = block_aabb(blk);
        if txx + tol >= xn && xx + tol >= txn
            && tyx + tol >= yn && yx + tol >= tyn
            && tzx + tol >= zn && zx + tol >= tzn
        {
            overlapping.push(bi);
        }
    }
    println!("  {} AABB-overlapping blocks", overlapping.len());

    // ── For each remaining outer face, try direct face matching against overlapping blocks ──
    println!("\nDirect face intersection test for each remaining outer face:");
    for f in &my_outer {
        println!(
            "\n  Face [{},{},{}->{},{},{}]:",
            f.i_lo(), f.j_lo(), f.k_lo(), f.i_hi(), f.j_hi(), f.k_hi()
        );

        // Build a Face from the FaceRecord
        let mut target_face = create_face_from_diagonals(
            &blocks[TARGET_BLOCK],
            f.i_lo(), f.j_lo(), f.k_lo(),
            f.i_hi(), f.j_hi(), f.k_hi(),
        );
        target_face.set_block_index(TARGET_BLOCK);

        let mut found_any = false;
        for &bj in &overlapping {
            let (other_faces, _) = get_outer_faces(&blocks[bj]);
            for of in &other_faces {
                // Quick AABB pre-check
                let (fx1, fy1, fz1) = blocks[TARGET_BLOCK].xyz(f.i_lo(), f.j_lo(), f.k_lo());
                let (fx2, fy2, fz2) = blocks[TARGET_BLOCK].xyz(f.i_hi(), f.j_hi(), f.k_hi());
                let (gx1, gy1, gz1) = blocks[bj].xyz(of.imin(), of.jmin(), of.kmin());
                let (gx2, gy2, gz2) = blocks[bj].xyz(of.imax(), of.jmax(), of.kmax());

                let face_tol = 0.1;
                let fxn = fx1.min(fx2);
                let fxx_v = fx1.max(fx2);
                let fyn = fy1.min(fy2);
                let fyx_v = fy1.max(fy2);
                let fzn = fz1.min(fz2);
                let fzx_v = fz1.max(fz2);
                let gxn = gx1.min(gx2);
                let gxx = gx1.max(gx2);
                let gyn = gy1.min(gy2);
                let gyx = gy1.max(gy2);
                let gzn = gz1.min(gz2);
                let gzx = gz1.max(gz2);

                if fxx_v + face_tol < gxn || gxx + face_tol < fxn
                    || fyx_v + face_tol < gyn || gyx + face_tol < fyn
                    || fzx_v + face_tol < gzn || gzx + face_tol < fzn
                {
                    continue;
                }

                let (pts, _, _) = get_face_intersection(
                    &target_face, of, &blocks[TARGET_BLOCK], &blocks[bj], 1e-6,
                );
                if !pts.is_empty() {
                    println!(
                        "    MATCH with block {} face [{},{},{}->{},{},{}]: {} matched points",
                        bj, of.imin(), of.imax(), of.jmin(), of.jmax(), of.kmin(), of.kmax(),
                        pts.len()
                    );
                    // Show first few points
                    for (pi, pt) in pts.iter().take(3).enumerate() {
                        let (x1, y1, z1) = blocks[TARGET_BLOCK].xyz(pt.i1, pt.j1, pt.k1);
                        let (x2, y2, z2) = blocks[bj].xyz(pt.i2, pt.j2, pt.k2);
                        let dist = ((x1-x2).powi(2) + (y1-y2).powi(2) + (z1-z2).powi(2)).sqrt();
                        println!(
                            "      pt[{}]: ({},{},{})=({:.6},{:.6},{:.6}) <-> ({},{},{})=({:.6},{:.6},{:.6}) dist={:.2e}",
                            pi, pt.i1, pt.j1, pt.k1, x1, y1, z1,
                            pt.i2, pt.j2, pt.k2, x2, y2, z2, dist
                        );
                    }
                    found_any = true;
                }
            }
        }
        if !found_any {
            println!("    No matching faces found — likely a genuine boundary face");

            // Check if this is a periodic candidate
            let (fx1, fy1, fz1) = blocks[TARGET_BLOCK].xyz(f.i_lo(), f.j_lo(), f.k_lo());
            let (fx2, fy2, fz2) = blocks[TARGET_BLOCK].xyz(f.i_hi(), f.j_hi(), f.k_hi());
            let theta1 = fy1.atan2(fz1);
            let theta2 = fy2.atan2(fz2);
            let r1 = (fy1*fy1 + fz1*fz1).sqrt();
            let r2 = (fy2*fy2 + fz2*fz2).sqrt();
            println!("    theta range: {:.4} to {:.4} rad ({:.2} to {:.2} deg)",
                theta1, theta2, theta1.to_degrees(), theta2.to_degrees());
            println!("    radius range: {:.6} to {:.6}", r1, r2);
            println!("    x range: {:.6} to {:.6}", fx1, fx2);
        }
    }

    // ── Check if block 2118 appears in periodicity results ──
    println!("\n\n=== Periodicity check for block {} ===", TARGET_BLOCK);
    let rotation_angle_deg: f64 = 360.0 / 44.0;
    let (periodic_faces, remaining) = plot3d::rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle_deg,
        'x',
        true,
    );

    let my_periodic: Vec<&FaceMatch> = periodic_faces
        .iter()
        .filter(|fm| fm.block1.block_index == TARGET_BLOCK || fm.block2.block_index == TARGET_BLOCK)
        .collect();
    println!("\nPeriodic matches involving block {}:", TARGET_BLOCK);
    for fm in &my_periodic {
        let b1 = &fm.block1;
        let b2 = &fm.block2;
        println!(
            "  block{}[{},{},{}->{},{},{}] <-> block{}[{},{},{}->{},{},{}]",
            b1.block_index, b1.i_lo(), b1.j_lo(), b1.k_lo(), b1.i_hi(), b1.j_hi(), b1.k_hi(),
            b2.block_index, b2.i_lo(), b2.j_lo(), b2.k_lo(), b2.i_hi(), b2.j_hi(), b2.k_hi(),
        );
    }

    let my_remaining: Vec<&FaceRecord> = remaining
        .iter()
        .filter(|f| f.block_index == TARGET_BLOCK)
        .collect();
    println!("\nRemaining non-connected faces for block {} after periodicity:", TARGET_BLOCK);
    for f in &my_remaining {
        let (x1, y1, z1) = blocks[TARGET_BLOCK].xyz(f.i_lo(), f.j_lo(), f.k_lo());
        let (x2, y2, z2) = blocks[TARGET_BLOCK].xyz(f.i_hi(), f.j_hi(), f.k_hi());
        println!(
            "  [{},{},{}->{},{},{}]",
            f.i_lo(), f.j_lo(), f.k_lo(), f.i_hi(), f.j_hi(), f.k_hi()
        );
        println!("    corner1: ({:.6}, {:.6}, {:.6})", x1, y1, z1);
        println!("    corner2: ({:.6}, {:.6}, {:.6})", x2, y2, z2);
        let theta1 = fy_to_theta(y1, z1);
        let theta2 = fy_to_theta(y2, z2);
        println!("    theta: {:.4} to {:.4} rad ({:.2} to {:.2} deg)",
            theta1, theta2, theta1.to_degrees(), theta2.to_degrees());
    }

    // ── Deep corner search: check 4 corners of K=24 face against all blocks ──
    println!("\n=== Deep corner search on K=24 face ===");
    let kface = 24usize;
    let b2118 = &blocks[TARGET_BLOCK];
    let corners = [
        (0usize, 0usize, kface),
        (0, b2118.jmax - 1, kface),
        (b2118.imax - 1, 0, kface),
        (b2118.imax - 1, b2118.jmax - 1, kface),
    ];
    let corner_coords: Vec<[f64; 3]> = corners.iter().map(|&(i, j, k)| {
        let (x, y, z) = b2118.xyz(i, j, k);
        [x, y, z]
    }).collect();

    println!("  4 corner coordinates:");
    for (idx, ((i, j, k), coord)) in corners.iter().zip(corner_coords.iter()).enumerate() {
        println!("    [{}] ({},{},{}) -> ({:.6}, {:.6}, {:.6})", idx, i, j, k, coord[0], coord[1], coord[2]);
    }

    let search_tol = 1e-4;
    let mut matching_blocks: std::collections::HashMap<usize, Vec<String>> = std::collections::HashMap::new();
    for (ci, coord) in corner_coords.iter().enumerate() {
        for (bi, blk) in blocks.iter().enumerate() {
            if bi == TARGET_BLOCK {
                continue;
            }
            // Quick AABB check
            let (xn, xx, yn, yx, zn, zx) = block_aabb(blk);
            if coord[0] < xn - search_tol || coord[0] > xx + search_tol
                || coord[1] < yn - search_tol || coord[1] > yx + search_tol
                || coord[2] < zn - search_tol || coord[2] > zx + search_tol
            {
                continue;
            }
            // Check only boundary face nodes (6 faces)
            let boundary_indices: Vec<(usize, usize, usize)> = {
                let mut v = Vec::new();
                // I=0 and I=imax-1 faces
                for &iface in &[0, blk.imax - 1] {
                    for j in 0..blk.jmax {
                        for k in 0..blk.kmax {
                            v.push((iface, j, k));
                        }
                    }
                }
                // J=0 and J=jmax-1 faces (avoid duplicating corners)
                for &jface in &[0, blk.jmax - 1] {
                    for i in 1..blk.imax-1 {
                        for k in 0..blk.kmax {
                            v.push((i, jface, k));
                        }
                    }
                }
                // K=0 and K=kmax-1 faces (avoid duplicating edges)
                for &kf in &[0, blk.kmax - 1] {
                    for i in 1..blk.imax-1 {
                        for j in 1..blk.jmax-1 {
                            v.push((i, j, kf));
                        }
                    }
                }
                v
            };
            for &(i, j, k) in &boundary_indices {
                let (bx, by, bz) = blk.xyz(i, j, k);
                let dist = ((bx - coord[0]).powi(2) + (by - coord[1]).powi(2) + (bz - coord[2]).powi(2)).sqrt();
                if dist < search_tol {
                    matching_blocks.entry(bi).or_default().push(
                        format!("corner[{}] matched ({},{},{}) dist={:.2e}", ci, i, j, k, dist)
                    );
                    break; // one match per block per corner is enough
                }
            }
        }
    }

    if matching_blocks.is_empty() {
        println!("  NO blocks share any boundary nodes with K=24 corners - this is a genuine boundary");
    } else {
        println!("  {} blocks share boundary nodes with K=24 corners:", matching_blocks.len());
        for (bi, msgs) in &matching_blocks {
            println!("    Block {} (dims {}x{}x{}):", bi, blocks[*bi].imax, blocks[*bi].jmax, blocks[*bi].kmax);
            for msg in msgs {
                println!("      {}", msg);
            }
        }
    }

    // ── Also check rotated corners (periodic candidate) ──
    println!("\n=== Rotated corner search (periodic candidate) ===");
    let rot_angle = (360.0_f64 / 44.0).to_radians();
    for direction in &["forward", "backward"] {
        let angle = if *direction == "forward" { rot_angle } else { -rot_angle };
        println!("  Rotation {} ({:.4} deg):", direction, angle.to_degrees());
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for (ci, coord) in corner_coords.iter().enumerate() {
            // Rotate around x-axis: y' = y*cos - z*sin, z' = y*sin + z*cos
            let rotated = [coord[0], coord[1] * cos_a - coord[2] * sin_a, coord[1] * sin_a + coord[2] * cos_a];

            let mut found = false;
            for (bi, blk) in blocks.iter().enumerate() {
                if bi == TARGET_BLOCK { continue; }
                let (xn, xx, yn, yx, zn, zx) = block_aabb(blk);
                if rotated[0] < xn - search_tol || rotated[0] > xx + search_tol
                    || rotated[1] < yn - search_tol || rotated[1] > yx + search_tol
                    || rotated[2] < zn - search_tol || rotated[2] > zx + search_tol
                { continue; }
                // Check boundary nodes
                for &iface in &[0usize, blk.imax - 1] {
                    for j in 0..blk.jmax {
                        for k in 0..blk.kmax {
                            let (bx, by, bz) = blk.xyz(iface, j, k);
                            let dist = ((bx - rotated[0]).powi(2) + (by - rotated[1]).powi(2) + (bz - rotated[2]).powi(2)).sqrt();
                            if dist < search_tol {
                                println!("    corner[{}] rotated({:.6},{:.6},{:.6}) matches block {} ({},{},{}) dist={:.2e}",
                                    ci, rotated[0], rotated[1], rotated[2], bi, iface, j, k, dist);
                                found = true;
                                break;
                            }
                        }
                        if found { break; }
                    }
                    if found { break; }
                }
                if !found {
                    for &jface in &[0usize, blk.jmax - 1] {
                        for i in 0..blk.imax {
                            for k in 0..blk.kmax {
                                let (bx, by, bz) = blk.xyz(i, jface, k);
                                let dist = ((bx - rotated[0]).powi(2) + (by - rotated[1]).powi(2) + (bz - rotated[2]).powi(2)).sqrt();
                                if dist < search_tol {
                                    println!("    corner[{}] rotated({:.6},{:.6},{:.6}) matches block {} ({},{},{}) dist={:.2e}",
                                        ci, rotated[0], rotated[1], rotated[2], bi, i, jface, k, dist);
                                    found = true;
                                    break;
                                }
                            }
                            if found { break; }
                        }
                        if found { break; }
                    }
                }
                if !found {
                    for &kf in &[0usize, blk.kmax - 1] {
                        for i in 0..blk.imax {
                            for j in 0..blk.jmax {
                                let (bx, by, bz) = blk.xyz(i, j, kf);
                                let dist = ((bx - rotated[0]).powi(2) + (by - rotated[1]).powi(2) + (bz - rotated[2]).powi(2)).sqrt();
                                if dist < search_tol {
                                    println!("    corner[{}] rotated({:.6},{:.6},{:.6}) matches block {} ({},{},{}) dist={:.2e}",
                                        ci, rotated[0], rotated[1], rotated[2], bi, i, j, kf, dist);
                                    found = true;
                                    break;
                                }
                            }
                            if found { break; }
                        }
                        if found { break; }
                    }
                }
                if found { break; }
            }
            if !found {
                println!("    corner[{}] rotated({:.6},{:.6},{:.6}) - no match found",
                    ci, rotated[0], rotated[1], rotated[2]);
            }
        }
    }

    // ── Targeted test: block 2386 K=0 face vs block 2118 K=24 face ──
    println!("\n\n=== Targeted test: missing GridPro matches ===");

    // Match 1: block2386[0,216,0->24,360,0] <-> block2118[0,0,24->144,24,24]
    println!("\n--- block2386 K=0 vs block2118 K=24 (j=0..24 half) ---");
    let b2386 = &blocks[2386];
    println!("  Block 2386 dims: {}x{}x{}", b2386.imax, b2386.jmax, b2386.kmax);

    // Check if the full K=0 face of block 2386 appears in get_outer_faces
    let (faces_2386, _) = get_outer_faces(b2386);
    println!("  Block 2386 has {} outer faces:", faces_2386.len());
    for (fi, face) in faces_2386.iter().enumerate() {
        println!("    [{}] i={}..{}, j={}..{}, k={}..{}", fi,
            face.imin(), face.imax(), face.jmin(), face.jmax(), face.kmin(), face.kmax());
    }

    // Find the K=0 face
    let k0_face = faces_2386.iter().find(|f| f.kmin() == 0 && f.kmax() == 0);
    if let Some(k0) = k0_face {
        println!("\n  Testing K=0 face of block 2386 (i={}..{}, j={}..{}) vs K=24 face of block 2118:",
            k0.imin(), k0.imax(), k0.jmin(), k0.jmax());

        // Build the target face (block 2118 K=24)
        let mut target_k24 = create_face_from_diagonals(
            &blocks[TARGET_BLOCK], 0, 0, 24, 144, 48, 24);
        target_k24.set_block_index(TARGET_BLOCK);

        let (pts, _, _) = get_face_intersection(
            &target_k24, k0, &blocks[TARGET_BLOCK], b2386, 1e-6);
        println!("  get_face_intersection result: {} matched points", pts.len());
        for pt in pts.iter().take(5) {
            let (x1, y1, z1) = blocks[TARGET_BLOCK].xyz(pt.i1, pt.j1, pt.k1);
            let (x2, y2, z2) = b2386.xyz(pt.i2, pt.j2, pt.k2);
            let dist = ((x1-x2).powi(2) + (y1-y2).powi(2) + (z1-z2).powi(2)).sqrt();
            println!("    ({},{},{}) <-> ({},{},{})  dist={:.2e}", pt.i1, pt.j1, pt.k1, pt.i2, pt.j2, pt.k2, dist);
        }
    }

    // Match 2: block2388[216,0,0->360,0,24] <-> block2118[0,24,24->144,48,24]
    println!("\n--- block2388 J=0 vs block2118 K=24 (j=24..48 half) ---");
    let b2388 = &blocks[2388];
    println!("  Block 2388 dims: {}x{}x{}", b2388.imax, b2388.jmax, b2388.kmax);

    let (faces_2388, _) = get_outer_faces(b2388);
    println!("  Block 2388 has {} outer faces:", faces_2388.len());
    for (fi, face) in faces_2388.iter().enumerate() {
        println!("    [{}] i={}..{}, j={}..{}, k={}..{}", fi,
            face.imin(), face.imax(), face.jmin(), face.jmax(), face.kmin(), face.kmax());
    }

    // Find the J=0 face
    let j0_face = faces_2388.iter().find(|f| f.jmin() == 0 && f.jmax() == 0);
    if let Some(j0) = j0_face {
        println!("\n  Testing J=0 face of block 2388 (i={}..{}, k={}..{}) vs K=24 face of block 2118:",
            j0.imin(), j0.imax(), j0.kmin(), j0.kmax());

        let mut target_k24 = create_face_from_diagonals(
            &blocks[TARGET_BLOCK], 0, 0, 24, 144, 48, 24);
        target_k24.set_block_index(TARGET_BLOCK);

        let (pts, _, _) = get_face_intersection(
            &target_k24, j0, &blocks[TARGET_BLOCK], b2388, 1e-6);
        println!("  get_face_intersection result: {} matched points", pts.len());
        for pt in pts.iter().take(5) {
            let (x1, y1, z1) = blocks[TARGET_BLOCK].xyz(pt.i1, pt.j1, pt.k1);
            let (x2, y2, z2) = b2388.xyz(pt.i2, pt.j2, pt.k2);
            let dist = ((x1-x2).powi(2) + (y1-y2).powi(2) + (z1-z2).powi(2)).sqrt();
            println!("    ({},{},{}) <-> ({},{},{})  dist={:.2e}", pt.i1, pt.j1, pt.k1, pt.i2, pt.j2, pt.k2, dist);
        }
    }

    // Check if blocks 2386 and 2388 are in the AABB-overlapping list
    println!("\n  Block 2386 in overlapping list: {}", overlapping.contains(&2386));
    println!("  Block 2388 in overlapping list: {}", overlapping.contains(&2388));

    // Check AABB of blocks 2386 and 2388
    let (xn, xx, yn, yx, zn, zx) = block_aabb(b2386);
    println!("  Block 2386 AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]", xn, xx, yn, yx, zn, zx);
    let (xn, xx, yn, yx, zn, zx) = block_aabb(b2388);
    println!("  Block 2388 AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]", xn, xx, yn, yx, zn, zx);

    // ── Replicate Phase 3 with reduced blocks to find the breakdown ──
    println!("\n\n=== Simulating Phase 3 with REDUCED blocks ===");
    let gcd = plot3d::utils::compute_min_gcd(&blocks);
    println!("  GCD = {}", gcd);
    let reduced = plot3d::reduce_blocks(&blocks, gcd);
    println!("  Block 2118 reduced: {}x{}x{}", reduced[TARGET_BLOCK].imax, reduced[TARGET_BLOCK].jmax, reduced[TARGET_BLOCK].kmax);
    println!("  Block 2386 reduced: {}x{}x{}", reduced[2386].imax, reduced[2386].jmax, reduced[2386].kmax);
    println!("  Block 2388 reduced: {}x{}x{}", reduced[2388].imax, reduced[2388].jmax, reduced[2388].kmax);

    // Check AABB overlap between reduced blocks 2118 and 2386/2388
    let (xn1, xx1, yn1, yx1, zn1, zx1) = block_aabb(&reduced[TARGET_BLOCK]);
    let (xn2, xx2, yn2, yx2, zn2, zx2) = block_aabb(&reduced[2386]);
    let (xn3, xx3, yn3, yx3, zn3, zx3) = block_aabb(&reduced[2388]);
    println!("  Reduced block 2118 AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]", xn1, xx1, yn1, yx1, zn1, zx1);
    println!("  Reduced block 2386 AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]", xn2, xx2, yn2, yx2, zn2, zx2);
    println!("  Reduced block 2388 AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]", xn3, xx3, yn3, yx3, zn3, zx3);

    let tol = 1e-6;
    let overlap_2386 = xx1 + tol >= xn2 && xx2 + tol >= xn1
        && yx1 + tol >= yn2 && yx2 + tol >= yn1
        && zx1 + tol >= zn2 && zx2 + tol >= zn1;
    let overlap_2388 = xx1 + tol >= xn3 && xx3 + tol >= xn1
        && yx1 + tol >= yn3 && yx3 + tol >= yn1
        && zx1 + tol >= zn3 && zx3 + tol >= zn1;
    println!("  Reduced AABB overlap 2118-2386: {}", overlap_2386);
    println!("  Reduced AABB overlap 2118-2388: {}", overlap_2388);

    // Get fresh outer faces for reduced blocks
    let (fresh_2386, _) = get_outer_faces(&reduced[2386]);
    let (fresh_2388, _) = get_outer_faces(&reduced[2388]);

    // Build the K=max face for reduced block 2118
    let k_max_reduced = reduced[TARGET_BLOCK].kmax - 1;
    let mut target_face_reduced = create_face_from_diagonals(
        &reduced[TARGET_BLOCK],
        0, 0, k_max_reduced,
        reduced[TARGET_BLOCK].imax - 1, reduced[TARGET_BLOCK].jmax - 1, k_max_reduced,
    );
    target_face_reduced.set_block_index(TARGET_BLOCK);
    println!("  Target face (reduced): i={}..{}, j={}..{}, k={}",
        target_face_reduced.imin(), target_face_reduced.imax(),
        target_face_reduced.jmin(), target_face_reduced.jmax(), k_max_reduced);

    // Face-level AABB for target face
    let (fx1, fy1, fz1) = reduced[TARGET_BLOCK].xyz(
        target_face_reduced.imin(), target_face_reduced.jmin(), target_face_reduced.kmin());
    let (fx2, fy2, fz2) = reduced[TARGET_BLOCK].xyz(
        target_face_reduced.imax(), target_face_reduced.jmax(), target_face_reduced.kmax());
    let (fxn, fxx) = (fx1.min(fx2), fx1.max(fx2));
    let (fyn, fyx) = (fy1.min(fy2), fy1.max(fy2));
    let (fzn, fzx) = (fz1.min(fz2), fz1.max(fz2));
    println!("  Target face AABB: x=[{:.6},{:.6}], y=[{:.6},{:.6}], z=[{:.6},{:.6}]",
        fxn, fxx, fyn, fyx, fzn, fzx);

    // Check each face of block 2386
    println!("\n  Testing block 2386 faces:");
    for (fi, ff) in fresh_2386.iter().enumerate() {
        let (gx1, gy1, gz1) = reduced[2386].xyz(ff.imin(), ff.jmin(), ff.kmin());
        let (gx2, gy2, gz2) = reduced[2386].xyz(ff.imax(), ff.jmax(), ff.kmax());
        let tol_pre = 0.1;
        let reject = fxx + tol_pre < gx1.min(gx2)
            || gx1.max(gx2) + tol_pre < fxn
            || fyx + tol_pre < gy1.min(gy2)
            || gy1.max(gy2) + tol_pre < fyn
            || fzx + tol_pre < gz1.min(gz2)
            || gz1.max(gz2) + tol_pre < fzn;
        println!("    [{}] i={}..{}, j={}..{}, k={}..{} | corners ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4}) | AABB reject: {}",
            fi, ff.imin(), ff.imax(), ff.jmin(), ff.jmax(), ff.kmin(), ff.kmax(),
            gx1, gy1, gz1, gx2, gy2, gz2, reject);
        if !reject {
            let (pts, _, _) = get_face_intersection(
                &target_face_reduced, ff, &reduced[TARGET_BLOCK], &reduced[2386], 1e-6);
            println!("      -> get_face_intersection: {} points", pts.len());
        }
    }

    // Check each face of block 2388
    println!("\n  Testing block 2388 faces:");
    for (fi, ff) in fresh_2388.iter().enumerate() {
        let (gx1, gy1, gz1) = reduced[2388].xyz(ff.imin(), ff.jmin(), ff.kmin());
        let (gx2, gy2, gz2) = reduced[2388].xyz(ff.imax(), ff.jmax(), ff.kmax());
        let tol_pre = 0.1;
        let reject = fxx + tol_pre < gx1.min(gx2)
            || gx1.max(gx2) + tol_pre < fxn
            || fyx + tol_pre < gy1.min(gy2)
            || gy1.max(gy2) + tol_pre < fyn
            || fzx + tol_pre < gz1.min(gz2)
            || gz1.max(gz2) + tol_pre < fzn;
        println!("    [{}] i={}..{}, j={}..{}, k={}..{} | corners ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4}) | AABB reject: {}",
            fi, ff.imin(), ff.imax(), ff.jmin(), ff.jmax(), ff.kmin(), ff.kmax(),
            gx1, gy1, gz1, gx2, gy2, gz2, reject);
        if !reject {
            let (pts, _, _) = get_face_intersection(
                &target_face_reduced, ff, &reduced[TARGET_BLOCK], &reduced[2388], 1e-6);
            println!("      -> get_face_intersection: {} points", pts.len());
        }
    }

    println!("\nDone.");
}

fn block_aabb(b: &plot3d::Block) -> (f64, f64, f64, f64, f64, f64) {
    let mut xn = f64::INFINITY;
    let mut xx = f64::NEG_INFINITY;
    let mut yn = f64::INFINITY;
    let mut yx = f64::NEG_INFINITY;
    let mut zn = f64::INFINITY;
    let mut zx = f64::NEG_INFINITY;
    for &v in &b.x { xn = xn.min(v); xx = xx.max(v); }
    for &v in &b.y { yn = yn.min(v); yx = yx.max(v); }
    for &v in &b.z { zn = zn.min(v); zx = zx.max(v); }
    (xn, xx, yn, yx, zn, zx)
}

fn fy_to_theta(y: f64, z: f64) -> f64 {
    z.atan2(y)
}
