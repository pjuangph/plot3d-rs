//! End-to-end test: read VSPT binary mesh, compute connectivity + rotational
//! periodicity, serialize to connectivity.json, then validate geometry.
//!
//! Mirrors the Python `create_connectivity_json.py` + `test_connectivity.py`
//! workflow from the Plot3D_utilities repository.

use plot3d::{
    connectivity_fast, create_rotation_matrix, face_match_to_json, face_matches_to_dict,
    read_plot3d_binary, rotated_periodicity, verify_connectivity, verify_periodicity,
    align_face_orientations, permutation_matrices_json,
    BinaryFormat, Endian, Float, FloatPrecision,
};
use serde_json::{json, Value};

const MESH_PATH: &str = "tests/data/vspt_mesh_scaled.xyz";
const NBLADES: usize = 55;
const ROTATION_AXIS: char = 'x';
const TOL: Float = 1e-4;

/// Build the full connectivity.json payload as a serde_json::Value.
fn build_connectivity_json() -> (Vec<plot3d::Block>, Value) {
    let blocks =
        read_plot3d_binary(MESH_PATH, BinaryFormat::Raw, FloatPrecision::F32, Endian::Little)
            .expect("read mesh failed");

    let rotation_angle_deg = 360.0 / NBLADES as Float;
    let rotation_angle_rad = rotation_angle_deg.to_radians();

    // Connectivity
    let (face_matches, outer_faces) = connectivity_fast(&blocks);

    // Directed-diagonal + orientation
    let face_matches = face_matches_to_dict(&blocks, &face_matches);
    let (face_matches, _unverified) = verify_connectivity(&blocks, &face_matches, TOL);
    let (face_matches, _unaligned) = align_face_orientations(&blocks, &face_matches, TOL);

    // Rotated periodicity
    let (periodic_pairs, outer_faces_remaining) = rotated_periodicity(
        &blocks,
        &face_matches,
        &outer_faces,
        rotation_angle_deg,
        ROTATION_AXIS,
        true,
    );

    let (periodic_pairs, _unverified) =
        verify_periodicity(&blocks, &periodic_pairs, rotation_angle_rad, ROTATION_AXIS, TOL);

    // Rotation matrix for JSON
    let rot_mat = create_rotation_matrix(rotation_angle_rad, ROTATION_AXIS);
    let rot_mat_json: Vec<Vec<Float>> = rot_mat.iter().map(|row| row.to_vec()).collect();

    // Serialize
    let fm_json: Vec<Value> = face_matches.iter().map(|m| face_match_to_json(m)).collect();
    let pf_json: Vec<Value> = periodic_pairs.iter().map(|m| face_match_to_json(m)).collect();
    let of_json: Vec<Value> = outer_faces_remaining
        .iter()
        .map(|f| {
            json!({
                "lb": [f.il, f.jl, f.kl],
                "ub": [f.ih, f.jh, f.kh],
                "block_index": f.block_index,
            })
        })
        .collect();

    let payload = json!({
        "mesh_file": MESH_PATH,
        "nblocks": blocks.len(),
        "face_matches": fm_json,
        "outer_faces": of_json,
        "periodic_faces": pf_json,
        "periodicity": {
            "nblades": NBLADES,
            "rotation_axis": ROTATION_AXIS.to_string(),
            "rotation_angle_rad": rotation_angle_rad,
            "rotation_angle_deg": rotation_angle_deg,
            "transformation_matrix": rot_mat_json,
            "convention": "Face_B_points = (transformation_matrix @ Face_A_points.T).T",
            "source": "plot3d-rs::rotated_periodicity",
        },
        "permutation_matrices": permutation_matrices_json(),
    });

    (blocks, payload)
}

/// Extract all xyz points on a face as Vec<[Float; 3]>.
/// Handles reversed lb/ub ordering.
fn extract_face_points(block: &plot3d::Block, lb: &[usize; 3], ub: &[usize; 3]) -> Vec<[Float; 3]> {
    let imin = lb[0].min(ub[0]);
    let imax = lb[0].max(ub[0]);
    let jmin = lb[1].min(ub[1]);
    let jmax = lb[1].max(ub[1]);
    let kmin = lb[2].min(ub[2]);
    let kmax = lb[2].max(ub[2]);

    let mut pts = Vec::new();
    for i in imin..=imax {
        for j in jmin..=jmax {
            for k in kmin..=kmax {
                let (x, y, z) = block.xyz(i, j, k);
                pts.push([x, y, z]);
            }
        }
    }
    pts
}

/// Parse [i,j,k] from JSON array.
fn parse_ijk(v: &Value) -> [usize; 3] {
    let arr = v.as_array().unwrap();
    [
        arr[0].as_u64().unwrap() as usize,
        arr[1].as_u64().unwrap() as usize,
        arr[2].as_u64().unwrap() as usize,
    ]
}

/// Find nearest point distance from each pt in `query` to `target`.
fn max_nearest_dist(query: &[[Float; 3]], target: &[[Float; 3]]) -> Float {
    query
        .iter()
        .map(|q| {
            target
                .iter()
                .map(|t| {
                    let dx = q[0] - t[0];
                    let dy = q[1] - t[1];
                    let dz = q[2] - t[2];
                    dx * dx + dy * dy + dz * dz
                })
                .fold(Float::INFINITY, Float::min)
                .sqrt()
        })
        .fold(0.0 as Float, Float::max)
}

// ── Tests ──

/// `MESH_PATH` is a local fixture, not committed (see .gitignore's `*.xyz`),
/// so CI and fresh checkouts must skip rather than panic when it's absent.
macro_rules! require_mesh {
    () => {
        if !std::path::Path::new(MESH_PATH).exists() {
            eprintln!("{MESH_PATH} not found, skipping test.");
            return;
        }
    };
}

#[test]
fn test_creates_connectivity_json() {
    require_mesh!();
    let (blocks, payload) = build_connectivity_json();

    assert_eq!(payload["nblocks"].as_u64().unwrap() as usize, blocks.len());
    assert!(payload["face_matches"].as_array().unwrap().len() > 0);
    assert!(payload["periodic_faces"].as_array().unwrap().len() > 0);

    // Write to file for inspection
    let json_str = serde_json::to_string_pretty(&payload).unwrap();
    std::fs::write("vspt_connectivity.json", &json_str).unwrap();
    println!("Wrote vspt_connectivity.json ({} bytes)", json_str.len());
}

#[test]
fn test_periodic_faces_geometry() {
    require_mesh!();
    let (blocks, payload) = build_connectivity_json();

    let rot_mat_json = payload["periodicity"]["transformation_matrix"]
        .as_array()
        .unwrap();
    let rot_mat: [[Float; 3]; 3] = {
        let mut m = [[0.0; 3]; 3];
        for (i, row) in rot_mat_json.iter().enumerate() {
            for (j, val) in row.as_array().unwrap().iter().enumerate() {
                m[i][j] = val.as_f64().unwrap() as Float;
            }
        }
        m
    };

    for (idx, pf) in payload["periodic_faces"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let b1_idx = pf["block1"]["block_index"].as_u64().unwrap() as usize;
        let b2_idx = pf["block2"]["block_index"].as_u64().unwrap() as usize;
        let lb1 = parse_ijk(&pf["block1"]["lb"]);
        let ub1 = parse_ijk(&pf["block1"]["ub"]);
        let lb2 = parse_ijk(&pf["block2"]["lb"]);
        let ub2 = parse_ijk(&pf["block2"]["ub"]);

        let pts1 = extract_face_points(&blocks[b1_idx], &lb1, &ub1);
        let pts2 = extract_face_points(&blocks[b2_idx], &lb2, &ub2);

        // Try both forward and backward rotation (block1/block2 ordering may vary)
        let apply_rot = |p: &[Float; 3], m: &[[Float; 3]; 3]| -> [Float; 3] {
            [
                m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
                m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
            ]
        };
        // Inverse rotation = transpose (rotation matrices are orthogonal)
        let rot_inv: [[Float; 3]; 3] = [
            [rot_mat[0][0], rot_mat[1][0], rot_mat[2][0]],
            [rot_mat[0][1], rot_mat[1][1], rot_mat[2][1]],
            [rot_mat[0][2], rot_mat[1][2], rot_mat[2][2]],
        ];

        let pts1_fwd: Vec<_> = pts1.iter().map(|p| apply_rot(p, &rot_mat)).collect();
        let pts1_rev: Vec<_> = pts1.iter().map(|p| apply_rot(p, &rot_inv)).collect();

        let dist_fwd = max_nearest_dist(&pts1_fwd, &pts2);
        let dist_rev = max_nearest_dist(&pts1_rev, &pts2);
        let max_dist = dist_fwd.min(dist_rev);
        assert!(
            max_dist < TOL,
            "periodic[{idx}]: blk{b1_idx} -> blk{b2_idx} max_dist={max_dist:.2e} >= TOL (fwd={dist_fwd:.2e}, rev={dist_rev:.2e})"
        );
    }
}

#[test]
fn test_connectivity_faces_geometry() {
    require_mesh!();
    let (blocks, payload) = build_connectivity_json();

    for (idx, fm) in payload["face_matches"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let b1_idx = fm["block1"]["block_index"].as_u64().unwrap() as usize;
        let b2_idx = fm["block2"]["block_index"].as_u64().unwrap() as usize;
        let lb1 = parse_ijk(&fm["block1"]["lb"]);
        let ub1 = parse_ijk(&fm["block1"]["ub"]);
        let lb2 = parse_ijk(&fm["block2"]["lb"]);
        let ub2 = parse_ijk(&fm["block2"]["ub"]);

        let pts1 = extract_face_points(&blocks[b1_idx], &lb1, &ub1);
        let pts2 = extract_face_points(&blocks[b2_idx], &lb2, &ub2);

        let max_dist = max_nearest_dist(&pts1, &pts2);
        assert!(
            max_dist < TOL,
            "connectivity[{idx}]: blk{b1_idx} <-> blk{b2_idx} max_dist={max_dist:.2e} >= TOL"
        );
    }
}

/// Invariant regression guard for the Phase 3 "fresh-face validation"
/// overlap bug: two distinct face matches must never share a block pair
/// AND have overlapping (il..ih, jl..jh, kl..kh) index regions on the
/// same side. On an unpatched Phase 3 loop, O-grid-seam meshes emit a
/// wrap-around merged record that overlaps two clean Phase 2 sub-faces
/// on the same pair — see `phase3_overlaps_existing` in
/// `src/connectivity.rs`.
///
/// This assertion is mesh-agnostic and will fire whenever a regression
/// reintroduces the duplicate. It also guards against accidental
/// double-emission from other future refactors of the match pipeline.
#[test]
fn test_no_overlapping_face_matches() {
    require_mesh!();
    let (_blocks, payload) = build_connectivity_json();

    let fms = payload["face_matches"].as_array().unwrap();

    fn norm(a: usize, b: usize) -> (usize, usize) {
        (a.min(b), a.max(b))
    }
    fn ranges_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
        !(a.1 < b.0 || b.1 < a.0)
    }
    fn side_overlaps(a_lb: [usize; 3], a_ub: [usize; 3], b_lb: [usize; 3], b_ub: [usize; 3]) -> bool {
        (0..3).all(|d| ranges_overlap(norm(a_lb[d], a_ub[d]), norm(b_lb[d], b_ub[d])))
    }

    for i in 0..fms.len() {
        for j in (i + 1)..fms.len() {
            let a = &fms[i];
            let b = &fms[j];
            let a_b1 = a["block1"]["block_index"].as_u64().unwrap() as usize;
            let a_b2 = a["block2"]["block_index"].as_u64().unwrap() as usize;
            let b_b1 = b["block1"]["block_index"].as_u64().unwrap() as usize;
            let b_b2 = b["block2"]["block_index"].as_u64().unwrap() as usize;

            let a_lb1 = parse_ijk(&a["block1"]["lb"]);
            let a_ub1 = parse_ijk(&a["block1"]["ub"]);
            let a_lb2 = parse_ijk(&a["block2"]["lb"]);
            let a_ub2 = parse_ijk(&a["block2"]["ub"]);

            // Align b to a's block ordering, or skip if the pair doesn't match.
            let (b_lb1, b_ub1, b_lb2, b_ub2) = if a_b1 == b_b1 && a_b2 == b_b2 {
                (
                    parse_ijk(&b["block1"]["lb"]),
                    parse_ijk(&b["block1"]["ub"]),
                    parse_ijk(&b["block2"]["lb"]),
                    parse_ijk(&b["block2"]["ub"]),
                )
            } else if a_b1 == b_b2 && a_b2 == b_b1 {
                (
                    parse_ijk(&b["block2"]["lb"]),
                    parse_ijk(&b["block2"]["ub"]),
                    parse_ijk(&b["block1"]["lb"]),
                    parse_ijk(&b["block1"]["ub"]),
                )
            } else {
                continue;
            };

            let side1 = side_overlaps(a_lb1, a_ub1, b_lb1, b_ub1);
            let side2 = side_overlaps(a_lb2, a_ub2, b_lb2, b_ub2);
            assert!(
                !(side1 && side2),
                "face_matches[{i}] and face_matches[{j}] on pair \
                 ({a_b1},{a_b2}) overlap on both sides: \
                 a=(blk{a_b1} {a_lb1:?}..{a_ub1:?}, blk{a_b2} {a_lb2:?}..{a_ub2:?}) \
                 b=(blk{a_b1} {b_lb1:?}..{b_ub1:?}, blk{a_b2} {b_lb2:?}..{b_ub2:?})"
            );
        }
    }
}
