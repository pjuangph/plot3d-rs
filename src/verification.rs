//! Gold-standard verification for connectivity and periodicity.
//!
//! Pure pass/fail checks — takes lb/ub as given from the pipeline,
//! does directed traversal, compares point-by-point.
//! Do NOT modify these to get better results.
//!
//! Checks:
//!   1. Face AREA must match: `di*dj*dk == di'*dj'*dk'`
//!   2. Every point from face1 (traversing lb→ub) must match the
//!      corresponding point from face2 (traversing lb→ub).

use crate::block::Block;
use crate::face_record::{FaceMatch, FaceRecord};
use crate::rotational_periodicity::create_rotation_matrix;
use crate::utils::apply_rotation;
use crate::Float;

/// Compute face area (total number of points) from diagonal corners.
fn face_area(rec: &FaceRecord) -> usize {
    let di = if rec.ih >= rec.il { rec.ih - rec.il + 1 } else { rec.il - rec.ih + 1 };
    let dj = if rec.jh >= rec.jl { rec.jh - rec.jl + 1 } else { rec.jl - rec.jh + 1 };
    let dk = if rec.kh >= rec.kl { rec.kh - rec.kl + 1 } else { rec.kl - rec.kh + 1 };
    di * dj * dk
}

/// Build an inclusive range from `start` to `end`, stepping +1 or −1.
fn directed_range(start: usize, end: usize) -> Vec<usize> {
    if start <= end {
        (start..=end).collect()
    } else {
        (end..=start).rev().collect()
    }
}

/// Extract face points in the directed traversal order defined by the FaceRecord
/// diagonals (il,jl,kl) → (ih,jh,kh).
///
/// Point n from face A must match point n from face B — the diagonal
/// convention preserves the node-to-node mapping between blocks.
fn extract_face_points(block: &Block, rec: &FaceRecord) -> Vec<(Float, Float, Float)> {
    let il = rec.il.min(block.imax.saturating_sub(1));
    let ih = rec.ih.min(block.imax.saturating_sub(1));
    let jl = rec.jl.min(block.jmax.saturating_sub(1));
    let jh = rec.jh.min(block.jmax.saturating_sub(1));
    let kl = rec.kl.min(block.kmax.saturating_sub(1));
    let kh = rec.kh.min(block.kmax.saturating_sub(1));

    let i_range = directed_range(il, ih);
    let j_range = directed_range(jl, jh);
    let k_range = directed_range(kl, kh);

    let mut pts = Vec::with_capacity(i_range.len() * j_range.len() * k_range.len());
    for &i in &i_range {
        for &j in &j_range {
            for &k in &k_range {
                pts.push(block.xyz(i, j, k));
            }
        }
    }
    pts
}

/// Verify connectivity face matches using area test + directed traversal.
///
/// For each match:
///   1. Check that face areas match (di*dj*dk).
///   2. Extract points via directed lb→ub traversal and compare point-by-point.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_connectivity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            eprintln!(
                "  AREA MISMATCH: match block {}↔{} area {} vs {}",
                b1.block_index, b2.block_index, area1, area2
            );
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // 2. Directed traversal point-by-point comparison
        let pts1 = extract_face_points(block1, b1);
        let pts2 = extract_face_points(block2, b2);

        if pts1.len() != pts2.len() {
            eprintln!(
                "  LENGTH MISMATCH: match block {}↔{} {} vs {} points",
                b1.block_index, b2.block_index, pts1.len(), pts2.len()
            );
            point_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let worst = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
            .fold(0.0 as Float, Float::max);

        if worst > tol2 {
            let n_bad = pts1
                .iter()
                .zip(pts2.iter())
                .filter(|(a, b)| {
                    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2) > tol2
                })
                .count();
            eprintln!(
                "  POINT MISMATCH: match block {}↔{} {}/{} points, max_dist={:.6e}",
                b1.block_index,
                b2.block_index,
                n_bad,
                pts1.len(),
                worst.sqrt()
            );
            point_failures += 1;
            mismatched.push(fm.clone());
        } else {
            verified.push(fm.clone());
        }
    }

    eprintln!(
        "  verify_connectivity: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}

/// Brute-force connectivity verification: tries all 8 diagonal orientations of face2.
///
/// For each match, keeps face1 fixed and tries all 8 possible lb/ub assignments
/// for face2 (all combinations of lo/hi for each axis). If any orientation
/// produces a point-by-point match via directed traversal, the match is verified
/// and face2's lb/ub are set to the matching orientation.
///
/// This is the "true" solution — exhaustive search that finds the correct
/// orientation if one exists. Use this to validate that the pipeline is
/// finding all valid connectivities.
///
/// # Returns
/// `(verified, mismatched)` — verified matches have face2's lb/ub corrected.
pub fn connectivity_bruteforce(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // Face1 points — fixed orientation
        let pts1 = extract_face_points(block1, b1);

        // 2. Try all 8 diagonal orientations of face2
        let i_lo = b2.i_lo().min(block2.imax.saturating_sub(1));
        let i_hi = b2.i_hi().min(block2.imax.saturating_sub(1));
        let j_lo = b2.j_lo().min(block2.jmax.saturating_sub(1));
        let j_hi = b2.j_hi().min(block2.jmax.saturating_sub(1));
        let k_lo = b2.k_lo().min(block2.kmax.saturating_sub(1));
        let k_hi = b2.k_hi().min(block2.kmax.saturating_sub(1));

        let diagonals = [
            (i_lo, j_lo, k_lo, i_hi, j_hi, k_hi),
            (i_hi, j_lo, k_lo, i_lo, j_hi, k_hi),
            (i_lo, j_hi, k_lo, i_hi, j_lo, k_hi),
            (i_lo, j_lo, k_hi, i_hi, j_hi, k_lo),
            (i_hi, j_hi, k_lo, i_lo, j_lo, k_hi),
            (i_hi, j_lo, k_hi, i_lo, j_hi, k_lo),
            (i_lo, j_hi, k_hi, i_hi, j_lo, k_lo),
            (i_hi, j_hi, k_hi, i_lo, j_lo, k_lo),
        ];

        let mut found = false;
        for &(il, jl, kl, ih, jh, kh) in &diagonals {
            let mut rec2 = b2.clone();
            rec2.il = il;
            rec2.jl = jl;
            rec2.kl = kl;
            rec2.ih = ih;
            rec2.jh = jh;
            rec2.kh = kh;

            let pts2 = extract_face_points(block2, &rec2);
            if pts1.len() != pts2.len() {
                continue;
            }

            let worst = pts1
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }
        }

        if !found {
            point_failures += 1;
            mismatched.push(fm.clone());
        }
    }

    eprintln!(
        "  connectivity_bruteforce: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}

/// Brute-force periodic verification: tries all 8 diagonal orientations × ±theta.
///
/// Same as `connectivity_bruteforce` but rotates face1 points by ±theta
/// before comparing. Tries all 16 combinations (8 diagonals × 2 rotations).
///
/// # Returns
/// `(verified, mismatched)` — verified matches have face2's lb/ub corrected.
pub fn periodicity_bruteforce(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    theta: Float,
    rotation_axis: char,
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let rot_pos = create_rotation_matrix(theta, rotation_axis);
    let rot_neg = create_rotation_matrix(-theta, rotation_axis);

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // Face1 points — fixed orientation
        let pts1_raw = extract_face_points(block1, b1);

        // Pre-compute rotated versions of face1
        let pts1_pos: Vec<(Float, Float, Float)> = pts1_raw
            .iter()
            .map(|p| {
                let r = apply_rotation([p.0, p.1, p.2], rot_pos);
                (r[0], r[1], r[2])
            })
            .collect();
        let pts1_neg: Vec<(Float, Float, Float)> = pts1_raw
            .iter()
            .map(|p| {
                let r = apply_rotation([p.0, p.1, p.2], rot_neg);
                (r[0], r[1], r[2])
            })
            .collect();

        // 2. Try all 8 diagonal orientations × 2 rotations
        let i_lo = b2.i_lo().min(block2.imax.saturating_sub(1));
        let i_hi = b2.i_hi().min(block2.imax.saturating_sub(1));
        let j_lo = b2.j_lo().min(block2.jmax.saturating_sub(1));
        let j_hi = b2.j_hi().min(block2.jmax.saturating_sub(1));
        let k_lo = b2.k_lo().min(block2.kmax.saturating_sub(1));
        let k_hi = b2.k_hi().min(block2.kmax.saturating_sub(1));

        let diagonals = [
            (i_lo, j_lo, k_lo, i_hi, j_hi, k_hi),
            (i_hi, j_lo, k_lo, i_lo, j_hi, k_hi),
            (i_lo, j_hi, k_lo, i_hi, j_lo, k_hi),
            (i_lo, j_lo, k_hi, i_hi, j_hi, k_lo),
            (i_hi, j_hi, k_lo, i_lo, j_lo, k_hi),
            (i_hi, j_lo, k_hi, i_lo, j_hi, k_lo),
            (i_lo, j_hi, k_hi, i_hi, j_lo, k_lo),
            (i_hi, j_hi, k_hi, i_lo, j_lo, k_lo),
        ];

        let mut found = false;
        for &(il, jl, kl, ih, jh, kh) in &diagonals {
            let mut rec2 = b2.clone();
            rec2.il = il;
            rec2.jl = jl;
            rec2.kl = kl;
            rec2.ih = ih;
            rec2.jh = jh;
            rec2.kh = kh;

            let pts2 = extract_face_points(block2, &rec2);
            if pts1_pos.len() != pts2.len() {
                continue;
            }

            // Try +theta
            let worst_pos = pts1_pos
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst_pos <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }

            // Try -theta
            let worst_neg = pts1_neg
                .iter()
                .zip(pts2.iter())
                .map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2))
                .fold(0.0 as Float, Float::max);

            if worst_neg <= tol2 {
                let mut fm_out = fm.clone();
                fm_out.block2.il = il;
                fm_out.block2.jl = jl;
                fm_out.block2.kl = kl;
                fm_out.block2.ih = ih;
                fm_out.block2.jh = jh;
                fm_out.block2.kh = kh;
                verified.push(fm_out);
                found = true;
                break;
            }
        }

        if !found {
            point_failures += 1;
            mismatched.push(fm.clone());
        }
    }

    eprintln!(
        "  periodicity_bruteforce: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}

/// Verify periodic face matches using area test + directed traversal with rotation.
///
/// For each match:
///   1. Check that face areas match.
///   2. Extract points via directed lb→ub traversal.
///   3. Rotate face1 points by +theta and -theta.
///   4. Compare point-by-point, picking the better rotation.
///
/// # Returns
/// `(verified, mismatched)` vectors of face matches.
pub fn verify_periodicity(
    blocks: &[Block],
    face_matches: &[FaceMatch],
    theta: Float,
    rotation_axis: char,
    tol: Float,
) -> (Vec<FaceMatch>, Vec<FaceMatch>) {
    let tol2 = tol * tol;
    let rot_pos = create_rotation_matrix(theta, rotation_axis);
    let rot_neg = create_rotation_matrix(-theta, rotation_axis);

    let mut verified = Vec::new();
    let mut mismatched = Vec::new();
    let mut area_failures = 0usize;
    let mut point_failures = 0usize;

    for fm in face_matches {
        let b1 = &fm.block1;
        let b2 = &fm.block2;

        if b1.block_index >= blocks.len() || b2.block_index >= blocks.len() {
            mismatched.push(fm.clone());
            continue;
        }

        // 1. Area check
        let area1 = face_area(b1);
        let area2 = face_area(b2);
        if area1 != area2 {
            eprintln!(
                "  AREA MISMATCH: periodic match block {}↔{} area {} vs {}",
                b1.block_index, b2.block_index, area1, area2
            );
            area_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        let block1 = &blocks[b1.block_index];
        let block2 = &blocks[b2.block_index];

        // 2. Extract points via directed traversal
        let pts1 = extract_face_points(block1, b1);
        let pts2 = extract_face_points(block2, b2);

        if pts1.len() != pts2.len() {
            eprintln!(
                "  LENGTH MISMATCH: periodic match block {}↔{} {} vs {} points",
                b1.block_index, b2.block_index, pts1.len(), pts2.len()
            );
            point_failures += 1;
            mismatched.push(fm.clone());
            continue;
        }

        // 3. Try +theta and -theta, pick better
        let worst_pos = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| {
                let r = apply_rotation([a.0, a.1, a.2], rot_pos);
                (r[0] - b.0).powi(2) + (r[1] - b.1).powi(2) + (r[2] - b.2).powi(2)
            })
            .fold(0.0 as Float, Float::max);

        let worst_neg = pts1
            .iter()
            .zip(pts2.iter())
            .map(|(a, b)| {
                let r = apply_rotation([a.0, a.1, a.2], rot_neg);
                (r[0] - b.0).powi(2) + (r[1] - b.1).powi(2) + (r[2] - b.2).powi(2)
            })
            .fold(0.0 as Float, Float::max);

        let best_worst = worst_pos.min(worst_neg);

        if best_worst > tol2 {
            let best_rot = if worst_pos <= worst_neg {
                rot_pos
            } else {
                rot_neg
            };
            let n_bad = pts1
                .iter()
                .zip(pts2.iter())
                .filter(|(a, b)| {
                    let r = apply_rotation([a.0, a.1, a.2], best_rot);
                    (r[0] - b.0).powi(2) + (r[1] - b.1).powi(2) + (r[2] - b.2).powi(2) > tol2
                })
                .count();
            eprintln!(
                "  POINT MISMATCH: periodic match block {}↔{} {}/{} points, max_dist={:.6e}",
                b1.block_index,
                b2.block_index,
                n_bad,
                pts1.len(),
                best_worst.sqrt()
            );
            point_failures += 1;
            mismatched.push(fm.clone());
        } else {
            verified.push(fm.clone());
        }
    }

    eprintln!(
        "  verify_periodicity: {} verified, {} area failures, {} point failures",
        verified.len(),
        area_failures,
        point_failures
    );
    (verified, mismatched)
}
