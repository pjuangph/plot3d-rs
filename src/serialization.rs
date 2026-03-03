//! JSON serialization for face records and face matches.
//!
//! Provides two output formats controlled by the `--diagonal` flag:
//!
//! # Default format (lo/hi)
//!
//! Face bounds use ascending `lo`/`hi` keys. Every match includes a
//! `permutation_index` (0-7) indicating which [`PERMUTATION_MATRICES`]
//! entry transforms face B to match face A.
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lo": [0,0,0], "hi": [0,101,33] },
//!   "block2": { "block_index": 30, "lo": [0,0,0], "hi": [0,101,33] },
//!   "permutation_index": 3
//! }
//! ```
//!
//! # Diagonal format (`--diagonal`)
//!
//! - **In-plane** matches (perm 0-3): block2's `lb`/`ub` encodes traversal
//!   direction. `permutation_index: -1` (direction is fully in the bounds).
//! - **Cross-plane** matches (perm 4-7): ascending `lb`/`ub` bounds with the
//!   actual `permutation_index`, since lb/ub can't encode a swap.
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lb": [0,0,0], "ub": [0,101,33] },
//!   "block2": { "block_index": 30, "lb": [0,101,33], "ub": [0,0,0] },
//!   "permutation_index": -1
//! }
//! ```
//!
//! [`PERMUTATION_MATRICES`]: crate::face_record::PERMUTATION_MATRICES

use crate::face_record::{FaceMatch, FaceRecord, OrientationPlane, PERMUTATION_MATRICES};
use serde_json::{json, Value};

// ── Default format (lo/hi) ──────────────────────────────────────────────

/// Convert a [`FaceRecord`] to JSON with ascending `lo`/`hi` bounds.
pub fn face_record_to_json(rec: &FaceRecord) -> Value {
    let mut obj = json!({
        "block_index": rec.block_index,
        "lo": [rec.il.min(rec.ih), rec.jl.min(rec.jh), rec.kl.min(rec.kh)],
        "hi": [rec.il.max(rec.ih), rec.jl.max(rec.jh), rec.kl.max(rec.kh)],
    });
    if let Some(id) = rec.id {
        obj["id"] = json!(id);
    }
    obj
}

/// Convert a [`FaceMatch`] to JSON (`lo`/`hi` + `permutation_index` 0-7).
pub fn face_match_to_json(fm: &FaceMatch) -> Value {
    let perm_idx: i8 = fm
        .orientation
        .as_ref()
        .map(|o| o.permutation_index as i8)
        .unwrap_or(0);
    json!({
        "block1": face_record_to_json(&fm.block1),
        "block2": face_record_to_json(&fm.block2),
        "permutation_index": perm_idx,
    })
}

// ── Diagonal format (lb/ub) ─────────────────────────────────────────────

/// Convert a [`FaceRecord`] to JSON with ascending `lb`/`ub` bounds.
pub fn face_record_to_diagonal_json(rec: &FaceRecord) -> Value {
    let mut obj = json!({
        "block_index": rec.block_index,
        "lb": [rec.il.min(rec.ih), rec.jl.min(rec.jh), rec.kl.min(rec.kh)],
        "ub": [rec.il.max(rec.ih), rec.jl.max(rec.jh), rec.kl.max(rec.kh)],
    });
    if let Some(id) = rec.id {
        obj["id"] = json!(id);
    }
    obj
}

/// Convert a [`FaceRecord`] to JSON with directional `lb`/`ub` based on permutation.
///
/// Reconstructs traversal direction from `permutation_index` bits:
/// - bit 0 (`u_reversed`): reverse the first varying axis
/// - bit 1 (`v_reversed`): reverse the second varying axis
fn face_record_to_directed_diagonal_json(rec: &FaceRecord, perm_idx: u8) -> Value {
    let lo = [
        rec.il.min(rec.ih),
        rec.jl.min(rec.jh),
        rec.kl.min(rec.kh),
    ];
    let hi = [
        rec.il.max(rec.ih),
        rec.jl.max(rec.jh),
        rec.kl.max(rec.kh),
    ];

    // Find constant axis (where lo == hi)
    let const_ax = (0..3usize).find(|&d| lo[d] == hi[d]);

    let (lb, ub) = match const_ax {
        Some(c) => {
            let vary: Vec<usize> = (0..3).filter(|&d| d != c).collect();
            let d0 = vary[0]; // u axis
            let d1 = vary[1]; // v axis
            let u_rev = perm_idx & 1 != 0;
            let v_rev = perm_idx & 2 != 0;

            let mut lb = lo;
            let mut ub = hi;
            if u_rev {
                lb[d0] = hi[d0];
                ub[d0] = lo[d0];
            }
            if v_rev {
                lb[d1] = hi[d1];
                ub[d1] = lo[d1];
            }
            (lb, ub)
        }
        None => (lo, hi),
    };

    let mut obj = json!({
        "block_index": rec.block_index,
        "lb": lb,
        "ub": ub,
    });
    if let Some(id) = rec.id {
        obj["id"] = json!(id);
    }
    obj
}

/// Convert a [`FaceMatch`] to diagonal JSON format.
///
/// - **In-plane** (perm 0-3): block2's `lb`/`ub` encodes direction. `permutation_index: -1`.
/// - **Cross-plane** (perm 4-7): ascending `lb`/`ub`. `permutation_index: N` (actual index).
pub fn face_match_to_diagonal_json(fm: &FaceMatch) -> Value {
    let orient = fm.orientation.as_ref();
    let perm_idx = orient.map(|o| o.permutation_index).unwrap_or(0);
    let is_cross_plane = orient
        .map(|o| o.plane == OrientationPlane::CrossPlane)
        .unwrap_or(false);

    if is_cross_plane {
        // Cross-plane: lb/ub can't encode swap → ascending bounds + actual permutation_index
        json!({
            "block1": face_record_to_diagonal_json(&fm.block1),
            "block2": face_record_to_diagonal_json(&fm.block2),
            "permutation_index": perm_idx,
        })
    } else {
        // In-plane: encode direction in block2's lb/ub, permutation_index = -1
        json!({
            "block1": face_record_to_diagonal_json(&fm.block1),
            "block2": face_record_to_directed_diagonal_json(&fm.block2, perm_idx),
            "permutation_index": -1,
        })
    }
}

/// Serialize the 8 permutation matrices as a JSON array (for inclusion in output headers).
pub fn permutation_matrices_json() -> Vec<Value> {
    PERMUTATION_MATRICES
        .iter()
        .map(|m| json!([[m[0][0], m[0][1]], [m[1][0], m[1][1]]]))
        .collect()
}
