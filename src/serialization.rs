//! JSON serialization for face records and face matches.
//!
//! Face records export directed `lb`/`ub` corners (raw `il/jl/kl` and
//! `ih/jh/kh`). After [`face_matches_to_dict`], these corners are physically
//! corresponding: block1's `lb` matches block2's `lb` in xyz space, and
//! block1's `ub` matches block2's `ub`.
//!
//! The `permutation_index` (0-7) indicates which [`PERMUTATION_MATRICES`]
//! entry transforms face B's **ascending canonical grid** to match face A's.
//! It is included as metadata; the directed corners already encode the
//! corner-to-corner mapping.
//!
//! ```json
//! {
//!   "block1": { "block_index": 0, "lb": [0,0,0], "ub": [24,408,0] },
//!   "block2": { "block_index": 1, "lb": [408,0,0], "ub": [0,0,24] },
//!   "permutation_index": 5
//! }
//! ```
//!
//! [`face_matches_to_dict`]: crate::connectivity::face_matches_to_dict
//! [`PERMUTATION_MATRICES`]: crate::face_record::PERMUTATION_MATRICES

use crate::face_record::{FaceMatch, FaceRecord, PERMUTATION_MATRICES};
use serde_json::{json, Value};

/// Convert a [`FaceRecord`] to JSON with directed `lb`/`ub` corners.
///
/// Uses `il/jl/kl` as `lb` and `ih/jh/kh` as `ub` directly — no sorting.
/// After `face_matches_to_dict`, these corners are physically corresponding:
/// block1's lb matches block2's lb in xyz space.
pub fn face_record_to_json(rec: &FaceRecord) -> Value {
    let lb = [rec.il, rec.jl, rec.kl];
    let ub = [rec.ih, rec.jh, rec.kh];
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

/// Convert a [`FaceMatch`] to JSON with directed `lb`/`ub` corners.
///
/// Both block1 and block2 export raw `lb`/`ub` (il/jl/kl → ih/jh/kh).
/// The `permutation_index` is included for reference but the corners
/// already encode the full mapping: lb1 ↔ lb2, ub1 ↔ ub2 in physical space.
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

/// Serialize the 8 permutation matrices as a JSON array (for inclusion in output headers).
pub fn permutation_matrices_json() -> Vec<Value> {
    PERMUTATION_MATRICES
        .iter()
        .map(|m| json!([[m[0][0], m[0][1]], [m[1][0], m[1][1]]]))
        .collect()
}
