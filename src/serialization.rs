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
//! # Round-tripping
//!
//! The write side ([`face_record_to_json`], [`face_match_to_json`]) and the
//! read side ([`face_record_from_json`], [`face_match_from_json`]) form a
//! round-trip pair for the directed-corner JSON format shown above. Because
//! the struct field layout in [`FaceRecord`] (`il/jl/kl`, `ih/jh/kh`) does
//! not match the nested `lb`/`ub` array format that downstream tooling
//! emits, a plain `#[derive(Deserialize)]` is not sufficient: we parse
//! manually to honour the directed-corner contract.
//!
//! [`face_matches_to_dict`]: crate::connectivity::face_matches_to_dict
//! [`PERMUTATION_MATRICES`]: crate::face_record::PERMUTATION_MATRICES

use crate::face_record::{
    FaceMatch, FaceRecord, Orientation, OrientationPlane, PERMUTATION_MATRICES,
};
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
/// The optional `permutation_matrix` (2x2 i8) is exported when present so
/// downstream consumers (e.g. glennht-core) can use the declarative
/// matrix-driven path instead of the bare index.
pub fn face_match_to_json(fm: &FaceMatch) -> Value {
    let perm_idx: i8 = fm
        .orientation
        .as_ref()
        .map(|o| o.permutation_index as i8)
        .unwrap_or(0);
    let mut obj = json!({
        "block1": face_record_to_json(&fm.block1),
        "block2": face_record_to_json(&fm.block2),
        "permutation_index": perm_idx,
    });
    if let Some(m) = fm.orientation.as_ref().and_then(|o| o.permutation_matrix) {
        obj["permutation_matrix"] = json!([
            [m[0][0], m[0][1]],
            [m[1][0], m[1][1]],
        ]);
    }
    obj
}

/// Serialize the 8 permutation matrices as a JSON array (for inclusion in output headers).
pub fn permutation_matrices_json() -> Vec<Value> {
    PERMUTATION_MATRICES
        .iter()
        .map(|m| json!([[m[0][0], m[0][1]], [m[1][0], m[1][1]]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Deserialization (inverse of the `*_to_json` functions above).
//
// These parsers read the directed-corner JSON format back into [`FaceRecord`]
// and [`FaceMatch`]. They exist because:
//
//   1. `#[derive(Deserialize)]` on [`FaceRecord`] would look for flat
//      `il`, `jl`, `kl`, `ih`, `jh`, `kh` fields — not the nested
//      `lb: [il, jl, kl]` / `ub: [ih, jh, kh]` arrays used by the
//      canonical Python tooling and `face_record_to_json`.
//
//   2. Downstream consumers (e.g. solver startup pipelines) all need to
//      read the same connectivity JSON format. Having a single source of
//      truth here avoids re-implementing the parser in every consumer.
//
// Errors are reported as plain `String`s to avoid forcing a dependency on
// `anyhow` or `thiserror` onto callers. If richer error information is
// needed later, this can be upgraded to a dedicated error enum without
// breaking the public signature beyond the error type.
// ---------------------------------------------------------------------------

/// Helper: extract a `usize` from a JSON array element by index.
fn array_usize(arr: &[Value], idx: usize, ctx: &str) -> Result<usize, String> {
    arr.get(idx)
        .ok_or_else(|| format!("{}: missing index {}", ctx, idx))?
        .as_u64()
        .ok_or_else(|| format!("{}: element {} is not a non-negative integer", ctx, idx))
        .map(|v| v as usize)
}

/// Parse a [`FaceRecord`] from its directed-corner JSON representation.
///
/// Expected format (inverse of [`face_record_to_json`]):
///
/// ```json
/// { "block_index": 0, "lb": [0, 0, 0], "ub": [0, 408, 24], "id": 4 }
/// ```
///
/// The `id` field is optional: outer (boundary) faces carry a surface ID
/// while interior match faces typically do not. The optional `u_physical`
/// and `v_physical` fields are not part of the canonical JSON format and
/// are always set to `None` on read.
///
/// # Errors
/// Returns a `String` describing the first malformed or missing field.
pub fn face_record_from_json(val: &Value) -> Result<FaceRecord, String> {
    let block_index = val
        .get("block_index")
        .ok_or_else(|| "face_record: missing 'block_index'".to_string())?
        .as_u64()
        .ok_or_else(|| "face_record: 'block_index' is not a non-negative integer".to_string())?
        as usize;

    let lb = val
        .get("lb")
        .ok_or_else(|| "face_record: missing 'lb' array".to_string())?
        .as_array()
        .ok_or_else(|| "face_record: 'lb' is not an array".to_string())?;
    let ub = val
        .get("ub")
        .ok_or_else(|| "face_record: missing 'ub' array".to_string())?
        .as_array()
        .ok_or_else(|| "face_record: 'ub' is not an array".to_string())?;

    if lb.len() != 3 || ub.len() != 3 {
        return Err(format!(
            "face_record: 'lb' and 'ub' must be 3-element arrays (got lb={}, ub={})",
            lb.len(),
            ub.len()
        ));
    }

    let il = array_usize(lb, 0, "face_record.lb")?;
    let jl = array_usize(lb, 1, "face_record.lb")?;
    let kl = array_usize(lb, 2, "face_record.lb")?;
    let ih = array_usize(ub, 0, "face_record.ub")?;
    let jh = array_usize(ub, 1, "face_record.ub")?;
    let kh = array_usize(ub, 2, "face_record.ub")?;

    // Surface ID is optional; outer faces have it, interface faces do not.
    let id = val.get("id").and_then(|v| v.as_u64()).map(|v| v as usize);

    Ok(FaceRecord {
        block_index,
        il,
        jl,
        kl,
        ih,
        jh,
        kh,
        id,
        u_physical: None,
        v_physical: None,
    })
}

/// Parse a [`FaceMatch`] from its directed-corner JSON representation.
///
/// Expected format (inverse of [`face_match_to_json`]):
///
/// ```json
/// {
///   "block1": { "block_index": 0, "lb": [10, 0, 0], "ub": [10, 5, 5] },
///   "block2": { "block_index": 1, "lb": [0, 0, 0],  "ub": [0, 5, 5]  },
///   "permutation_index": 0
/// }
/// ```
///
/// The `permutation_index` is optional. If present, the returned `FaceMatch`
/// has `orientation = Some(Orientation { permutation_index, plane: InPlane })`.
/// The plane defaults to [`OrientationPlane::InPlane`] because the directed
/// `lb`/`ub` corners already encode the full corner-to-corner mapping — the
/// plane tag is only used by higher-level verification routines.
///
/// `points` (the legacy [`MatchPoint`] list used by multi-phase matching) is
/// set to `Vec::new()` on read; the directed-corner format does not include
/// it.
///
/// # Errors
/// Returns a `String` describing the first malformed or missing field.
///
/// [`MatchPoint`]: crate::face_record::MatchPoint
pub fn face_match_from_json(val: &Value) -> Result<FaceMatch, String> {
    let block1_val = val
        .get("block1")
        .ok_or_else(|| "face_match: missing 'block1'".to_string())?;
    let block2_val = val
        .get("block2")
        .ok_or_else(|| "face_match: missing 'block2'".to_string())?;

    let block1 = face_record_from_json(block1_val)
        .map_err(|e| format!("face_match.block1: {}", e))?;
    let block2 = face_record_from_json(block2_val)
        .map_err(|e| format!("face_match.block2: {}", e))?;

    // Parse the optional `permutation_matrix` (2x2 i8). Preferred over
    // `permutation_index` because it carries enough information to look
    // up the canonical index even when the JSON's index field is the
    // sentinel `-1` (some tools use this to indicate "matrix only").
    let permutation_matrix: Option<[[i8; 2]; 2]> = val
        .get("permutation_matrix")
        .and_then(|v| v.as_array())
        .and_then(|outer| {
            if outer.len() != 2 {
                return None;
            }
            let mut m = [[0i8; 2]; 2];
            for (r, row) in outer.iter().enumerate() {
                let row = row.as_array()?;
                if row.len() != 2 {
                    return None;
                }
                for (c, cell) in row.iter().enumerate() {
                    let v = cell.as_i64()?;
                    if !(-1..=1).contains(&v) {
                        return None;
                    }
                    m[r][c] = v as i8;
                }
            }
            Some(m)
        });

    // Parse orientation from permutation_matrix (preferred) or
    // permutation_index. The `permutation_index` written by
    // `face_match_to_json` is an i8 (signed), so accept any JSON integer
    // in the 0..=7 range.
    let orientation = if let Some(m) = permutation_matrix {
        let idx = Orientation::index_from_permutation_matrix(m).unwrap_or(0);
        Some(Orientation {
            permutation_index: idx,
            plane: OrientationPlane::InPlane,
            permutation_matrix: Some(m),
        })
    } else {
        val.get("permutation_index")
            .and_then(|v| v.as_i64())
            .map(|idx| {
                // Defensive clamp: 0..=7 are the only valid permutation indices.
                let clamped = idx.clamp(0, 7) as u8;
                Orientation {
                    permutation_index: clamped,
                    // Default to in-plane; downstream code that needs cross-plane
                    // information should re-run the verification pipeline, which
                    // populates `plane` explicitly.
                    plane: OrientationPlane::InPlane,
                    permutation_matrix: None,
                }
            })
    };

    Ok(FaceMatch {
        block1,
        block2,
        points: Vec::new(),
        orientation,
    })
}

// ---------------------------------------------------------------------------
// Round-trip tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_record_round_trip() {
        let original = FaceRecord {
            block_index: 2,
            il: 0,
            jl: 0,
            kl: 0,
            ih: 10,
            jh: 20,
            kh: 0,
            id: Some(7),
            u_physical: None,
            v_physical: None,
        };

        let as_json = face_record_to_json(&original);
        let parsed = face_record_from_json(&as_json).expect("round trip");

        assert_eq!(parsed.block_index, original.block_index);
        assert_eq!(parsed.il, original.il);
        assert_eq!(parsed.jl, original.jl);
        assert_eq!(parsed.kl, original.kl);
        assert_eq!(parsed.ih, original.ih);
        assert_eq!(parsed.jh, original.jh);
        assert_eq!(parsed.kh, original.kh);
        assert_eq!(parsed.id, original.id);
    }

    #[test]
    fn face_record_directed_corners_preserved() {
        // Reversed i-axis: il > ih. The parser must NOT sort these; it must
        // preserve the directed orientation.
        let json = json!({
            "block_index": 1,
            "lb": [24, 0, 0],
            "ub": [0, 408, 0]
        });
        let parsed = face_record_from_json(&json).unwrap();
        assert_eq!(parsed.il, 24);
        assert_eq!(parsed.ih, 0);
        assert!(parsed.id.is_none());
    }

    #[test]
    fn face_record_missing_block_index() {
        let json = json!({ "lb": [0, 0, 0], "ub": [1, 1, 1] });
        let err = face_record_from_json(&json).unwrap_err();
        assert!(err.contains("block_index"));
    }

    #[test]
    fn face_record_wrong_array_length() {
        let json = json!({
            "block_index": 0,
            "lb": [0, 0],
            "ub": [1, 1, 1]
        });
        let err = face_record_from_json(&json).unwrap_err();
        assert!(err.contains("3-element"));
    }

    #[test]
    fn face_match_round_trip() {
        let original = FaceMatch {
            block1: FaceRecord {
                block_index: 0,
                il: 10,
                jl: 0,
                kl: 0,
                ih: 10,
                jh: 5,
                kh: 5,
                id: None,
                u_physical: None,
                v_physical: None,
            },
            block2: FaceRecord {
                block_index: 1,
                il: 0,
                jl: 0,
                kl: 0,
                ih: 0,
                jh: 5,
                kh: 5,
                id: None,
                u_physical: None,
                v_physical: None,
            },
            points: Vec::new(),
            orientation: Some(Orientation {
                permutation_index: 5,
                plane: OrientationPlane::InPlane,
                permutation_matrix: None,
            }),
        };

        let as_json = face_match_to_json(&original);
        let parsed = face_match_from_json(&as_json).expect("round trip");

        assert_eq!(parsed.block1.block_index, 0);
        assert_eq!(parsed.block2.block_index, 1);
        assert_eq!(parsed.block1.il, 10);
        assert_eq!(parsed.block2.ih, 0);

        let orient = parsed.orientation.expect("orientation preserved");
        assert_eq!(orient.permutation_index, 5);
    }

    #[test]
    fn face_match_without_permutation_index() {
        let json = json!({
            "block1": { "block_index": 0, "lb": [0, 0, 0], "ub": [1, 1, 0] },
            "block2": { "block_index": 1, "lb": [0, 0, 0], "ub": [1, 1, 0] }
        });
        let parsed = face_match_from_json(&json).unwrap();
        assert!(parsed.orientation.is_none());
        assert!(parsed.points.is_empty());
    }

    #[test]
    fn face_match_missing_block1() {
        let json = json!({
            "block2": { "block_index": 1, "lb": [0, 0, 0], "ub": [1, 1, 0] }
        });
        let err = face_match_from_json(&json).unwrap_err();
        assert!(err.contains("block1"));
    }
}
