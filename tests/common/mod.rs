//! Shared test utilities for connectivity debug and diagnostic tests.
//!
//! Provides JSON deserialization types for GridPro reference data,
//! orientation-agnostic face comparison types, and helper functions
//! for comparing connectivity results.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use plot3d::{FaceMatch, FaceRecord};

// ---------------------------------------------------------------------------
// JSON deserialization types (GridPro reference format)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct JsonBlockRecord {
    pub block_index: usize,
    #[serde(rename = "IMIN")]
    pub imin: usize,
    #[serde(rename = "JMIN")]
    pub jmin: usize,
    #[serde(rename = "KMIN")]
    pub kmin: usize,
    #[serde(rename = "IMAX")]
    pub imax: usize,
    #[serde(rename = "JMAX")]
    pub jmax: usize,
    #[serde(rename = "KMAX")]
    pub kmax: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonFaceMatch {
    pub block1: JsonBlockRecord,
    pub block2: JsonBlockRecord,
}

#[derive(Debug, Deserialize)]
pub struct GridProConnectivity {
    pub face_matches: Vec<JsonFaceMatch>,
}

// ---------------------------------------------------------------------------
// Normalized face record (orientation-agnostic, imin<=imax etc.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct NormalizedFaceRecord {
    pub block_index: usize,
    pub imin: usize,
    pub jmin: usize,
    pub kmin: usize,
    pub imax: usize,
    pub jmax: usize,
    pub kmax: usize,
}

impl NormalizedFaceRecord {
    pub fn from_json_block(b: &JsonBlockRecord) -> Self {
        Self {
            block_index: b.block_index,
            imin: b.imin.min(b.imax),
            jmin: b.jmin.min(b.jmax),
            kmin: b.kmin.min(b.kmax),
            imax: b.imin.max(b.imax),
            jmax: b.jmin.max(b.jmax),
            kmax: b.kmin.max(b.kmax),
        }
    }

    pub fn from_face_record(r: &FaceRecord) -> Self {
        Self {
            block_index: r.block_index,
            imin: r.i_lo(),
            jmin: r.j_lo(),
            kmin: r.k_lo(),
            imax: r.i_hi(),
            jmax: r.j_hi(),
            kmax: r.k_hi(),
        }
    }

    pub fn face_tuple(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.imin, self.jmin, self.kmin, self.imax, self.jmax, self.kmax,
        )
    }
}

// ---------------------------------------------------------------------------
// MatchKey: canonical unordered pair of faces
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MatchKey {
    pub face_a: NormalizedFaceRecord,
    pub face_b: NormalizedFaceRecord,
}

impl MatchKey {
    pub fn new(a: NormalizedFaceRecord, b: NormalizedFaceRecord) -> Self {
        if a.block_index < b.block_index
            || (a.block_index == b.block_index && a.face_tuple() <= b.face_tuple())
        {
            MatchKey {
                face_a: a,
                face_b: b,
            }
        } else {
            MatchKey {
                face_a: b,
                face_b: a,
            }
        }
    }

    pub fn block_pair(&self) -> (usize, usize) {
        let lo = self.face_a.block_index.min(self.face_b.block_index);
        let hi = self.face_a.block_index.max(self.face_b.block_index);
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Key-building helpers
// ---------------------------------------------------------------------------

pub fn build_json_keys(json_matches: &[JsonFaceMatch]) -> HashSet<MatchKey> {
    json_matches
        .iter()
        .map(|jm| {
            let a = NormalizedFaceRecord::from_json_block(&jm.block1);
            let b = NormalizedFaceRecord::from_json_block(&jm.block2);
            MatchKey::new(a, b)
        })
        .collect()
}

pub fn build_computed_keys(computed_matches: &[FaceMatch]) -> HashSet<MatchKey> {
    computed_matches
        .iter()
        .map(|fm| {
            let a = NormalizedFaceRecord::from_face_record(&fm.block1);
            let b = NormalizedFaceRecord::from_face_record(&fm.block2);
            MatchKey::new(a, b)
        })
        .collect()
}

pub fn group_by_block_pair(keys: &HashSet<MatchKey>) -> HashMap<(usize, usize), Vec<MatchKey>> {
    let mut map: HashMap<(usize, usize), Vec<MatchKey>> = HashMap::new();
    for k in keys {
        map.entry(k.block_pair()).or_default().push(k.clone());
    }
    map
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

/// Compute the minimum corner distance between two JSON face records,
/// trying both direct and swapped orientations.
/// Returns (min_distance, is_swapped).
pub fn min_corner_distance(jm: &JsonFaceMatch, blocks: &[plot3d::Block]) -> Option<(f64, bool)> {
    let b1_idx = jm.block1.block_index;
    let b2_idx = jm.block2.block_index;
    if b1_idx >= blocks.len() || b2_idx >= blocks.len() {
        return None;
    }
    let block1 = &blocks[b1_idx];
    let block2 = &blocks[b2_idx];

    let n1 = NormalizedFaceRecord::from_json_block(&jm.block1);
    let n2 = NormalizedFaceRecord::from_json_block(&jm.block2);

    if n1.imax >= block1.imax || n1.jmax >= block1.jmax || n1.kmax >= block1.kmax {
        return None;
    }
    if n2.imax >= block2.imax || n2.jmax >= block2.jmax || n2.kmax >= block2.kmax {
        return None;
    }

    let (x1_lo, y1_lo, z1_lo) = block1.xyz(n1.imin, n1.jmin, n1.kmin);
    let (x1_hi, y1_hi, z1_hi) = block1.xyz(n1.imax, n1.jmax, n1.kmax);
    let (x2_lo, y2_lo, z2_lo) = block2.xyz(n2.imin, n2.jmin, n2.kmin);
    let (x2_hi, y2_hi, z2_hi) = block2.xyz(n2.imax, n2.jmax, n2.kmax);

    let d_direct = ((x1_lo - x2_lo).powi(2) + (y1_lo - y2_lo).powi(2) + (z1_lo - z2_lo).powi(2))
        .sqrt()
        .max(((x1_hi - x2_hi).powi(2) + (y1_hi - y2_hi).powi(2) + (z1_hi - z2_hi).powi(2)).sqrt());
    let d_swapped = ((x1_lo - x2_hi).powi(2) + (y1_lo - y2_hi).powi(2) + (z1_lo - z2_hi).powi(2))
        .sqrt()
        .max(((x1_hi - x2_lo).powi(2) + (y1_hi - y2_lo).powi(2) + (z1_hi - z2_lo).powi(2)).sqrt());

    if d_direct <= d_swapped {
        Some((d_direct, false))
    } else {
        Some((d_swapped, true))
    }
}

/// Check if two blocks have overlapping AABBs within tolerance.
pub fn aabb_overlap(b1: &plot3d::Block, b2: &plot3d::Block, tol: f64) -> bool {
    let aabb = |b: &plot3d::Block| -> [f64; 6] {
        let mut mn = [f64::INFINITY; 3];
        let mut mx = [f64::NEG_INFINITY; 3];
        for &x in &b.x {
            mn[0] = mn[0].min(x);
            mx[0] = mx[0].max(x);
        }
        for &y in &b.y {
            mn[1] = mn[1].min(y);
            mx[1] = mx[1].max(y);
        }
        for &z in &b.z {
            mn[2] = mn[2].min(z);
            mx[2] = mx[2].max(z);
        }
        [mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]
    };
    let a = aabb(b1);
    let b = aabb(b2);
    a[1] + tol >= b[0]
        && b[1] + tol >= a[0]
        && a[3] + tol >= b[2]
        && b[3] + tol >= a[2]
        && a[5] + tol >= b[4]
        && b[5] + tol >= a[4]
}
