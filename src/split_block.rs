//! Block splitting utilities that preserve multi-grid compatibility.

use crate::block::Block;
use crate::utils::gcd_three;

/// Direction along which to split a block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SplitDirection {
    I = 0,
    J = 1,
    K = 2,
}

/// Compute the maximum cell aspect ratio at the 8 corners of a block.
///
/// Examines the first-cell edge lengths at each corner and returns the
/// maximum ratio of the two largest edges to the smallest.
pub fn max_aspect_ratio(block: &Block) -> f64 {
    let ix = block.imax - 1;
    let jx = block.jmax - 1;
    let kx = block.kmax - 1;

    // 8 corners: (i1, j1, k1) and their diagonal neighbors (i2, j2, k2)
    let i1 = [0, ix, 0, 0, 0, ix, ix, ix];
    let j1 = [0, 0, jx, 0, jx, 0, jx, jx];
    let k1 = [0, 0, 0, kx, kx, kx, 0, kx];

    let i2 = [1, ix.saturating_sub(1).max(1), 1, 1, 1, ix.saturating_sub(1).max(1), ix.saturating_sub(1).max(1), ix.saturating_sub(1).max(1)];
    let j2 = [1, 1, jx.saturating_sub(1).max(1), 1, jx.saturating_sub(1).max(1), 1, jx.saturating_sub(1).max(1), jx.saturating_sub(1).max(1)];
    let k2 = [1, 1, 1, kx.saturating_sub(1).max(1), kx.saturating_sub(1).max(1), kx.saturating_sub(1).max(1), 1, kx.saturating_sub(1).max(1)];

    let dist = |a: (f64, f64, f64), b: (f64, f64, f64)| -> f64 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let dz = a.2 - b.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    };

    let mut max_ar = 0.0f64;
    for n in 0..8 {
        let base = block.xyz(i1[n], j1[n], k1[n]);
        let di = dist(block.xyz(i2[n], j1[n], k1[n]), base);
        let dj = dist(block.xyz(i1[n], j2[n], k1[n]), base);
        let dk = dist(block.xyz(i1[n], j1[n], k2[n]), base);

        let mut ds = [di, dj, dk];
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if ds[0] > 0.0 {
            max_ar = max_ar.max(ds[2] / ds[0]);
        }
    }
    max_ar
}

/// Search for a valid step size that maintains GCD compatibility.
fn step_search(
    total_cells: usize,
    gcd: usize,
    ncells_per_block: usize,
    denominator: usize,
    forward: bool,
) -> Option<usize> {
    if denominator == 0 || gcd == 0 {
        return None;
    }
    let initial_guess = ncells_per_block / denominator;
    if initial_guess == 0 {
        return None;
    }

    let mut step_size = initial_guess;
    let increment: isize = if forward { 1 } else { -1 };
    let lower = initial_guess / 2;
    let upper = initial_guess * 3 / 2;

    loop {
        let remainder_cells = total_cells % (step_size * denominator);
        let remainder_dim = remainder_cells / denominator;

        if step_size % gcd == 0
            && (remainder_dim == 0 || (remainder_dim > 0 && (remainder_dim - 1) % gcd == 0))
        {
            return Some(step_size);
        }

        let next = step_size as isize + increment;
        if next <= 0 {
            return None;
        }
        step_size = next as usize;

        if step_size < lower || step_size > upper {
            return None;
        }
    }
}

/// Split blocks to achieve approximately `ncells_per_block` cells per block.
///
/// The split preserves the greatest common divisor of `(imax-1, jmax-1, kmax-1)`
/// to maintain multi-grid compatibility.
///
/// If `direction` is `None`, the shortest axis of each block is chosen
/// automatically (preserves Python behavior).
pub fn split_blocks(
    blocks: &[Block],
    ncells_per_block: usize,
    direction: Option<SplitDirection>,
) -> Vec<Block> {
    let mut new_blocks = Vec::new();

    for block in blocks {
        let total_cells = block.imax * block.jmax * block.kmax;

        if total_cells <= ncells_per_block {
            new_blocks.push(block.clone());
            continue;
        }

        // Auto-select direction: pick the shortest axis (Python uses argmin)
        let dir = direction.unwrap_or_else(|| {
            let dims = [block.imax, block.jmax, block.kmax];
            let min_idx = dims
                .iter()
                .enumerate()
                .min_by_key(|&(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            match min_idx {
                0 => SplitDirection::I,
                1 => SplitDirection::J,
                _ => SplitDirection::K,
            }
        });

        let gcd = gcd_three(
            block.imax.saturating_sub(1).max(1),
            block.jmax.saturating_sub(1).max(1),
            block.kmax.saturating_sub(1).max(1),
        );

        let (axis_max, denominator) = match dir {
            SplitDirection::I => (block.imax, block.jmax * block.kmax),
            SplitDirection::J => (block.jmax, block.imax * block.kmax),
            SplitDirection::K => (block.kmax, block.imax * block.jmax),
        };

        let step_size = step_search(total_cells, gcd, ncells_per_block, denominator, false)
            .or_else(|| step_search(total_cells, gcd, ncells_per_block, denominator, true));

        let step_size = match step_size {
            Some(s) if s > 0 => s,
            _ => {
                // No valid step found; keep block as-is
                new_blocks.push(block.clone());
                continue;
            }
        };

        let mut prev = 0usize;
        let mut last_split_end = 0usize;

        let mut pos = step_size;
        while pos < axis_max {
            if pos + 1 > axis_max {
                break;
            }

            let sub = match dir {
                SplitDirection::I => {
                    block.sub_block(prev..=pos, 0..=(block.jmax - 1), 0..=(block.kmax - 1))
                }
                SplitDirection::J => {
                    block.sub_block(0..=(block.imax - 1), prev..=pos, 0..=(block.kmax - 1))
                }
                SplitDirection::K => {
                    block.sub_block(0..=(block.imax - 1), 0..=(block.jmax - 1), prev..=pos)
                }
            };
            new_blocks.push(sub);
            last_split_end = pos;
            prev = pos; // Blocks share boundary face
            pos += step_size;
        }

        // Remainder block
        if last_split_end + 1 < axis_max {
            let sub = match dir {
                SplitDirection::I => block.sub_block(
                    last_split_end..=(block.imax - 1),
                    0..=(block.jmax - 1),
                    0..=(block.kmax - 1),
                ),
                SplitDirection::J => block.sub_block(
                    0..=(block.imax - 1),
                    last_split_end..=(block.jmax - 1),
                    0..=(block.kmax - 1),
                ),
                SplitDirection::K => block.sub_block(
                    0..=(block.imax - 1),
                    0..=(block.jmax - 1),
                    last_split_end..=(block.kmax - 1),
                ),
            };
            new_blocks.push(sub);
        }
    }

    new_blocks
}
