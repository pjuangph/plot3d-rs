use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Block {
    pub imax: usize,
    pub jmax: usize,
    pub kmax: usize, // 2D supported via kmax == 1
    pub x: Vec<f64>, // length = imax*jmax*kmax
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// Data for a single block boundary face (a 2D grid of coordinates).
#[derive(Clone, Debug)]
pub struct FaceData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Dimensions of the face grid `(nu, nv)`.
    pub dims: (usize, usize),
}

impl Block {
    pub fn new(
        imax: usize,
        jmax: usize,
        kmax: usize,
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
    ) -> Self {
        let n = imax * jmax * kmax;
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        assert_eq!(z.len(), n);
        Self {
            imax,
            jmax,
            kmax,
            x,
            y,
            z,
        }
    }

    #[inline]
    pub fn npoints(&self) -> usize {
        self.imax * self.jmax * self.kmax
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        // i–j–k order (i fastest)
        debug_assert!(i < self.imax && j < self.jmax && k < self.kmax);
        (k * self.jmax + j) * self.imax + i
    }

    #[inline]
    pub fn xyz(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        let idx = self.idx(i, j, k);
        (self.x[idx], self.y[idx], self.z[idx])
    }

    #[inline]
    pub fn x_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.x[self.idx(i, j, k)]
    }

    #[inline]
    pub fn y_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.y[self.idx(i, j, k)]
    }

    #[inline]
    pub fn z_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.z[self.idx(i, j, k)]
    }

    #[inline]
    pub fn x_slice(&self) -> &[f64] {
        &self.x
    }

    #[inline]
    pub fn y_slice(&self) -> &[f64] {
        &self.y
    }

    #[inline]
    pub fn z_slice(&self) -> &[f64] {
        &self.z
    }

    #[inline]
    pub fn centroid(&self) -> (f64, f64, f64) {
        let n = self.npoints() as f64;
        let sum_x: f64 = self.x.iter().sum();
        let sum_y: f64 = self.y.iter().sum();
        let sum_z: f64 = self.z.iter().sum();
        (sum_x / n, sum_y / n, sum_z / n)
    }

    /// Print the XYZ coordinates at `(i, j, k)` in a readable format.
    pub fn print_xyz(&self, i: usize, j: usize, k: usize) {
        let (x, y, z) = self.xyz(i, j, k);
        println!("XYZ at (i={i}, j={j}, k={k}) is ({x:.6}, {y:.6}, {z:.6})");
    }

    pub fn shifted(&self, amount: f64, axis: char) -> Block {
        let mut new = self.clone();
        new.shift_in_place(amount, axis);
        new
    }

    pub fn shift_in_place(&mut self, amount: f64, axis: char) {
        if amount == 0.0 {
            return;
        }
        match axis.to_ascii_lowercase() {
            'x' => {
                for v in &mut self.x {
                    *v += amount;
                }
            }
            'y' => {
                for v in &mut self.y {
                    *v += amount;
                }
            }
            'z' => {
                for v in &mut self.z {
                    *v += amount;
                }
            }
            _ => {}
        }
    }

    /// Alias for [`npoints`]. Returns `imax * jmax * kmax`.
    #[inline]
    pub fn size(&self) -> usize {
        self.npoints()
    }

    /// Multiply all coordinates by `factor` in place.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.x {
            *v *= factor;
        }
        for v in &mut self.y {
            *v *= factor;
        }
        for v in &mut self.z {
            *v *= factor;
        }
    }

    /// Return a new block with coordinates scaled by `factor`.
    pub fn scaled(&self, factor: f64) -> Block {
        let mut new = self.clone();
        new.scale(factor);
        new
    }

    /// Convert to cylindrical coordinates (rotation axis = X).
    ///
    /// Returns `(r, theta)` where `r = sqrt(z^2 + y^2)` and `theta = atan2(y, z)`.
    /// Each vector has `npoints()` elements in the same index order as `x/y/z`.
    pub fn cylindrical(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.npoints();
        let mut r = Vec::with_capacity(n);
        let mut theta = Vec::with_capacity(n);
        for idx in 0..n {
            let yi = self.y[idx];
            let zi = self.z[idx];
            r.push((zi * zi + yi * yi).sqrt());
            theta.push(yi.atan2(zi));
        }
        (r, theta)
    }

    /// Compute volume of each cell using the Davies-Salmond hexahedral method.
    ///
    /// Returns a flat vector of length `imax * jmax * kmax` where cell `(i, j, k)` with
    /// `1 <= i < imax, 1 <= j < jmax, 1 <= k < kmax` stores its volume at
    /// index `(k * jmax + j) * imax + i`. Boundary entries (where any index is 0) are zero.
    ///
    /// Reference: Davies & Salmond, AIAA Journal, vol 23, No 6, pp 954-956, 1985.
    pub fn cell_volumes(&self) -> Vec<f64> {
        let (ni, nj, nk) = (self.imax, self.jmax, self.kmax);
        let n = ni * nj * nk;
        let idx = |i: usize, j: usize, k: usize| -> usize { (k * nj + j) * ni + i };

        // 9 auxiliary face-area component arrays
        let mut a = vec![vec![0.0f64; n]; 9];

        // Face csi=const (i varies freely, j and k >= 1)
        for k in 1..nk {
            for j in 1..nj {
                for i in 0..ni {
                    let dx1 = self.x[idx(i, j, k - 1)] - self.x[idx(i, j - 1, k)];
                    let dy1 = self.y[idx(i, j, k - 1)] - self.y[idx(i, j - 1, k)];
                    let dz1 = self.z[idx(i, j, k - 1)] - self.z[idx(i, j - 1, k)];

                    let dx2 = self.x[idx(i, j, k)] - self.x[idx(i, j - 1, k - 1)];
                    let dy2 = self.y[idx(i, j, k)] - self.y[idx(i, j - 1, k - 1)];
                    let dz2 = self.z[idx(i, j, k)] - self.z[idx(i, j - 1, k - 1)];

                    let id = idx(i, j, k);
                    a[0][id] = (dy1 * dz2 - dz1 * dy2) * 0.5;
                    a[1][id] = (dz1 * dx2 - dx1 * dz2) * 0.5;
                    a[2][id] = (dx1 * dy2 - dy1 * dx2) * 0.5;
                }
            }
        }

        // Face eta=const (j varies freely, i and k >= 1)
        for k in 1..nk {
            for j in 0..nj {
                for i in 1..ni {
                    let dx1 = self.x[idx(i, j, k)] - self.x[idx(i - 1, j, k - 1)];
                    let dy1 = self.y[idx(i, j, k)] - self.y[idx(i - 1, j, k - 1)];
                    let dz1 = self.z[idx(i, j, k)] - self.z[idx(i - 1, j, k - 1)];

                    let dx2 = self.x[idx(i, j, k - 1)] - self.x[idx(i - 1, j, k)];
                    let dy2 = self.y[idx(i, j, k - 1)] - self.y[idx(i - 1, j, k)];
                    let dz2 = self.z[idx(i, j, k - 1)] - self.z[idx(i - 1, j, k)];

                    let id = idx(i, j, k);
                    a[3][id] = (dy1 * dz2 - dz1 * dy2) * 0.5;
                    a[4][id] = (dz1 * dx2 - dx1 * dz2) * 0.5;
                    a[5][id] = (dx1 * dy2 - dy1 * dx2) * 0.5;
                }
            }
        }

        // Face zeta=const (k varies freely, i and j >= 1)
        for k in 0..nk {
            for j in 1..nj {
                for i in 1..ni {
                    let dx1 = self.x[idx(i, j, k)] - self.x[idx(i - 1, j - 1, k)];
                    let dy1 = self.y[idx(i, j, k)] - self.y[idx(i - 1, j - 1, k)];
                    let dz1 = self.z[idx(i, j, k)] - self.z[idx(i - 1, j - 1, k)];

                    let dx2 = self.x[idx(i - 1, j, k)] - self.x[idx(i, j - 1, k)];
                    let dy2 = self.y[idx(i - 1, j, k)] - self.y[idx(i, j - 1, k)];
                    let dz2 = self.z[idx(i - 1, j, k)] - self.z[idx(i, j - 1, k)];

                    let id = idx(i, j, k);
                    a[6][id] = (dy1 * dz2 - dz1 * dy2) * 0.5;
                    a[7][id] = (dz1 * dx2 - dx1 * dz2) * 0.5;
                    a[8][id] = (dx1 * dy2 - dy1 * dx2) * 0.5;
                }
            }
        }

        // Compute cell volumes from the 6 face centroids and face-area vectors
        let mut v = vec![0.0f64; n];
        for k in 1..nk {
            for j in 1..nj {
                for i in 1..ni {
                    // 6 face centroids (cf[face][component])
                    let mut cf = [[0.0f64; 3]; 6];
                    // Face 0: i-1 face (csi=const, low side)
                    cf[0][0] = self.x[idx(i - 1, j - 1, k - 1)] + self.x[idx(i - 1, j - 1, k)]
                        + self.x[idx(i - 1, j, k - 1)]
                        + self.x[idx(i - 1, j, k)];
                    cf[0][1] = self.y[idx(i - 1, j - 1, k - 1)] + self.y[idx(i - 1, j - 1, k)]
                        + self.y[idx(i - 1, j, k - 1)]
                        + self.y[idx(i - 1, j, k)];
                    cf[0][2] = self.z[idx(i - 1, j - 1, k - 1)] + self.z[idx(i - 1, j - 1, k)]
                        + self.z[idx(i - 1, j, k - 1)]
                        + self.z[idx(i - 1, j, k)];
                    // Face 1: i face (csi=const, high side)
                    cf[1][0] = self.x[idx(i, j - 1, k - 1)] + self.x[idx(i, j - 1, k)]
                        + self.x[idx(i, j, k - 1)]
                        + self.x[idx(i, j, k)];
                    cf[1][1] = self.y[idx(i, j - 1, k - 1)] + self.y[idx(i, j - 1, k)]
                        + self.y[idx(i, j, k - 1)]
                        + self.y[idx(i, j, k)];
                    cf[1][2] = self.z[idx(i, j - 1, k - 1)] + self.z[idx(i, j - 1, k)]
                        + self.z[idx(i, j, k - 1)]
                        + self.z[idx(i, j, k)];
                    // Face 2: j-1 face (eta=const, low side)
                    cf[2][0] = self.x[idx(i - 1, j - 1, k - 1)] + self.x[idx(i - 1, j - 1, k)]
                        + self.x[idx(i, j - 1, k - 1)]
                        + self.x[idx(i, j - 1, k)];
                    cf[2][1] = self.y[idx(i - 1, j - 1, k - 1)] + self.y[idx(i - 1, j - 1, k)]
                        + self.y[idx(i, j - 1, k - 1)]
                        + self.y[idx(i, j - 1, k)];
                    cf[2][2] = self.z[idx(i - 1, j - 1, k - 1)] + self.z[idx(i - 1, j - 1, k)]
                        + self.z[idx(i, j - 1, k - 1)]
                        + self.z[idx(i, j - 1, k)];
                    // Face 3: j face (eta=const, high side)
                    cf[3][0] = self.x[idx(i - 1, j, k - 1)] + self.x[idx(i - 1, j, k)]
                        + self.x[idx(i, j, k - 1)]
                        + self.x[idx(i, j, k)];
                    cf[3][1] = self.y[idx(i - 1, j, k - 1)] + self.y[idx(i - 1, j, k)]
                        + self.y[idx(i, j, k - 1)]
                        + self.y[idx(i, j, k)];
                    cf[3][2] = self.z[idx(i - 1, j, k - 1)] + self.z[idx(i - 1, j, k)]
                        + self.z[idx(i, j, k - 1)]
                        + self.z[idx(i, j, k)];
                    // Face 4: k-1 face (zeta=const, low side)
                    cf[4][0] = self.x[idx(i - 1, j - 1, k - 1)] + self.x[idx(i - 1, j, k - 1)]
                        + self.x[idx(i, j - 1, k - 1)]
                        + self.x[idx(i, j, k - 1)];
                    cf[4][1] = self.y[idx(i - 1, j - 1, k - 1)] + self.y[idx(i - 1, j, k - 1)]
                        + self.y[idx(i, j - 1, k - 1)]
                        + self.y[idx(i, j, k - 1)];
                    cf[4][2] = self.z[idx(i - 1, j - 1, k - 1)] + self.z[idx(i - 1, j, k - 1)]
                        + self.z[idx(i, j - 1, k - 1)]
                        + self.z[idx(i, j, k - 1)];
                    // Face 5: k face (zeta=const, high side)
                    cf[5][0] = self.x[idx(i - 1, j - 1, k)] + self.x[idx(i - 1, j, k)]
                        + self.x[idx(i, j - 1, k)]
                        + self.x[idx(i, j, k)];
                    cf[5][1] = self.y[idx(i - 1, j - 1, k)] + self.y[idx(i - 1, j, k)]
                        + self.y[idx(i, j - 1, k)]
                        + self.y[idx(i, j, k)];
                    cf[5][2] = self.z[idx(i - 1, j - 1, k)] + self.z[idx(i - 1, j, k)]
                        + self.z[idx(i, j - 1, k)]
                        + self.z[idx(i, j, k)];

                    let mut vol12 = 0.0f64;
                    for nn in 0..2usize {
                        let sign = if nn == 0 { -1.0 } else { 1.0 };
                        for l in 0..3usize {
                            // i-1+nn for csi face, j-1+nn for eta face, k-1+nn for zeta face
                            vol12 += sign
                                * (cf[nn][l] * a[l][idx(i - 1 + nn, j, k)]
                                    + cf[2 + nn][l] * a[3 + l][idx(i, j - 1 + nn, k)]
                                    + cf[4 + nn][l] * a[6 + l][idx(i, j, k - 1 + nn)]);
                        }
                    }
                    v[idx(i, j, k)] = vol12 / 12.0;
                }
            }
        }
        v
    }

    /// Return the six boundary faces as a map keyed by face name.
    ///
    /// Keys: `"imin"`, `"imax"`, `"jmin"`, `"jmax"`, `"kmin"`, `"kmax"`.
    /// Each [`FaceData`] contains the X, Y, Z coordinates of all nodes on that face,
    /// stored with the two varying indices in their natural order.
    pub fn get_faces(&self) -> HashMap<&'static str, FaceData> {
        let mut map = HashMap::with_capacity(6);

        // Helper to extract a face by iterating over the two free dimensions.
        let extract =
            |fix_axis: usize, fix_val: usize| -> FaceData {
                let (nu, nv) = match fix_axis {
                    0 => (self.jmax, self.kmax), // i-const: j varies first, k second
                    1 => (self.imax, self.kmax), // j-const: i varies first, k second
                    _ => (self.imax, self.jmax), // k-const: i varies first, j second
                };
                let cap = nu * nv;
                let mut xf = Vec::with_capacity(cap);
                let mut yf = Vec::with_capacity(cap);
                let mut zf = Vec::with_capacity(cap);
                for v in 0..nv {
                    for u in 0..nu {
                        let (i, j, k) = match fix_axis {
                            0 => (fix_val, u, v),
                            1 => (u, fix_val, v),
                            _ => (u, v, fix_val),
                        };
                        let id = self.idx(i, j, k);
                        xf.push(self.x[id]);
                        yf.push(self.y[id]);
                        zf.push(self.z[id]);
                    }
                }
                FaceData { x: xf, y: yf, z: zf, dims: (nu, nv) }
            };

        map.insert("imin", extract(0, 0));
        map.insert("imax", extract(0, self.imax - 1));
        map.insert("jmin", extract(1, 0));
        map.insert("jmax", extract(1, self.jmax - 1));
        map.insert("kmin", extract(2, 0));
        map.insert("kmax", extract(2, self.kmax - 1));
        map
    }

    /// Extract a sub-block defined by inclusive index ranges.
    pub fn sub_block(
        &self,
        i_range: std::ops::RangeInclusive<usize>,
        j_range: std::ops::RangeInclusive<usize>,
        k_range: std::ops::RangeInclusive<usize>,
    ) -> Block {
        let ni = i_range.end() - i_range.start() + 1;
        let nj = j_range.end() - j_range.start() + 1;
        let nk = k_range.end() - k_range.start() + 1;
        let cap = ni * nj * nk;
        let mut x = Vec::with_capacity(cap);
        let mut y = Vec::with_capacity(cap);
        let mut z = Vec::with_capacity(cap);
        for k in k_range {
            for j in j_range.clone() {
                for i in i_range.clone() {
                    let id = self.idx(i, j, k);
                    x.push(self.x[id]);
                    y.push(self.y[id]);
                    z.push(self.z[id]);
                }
            }
        }
        Block::new(ni, nj, nk, x, y, z)
    }
}
