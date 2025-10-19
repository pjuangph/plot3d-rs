#[derive(Clone, Debug)]
pub struct Block {
    pub imax: usize,
    pub jmax: usize,
    pub kmax: usize, // 2D supported via kmax == 1
    pub x: Vec<f64>, // length = imax*jmax*kmax
    pub y: Vec<f64>,
    pub z: Vec<f64>,
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
}
