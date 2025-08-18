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
    pub fn new(imax: usize, jmax: usize, kmax: usize, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Self {
        let n = imax * jmax * kmax;
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        assert_eq!(z.len(), n);
        Self { imax, jmax, kmax, x, y, z }
    }

    #[inline]
    pub fn npoints(&self) -> usize { self.imax * self.jmax * self.kmax }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        // i–j–k order (i fastest)
        debug_assert!(i < self.imax && j < self.jmax && k < self.kmax);
        (k * self.jmax + j) * self.imax + i
    }
}
