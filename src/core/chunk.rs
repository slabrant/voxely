use super::Voxel;

/// Default canvas dimensions for a fresh chunk.
pub const DEFAULT_CHUNK_SIZE: usize = 16;
/// Hard cap on any single dimension, so the UI can't request a multi-gigabyte
/// allocation by accident.
pub const MAX_CHUNK_SIZE: usize = 256;

/// A dense voxel grid. Dimensions are chosen at runtime (and changeable from the
/// UI), so they live as fields rather than compile-time constants.
pub struct Chunk {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub voxels: Vec<Voxel>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    pub fn new() -> Self {
        Self::with_size(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)
    }

    pub fn with_size(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            voxels: vec![Voxel::default(); width * height * depth],
        }
    }

    pub fn volume(&self) -> usize {
        self.width * self.height * self.depth
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.width + z * self.width * self.height
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&Voxel> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return None;
        }
        let i = self.index(x, y, z);
        Some(&self.voxels[i])
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) -> bool {
        if x >= self.width || y >= self.height || z >= self.depth {
            return false;
        }
        let i = self.index(x, y, z);
        self.voxels[i] = voxel;
        true
    }

    /// Returns a copy resized to `width`x`height`x`depth`, preserving every
    /// voxel that still falls inside the new bounds (anchored at the origin).
    pub fn resized(&self, width: usize, height: usize, depth: usize) -> Chunk {
        let mut out = Chunk::with_size(width, height, depth);
        for z in 0..depth.min(self.depth) {
            for y in 0..height.min(self.height) {
                for x in 0..width.min(self.width) {
                    if let Some(v) = self.get(x, y, z) {
                        out.set(x, y, z, *v);
                    }
                }
            }
        }
        out
    }
}
