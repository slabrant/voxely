use super::Voxel;

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_SIZE_CUBED: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub struct Chunk {
    pub voxels: Box<[Voxel; CHUNK_SIZE_CUBED]>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: Box::new([Voxel::default(); CHUNK_SIZE_CUBED]),
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&Voxel> {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return None;
        }
        Some(&self.voxels[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE])
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) -> bool {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return false;
        }
        self.voxels[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE] = voxel;
        true
    }
}
