use super::Voxel;
use serde::{Serialize, Deserialize};

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 16;
pub const CHUNK_DEPTH: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH;

#[derive(Serialize, Deserialize)]
pub struct Chunk {
    pub voxels: Vec<Voxel>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: vec![Voxel::default(); CHUNK_VOLUME],
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&Voxel> {
        if x >= CHUNK_WIDTH || y >= CHUNK_HEIGHT || z >= CHUNK_DEPTH {
            return None;
        }
        Some(&self.voxels[x + y * CHUNK_WIDTH + z * CHUNK_WIDTH * CHUNK_HEIGHT])
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) -> bool {
        if x >= CHUNK_WIDTH || y >= CHUNK_HEIGHT || z >= CHUNK_DEPTH {
            return false;
        }
        self.voxels[x + y * CHUNK_WIDTH + z * CHUNK_WIDTH * CHUNK_HEIGHT] = voxel;
        true
    }
}
