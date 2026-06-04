use crate::core::Chunk;
use crate::state::Vertex;

pub fn mesh_chunk(chunk: &Chunk) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for x in 0..crate::core::chunk::CHUNK_SIZE {
        for y in 0..crate::core::chunk::CHUNK_SIZE {
            for z in 0..crate::core::chunk::CHUNK_SIZE {
                let voxel = chunk.get(x, y, z).unwrap();
                if voxel.is_empty() {
                    continue;
                }

                let pos = [x as f32, y as f32, z as f32];
                let color = [1.0, 1.0, 1.0]; // Temporary solid white

                // Add a simple cube for each voxel (Naive Meshing)
                add_cube(&mut vertices, &mut indices, pos, color);
            }
        }
    }

    (vertices, indices)
}

fn add_cube(vertices: &mut Vec<Vertex>, indices: &mut Vec<u16>, pos: [f32; 3], color: [f32; 3]) {
    let start_idx = vertices.len() as u16;
    let offsets = [
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], // Front
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], // Back
    ];

    for offset in offsets {
        vertices.push(Vertex {
            position: [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]],
            color,
        });
    }

    let cube_indices = [
        // Front
        0, 1, 2, 2, 3, 0,
        // Right
        1, 5, 6, 6, 2, 1,
        // Back
        5, 4, 7, 7, 6, 5,
        // Left
        4, 0, 3, 3, 7, 4,
        // Top
        3, 2, 6, 6, 7, 3,
        // Bottom
        4, 5, 1, 1, 0, 4,
    ];

    for idx in cube_indices {
        indices.push(start_idx + idx);
    }
}
