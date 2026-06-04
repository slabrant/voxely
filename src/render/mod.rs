use crate::core::Chunk;
use crate::core::chunk::CHUNK_SIZE;
use crate::state::Vertex;

fn get_color(index: u8) -> [f32; 3] {
    match index {
        1 => [0.8, 0.1, 0.1], // Red
        2 => [0.1, 0.8, 0.1], // Green
        3 => [0.1, 0.1, 0.8], // Blue
        4 => [0.8, 0.8, 0.1], // Yellow
        _ => [1.0, 1.0, 1.0], // White
    }
}

pub fn mesh_chunk(chunk: &Chunk) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Each face can be defined by its orientation (axis and direction)
    // 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let mut x = [0; 3];
        let mut q = [0; 3];
        q[axis] = 1;

        // Mask for the current slice
        let mut mask = [None; CHUNK_SIZE * CHUNK_SIZE];

        for i in -1..CHUNK_SIZE as i32 {
            x[axis] = i;

            // Compute mask
            for j in 0..CHUNK_SIZE {
                x[u] = j as i32;
                for k in 0..CHUNK_SIZE {
                    x[v] = k as i32;

                    let voxel_a = if i >= 0 { chunk.get(x[0] as usize, x[1] as usize, x[2] as usize) } else { None };
                    let voxel_b = if i < (CHUNK_SIZE as i32 - 1) { chunk.get((x[0] + q[0]) as usize, (x[1] + q[1]) as usize, (x[2] + q[2]) as usize) } else { None };

                    let a_active = voxel_a.map(|v| !v.is_empty()).unwrap_or(false);
                    let b_active = voxel_b.map(|v| !v.is_empty()).unwrap_or(false);

                    if a_active != b_active {
                        mask[j + k * CHUNK_SIZE] = if a_active {
                            Some((voxel_a.unwrap().color_index, 1)) // 1 for +direction
                        } else {
                            Some((voxel_b.unwrap().color_index, -1)) // -1 for -direction
                        };
                    } else {
                        mask[j + k * CHUNK_SIZE] = None;
                    }
                }
            }

            // Generate mesh from mask
            for k in 0..CHUNK_SIZE {
                let mut j = 0;
                while j < CHUNK_SIZE {
                    if let Some((color_idx, dir)) = mask[j + k * CHUNK_SIZE] {
                        let mut width = 1;
                        while j + width < CHUNK_SIZE && mask[j + width + k * CHUNK_SIZE] == Some((color_idx, dir)) {
                            width += 1;
                        }

                        let mut height = 1;
                        let mut done = false;
                        while k + height < CHUNK_SIZE {
                            for m in 0..width {
                                if mask[j + m + (k + height) * CHUNK_SIZE] != Some((color_idx, dir)) {
                                    done = true;
                                    break;
                                }
                            }
                            if done { break; }
                            height += 1;
                        }

                        // Add quad
                        x[u] = j as i32;
                        x[v] = k as i32;
                        let mut du = [0; 3]; du[u] = width as i32;
                        let mut dv = [0; 3]; dv[v] = height as i32;
                        
                        let mut normal = [0.0; 3];
                        normal[axis] = if dir > 0 { 1.0 } else { -1.0 };

                        add_quad(
                            &mut vertices,
                            &mut indices,
                            [x[0] as f32 + (if dir > 0 { q[0] } else { 0 }) as f32, 
                             x[1] as f32 + (if dir > 0 { q[1] } else { 0 }) as f32, 
                             x[2] as f32 + (if dir > 0 { q[2] } else { 0 }) as f32],
                            [du[0] as f32, du[1] as f32, du[2] as f32],
                            [dv[0] as f32, dv[1] as f32, dv[2] as f32],
                            get_color(color_idx),
                            normal,
                            dir > 0
                        );

                        // Clear mask
                        for m in 0..height {
                            for n in 0..width {
                                mask[j + n + (k + m) * CHUNK_SIZE] = None;
                            }
                        }
                        j += width;
                    } else {
                        j += 1;
                    }
                }
            }
        }
    }

    (vertices, indices)
}

fn add_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u16>,
    pos: [f32; 3],
    du: [f32; 3],
    dv: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    backwards: bool,
) {
    let start_idx = vertices.len() as u16;

    vertices.push(Vertex { position: pos, color, normal });
    vertices.push(Vertex { position: [pos[0] + du[0], pos[1] + du[1], pos[2] + du[2]], color, normal });
    vertices.push(Vertex { position: [pos[0] + du[0] + dv[0], pos[1] + du[1] + dv[1], pos[2] + du[2] + dv[2]], color, normal });
    vertices.push(Vertex { position: [pos[0] + dv[0], pos[1] + dv[1], pos[2] + dv[2]], color, normal });

    if backwards {
        indices.extend_from_slice(&[start_idx, start_idx + 2, start_idx + 1, start_idx, start_idx + 3, start_idx + 2]);
    } else {
        indices.extend_from_slice(&[start_idx, start_idx + 1, start_idx + 2, start_idx, start_idx + 2, start_idx + 3]);
    }
}
