use crate::core::Chunk;
use crate::core::chunk::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH};
use crate::core::Palette;
use crate::state::Vertex;

pub fn mesh_chunk(chunk: &Chunk, palette: &Palette) -> (Vec<Vertex>, Vec<u32>) {
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

        let dim_axis = match axis { 0 => CHUNK_WIDTH, 1 => CHUNK_HEIGHT, 2 => CHUNK_DEPTH, _ => unreachable!() };
        let dim_u = match u { 0 => CHUNK_WIDTH, 1 => CHUNK_HEIGHT, 2 => CHUNK_DEPTH, _ => unreachable!() };
        let dim_v = match v { 0 => CHUNK_WIDTH, 1 => CHUNK_HEIGHT, 2 => CHUNK_DEPTH, _ => unreachable!() };

        // Mask for the current slice
        let mut mask = vec![None; dim_u * dim_v];

        for i in -1..dim_axis as i32 {
            x[axis] = i;

            // Compute mask
            for j in 0..dim_u {
                x[u] = j as i32;
                for k in 0..dim_v {
                    x[v] = k as i32;

                    let voxel_a = if i >= 0 { chunk.get(x[0] as usize, x[1] as usize, x[2] as usize) } else { None };
                    let voxel_b = if i < (dim_axis as i32 - 1) { chunk.get((x[0] + q[0]) as usize, (x[1] + q[1]) as usize, (x[2] + q[2]) as usize) } else { None };

                    let a_active = voxel_a.map(|v| !v.is_empty()).unwrap_or(false);
                    let b_active = voxel_b.map(|v| !v.is_empty()).unwrap_or(false);

                    if a_active != b_active {
                        mask[j + k * dim_u] = if a_active {
                            Some((voxel_a.unwrap().color_index, 1)) // 1 for +direction
                        } else {
                            Some((voxel_b.unwrap().color_index, -1)) // -1 for -direction
                        };
                    } else {
                        mask[j + k * dim_u] = None;
                    }
                }
            }

            // Generate mesh from mask
            for k in 0..dim_v {
                let mut j = 0;
                while j < dim_u {
                    if let Some((color_idx, dir)) = mask[j + k * dim_u] {
                        let mut width = 1;
                        while j + width < dim_u && mask[j + width + k * dim_u] == Some((color_idx, dir)) {
                            width += 1;
                        }

                        let mut height = 1;
                        let mut done = false;
                        while k + height < dim_v {
                            for m in 0..width {
                                if mask[j + m + (k + height) * dim_u] != Some((color_idx, dir)) {
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

                        // The face lies on the boundary plane between slice `i`
                        // and `i+1`, i.e. at `x[axis] + 1`, no matter which way it
                        // faces. (`q` is 1 on `axis`, 0 elsewhere.)
                        add_quad(
                            &mut vertices,
                            &mut indices,
                            [x[0] as f32 + q[0] as f32,
                             x[1] as f32 + q[1] as f32,
                             x[2] as f32 + q[2] as f32],
                            [du[0] as f32, du[1] as f32, du[2] as f32],
                            [dv[0] as f32, dv[1] as f32, dv[2] as f32],
                            palette.linear_rgb(color_idx),
                            normal,
                            dir > 0
                        );

                        // Clear mask
                        for m in 0..height {
                            for n in 0..width {
                                mask[j + n + (k + m) * dim_u] = None;
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
    indices: &mut Vec<u32>,
    pos: [f32; 3],
    du: [f32; 3],
    dv: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    backwards: bool,
) {
    let start_idx = vertices.len() as u32;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Voxel;

    #[test]
    fn single_voxel_meshes_to_its_own_unit_cube() {
        // An isolated voxel at (p) must produce exactly six faces whose every
        // vertex lies on the cube [p, p+1]^3. This guards against the
        // negative-face off-by-one (which placed faces at p-1).
        let p = 5.0;
        let mut chunk = Chunk::new();
        chunk.set(5, 5, 5, Voxel { color_index: 1 });

        let (vertices, indices) = mesh_chunk(&chunk, &Palette::default());

        assert_eq!(vertices.len(), 24, "6 faces * 4 verts");
        assert_eq!(indices.len(), 36, "6 faces * 2 tris * 3");
        for v in &vertices {
            for &c in &v.position {
                assert!(
                    (p..=p + 1.0).contains(&c),
                    "vertex coord {c} escaped the [{p}, {}] cube",
                    p + 1.0
                );
            }
        }
    }
}
