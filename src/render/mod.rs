use crate::core::Chunk;
use crate::core::Palette;
use crate::state::Vertex;

/// One entry of a slice mask: `0` means "no face here", otherwise the low byte
/// is the color index (never 0 for a face, since color 0 *is* empty) and bit 8
/// records which way the face points. Packing it into a `u16` rather than an
/// `Option<(u8, i32)>` keeps the mask a quarter of the size, which matters
/// because the greedy pass sweeps it repeatedly.
type MaskCell = u16;
const MASK_EMPTY: MaskCell = 0;
const MASK_POSITIVE: MaskCell = 1 << 8;

pub fn mesh_chunk(chunk: &Chunk, palette: &Palette) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let dims = [chunk.width, chunk.height, chunk.depth];
    // Linear index of a voxel is `sum(coord[a] * stride[a])`, so walking a slice
    // is pure integer arithmetic on these — no per-cell bounds check or index
    // recomputation, which is most of the cost at large canvas sizes.
    let strides = [1usize, chunk.width, chunk.width * chunk.height];
    let voxels = &chunk.voxels;

    // Faces are found one axis at a time: for each pair of adjacent slices, a
    // face exists wherever exactly one side is solid.
    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        let (dim_axis, dim_u, dim_v) = (dims[axis], dims[u], dims[v]);
        let (s_axis, s_u, s_v) = (strides[axis], strides[u], strides[v]);

        // Reused across every slice of this axis; the mask loop below writes
        // each cell unconditionally, so there is nothing to clear between them.
        let mut mask = vec![MASK_EMPTY; dim_u * dim_v];

        // `i == -1` and `i == dim_axis - 1` cover the outer faces of the canvas,
        // where the neighbouring slice is off the grid and reads as empty.
        for i in -1..dim_axis as i32 {
            let base_a = if i >= 0 { Some(i as usize * s_axis) } else { None };
            let base_b = if i + 1 < dim_axis as i32 { Some((i + 1) as usize * s_axis) } else { None };

            // Exactly one side solid => a visible face, pointing away from
            // whichever side holds the voxel.
            let face_at = |off: usize| -> MaskCell {
                let a = base_a.map_or(0, |b| voxels[b + off].color_index);
                let b = base_b.map_or(0, |b| voxels[b + off].color_index);
                match (a != 0, b != 0) {
                    (true, false) => a as MaskCell | MASK_POSITIVE,
                    (false, true) => b as MaskCell,
                    _ => MASK_EMPTY,
                }
            };

            // Sweep the slice with the *smaller* voxel stride innermost. Which
            // of `u`/`v` that is depends on the axis, and getting it wrong costs
            // a cache miss per cell: for `axis == 1`, `v` walks the grid one
            // voxel at a time while `u` jumps a whole `width * height` plane.
            if s_u <= s_v {
                for k in 0..dim_v {
                    let (kv, row) = (k * s_v, k * dim_u);
                    for j in 0..dim_u {
                        mask[row + j] = face_at(kv + j * s_u);
                    }
                }
            } else {
                for j in 0..dim_u {
                    let ju = j * s_u;
                    for k in 0..dim_v {
                        mask[j + k * dim_u] = face_at(ju + k * s_v);
                    }
                }
            }

            // Merge equal cells into maximal rectangles, widest-first.
            for k in 0..dim_v {
                let mut j = 0;
                while j < dim_u {
                    let cell = mask[j + k * dim_u];
                    if cell == MASK_EMPTY {
                        j += 1;
                        continue;
                    }

                    let mut width = 1;
                    while j + width < dim_u && mask[j + width + k * dim_u] == cell {
                        width += 1;
                    }

                    let mut height = 1;
                    'grow: while k + height < dim_v {
                        for m in 0..width {
                            if mask[j + m + (k + height) * dim_u] != cell {
                                break 'grow;
                            }
                        }
                        height += 1;
                    }

                    let positive = cell & MASK_POSITIVE != 0;
                    let color_idx = (cell & 0xFF) as u8;

                    let mut x = [0i32; 3];
                    x[axis] = i;
                    x[u] = j as i32;
                    x[v] = k as i32;
                    let mut du = [0; 3];
                    du[u] = width as i32;
                    let mut dv = [0; 3];
                    dv[v] = height as i32;

                    let mut normal = [0.0; 3];
                    normal[axis] = if positive { 1.0 } else { -1.0 };

                    // The face lies on the boundary plane between slice `i` and
                    // `i+1`, i.e. at `x[axis] + 1`, no matter which way it faces.
                    let mut pos = [x[0] as f32, x[1] as f32, x[2] as f32];
                    pos[axis] += 1.0;

                    add_quad(
                        &mut vertices,
                        &mut indices,
                        pos,
                        [du[0] as f32, du[1] as f32, du[2] as f32],
                        [dv[0] as f32, dv[1] as f32, dv[2] as f32],
                        palette.linear_rgb(color_idx),
                        normal,
                        positive,
                    );

                    // Consume the rectangle so its cells aren't emitted again.
                    for m in 0..height {
                        let row = (k + m) * dim_u;
                        mask[row + j..row + j + width].fill(MASK_EMPTY);
                    }
                    j += width;
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
