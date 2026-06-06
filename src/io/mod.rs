use std::error::Error;
use std::fmt::Write as _;
use std::path::Path;

use crate::core::{Chunk, Palette, Voxel};

/// An opened model: the voxel grid plus its palette. This is what every loader
/// returns and what the editor swaps in.
pub struct Project {
    pub chunk: Chunk,
    pub palette: Palette,
}

/// Open a model file. Only the native, lossless `.vox` format (voxel grid +
/// palette) can be opened; `.obj` is export-only.
pub fn open(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("vox") => load_vox(path),
        other => Err(format!(
            "unsupported file type: {} (only .vox can be opened)",
            other.unwrap_or("(none)")
        )
        .into()),
    }
}

/// Save the chunk and palette losslessly to a MagicaVoxel `.vox` file. This is
/// the native project format: it stores the voxel grid and real color palette,
/// round-trips through [`load_vox`], and opens in MagicaVoxel and most engines.
///
/// We are Y-up while `.vox` is Z-up, so engine `(x, y, z)` is written as vox
/// `(x, z, y)` — the inverse of the swap [`load_vox`] applies on the way in.
pub fn save_vox(path: impl AsRef<Path>, chunk: &Chunk, palette: &Palette) -> Result<(), Box<dyn Error>> {
    // `.vox` coordinates are single bytes, so each dimension must fit in 0..=256
    // (indices 0..=255). The editor caps dimensions well below this already.
    if chunk.width > 256 || chunk.height > 256 || chunk.depth > 256 {
        return Err("chunk is too large to save as .vox (max 256 per axis)".into());
    }

    let mut voxels = Vec::new();
    for x in 0..chunk.width {
        for y in 0..chunk.height {
            for z in 0..chunk.depth {
                if let Some(v) = chunk.get(x, y, z) {
                    if !v.is_empty() {
                        // engine (x, y, z) Y-up -> vox (x, z, y) Z-up; color
                        // index 1..=255 becomes vox 0-based `i` = index - 1.
                        voxels.push(dot_vox::Voxel {
                            x: x as u8,
                            y: z as u8,
                            z: y as u8,
                            i: v.color_index - 1,
                        });
                    }
                }
            }
        }
    }

    // 256-color RGBA table written 1:1 with our palette. `load_vox` (mirroring
    // dot_vox's default index map, which maps in-memory index i -> slot i + 1)
    // reads our color index C straight back from `data.palette[C]`, so writing
    // slot-for-slot makes the palette round-trip exactly.
    let pal = (0..256)
        .map(|k| {
            let c = palette.colors.get(k).copied().unwrap_or([0, 0, 0, 255]);
            dot_vox::Color { r: c[0], g: c[1], b: c[2], a: c[3] }
        })
        .collect();

    let data = dot_vox::DotVoxData {
        version: 150,
        index_map: Vec::new(),
        models: vec![dot_vox::Model {
            size: dot_vox::Size {
                x: chunk.width as u32,
                y: chunk.depth as u32,
                z: chunk.height as u32,
            },
            voxels,
        }],
        palette: pal,
        materials: Vec::new(),
        scenes: Vec::new(),
        layers: Vec::new(),
    };

    let mut file = std::fs::File::create(path)?;
    data.write_vox(&mut file)?;
    Ok(())
}

/// Load a MagicaVoxel `.vox` file, bringing along its real color palette.
/// MagicaVoxel is Z-up, so Y and Z are swapped to match our Y-up engine. Color
/// indices are shifted by one so index 0 stays "empty".
pub fn load_vox(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    let data = dot_vox::load(path.to_str().ok_or("non-UTF-8 path")?)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let model = data.models.first().ok_or("the .vox file has no models")?;

    let mut palette = Palette::default();
    // dot_vox stores in-memory indices 0..=254; remap through index_map (if
    // present) and shift into slots 1..=255 to keep slot 0 reserved for empty.
    for i in 0..255usize {
        let src = data.index_map.get(i).map(|&m| m as usize).unwrap_or(i);
        if let Some(c) = data.palette.get(src) {
            palette.colors[i + 1] = [c.r, c.g, c.b, c.a];
        }
    }

    // Size the chunk to the model so nothing is silently clipped.
    let mut chunk = Chunk::with_size(
        (model.size.x as usize).max(1),
        (model.size.z as usize).max(1),
        (model.size.y as usize).max(1),
    );
    let mut dropped = 0usize;
    for v in &model.voxels {
        let color_index = v.i.saturating_add(1);
        // (x, y, z) MagicaVoxel -> (x, z, y) engine (Z-up to Y-up).
        if !chunk.set(v.x as usize, v.z as usize, v.y as usize, Voxel { color_index }) {
            dropped += 1;
        }
    }
    if dropped > 0 {
        log::warn!(
            "{dropped} voxels were outside the {}x{}x{} chunk and were dropped on load",
            chunk.width, chunk.height, chunk.depth
        );
    }

    Ok(Project { chunk, palette })
}

/// Export the chunk to a Wavefront `.obj` file (plus a companion `.mtl` with one
/// material per used color).
///
/// The mesh is simplified before it is written. First, every face shared between
/// two solid voxels is dropped — it is sealed inside the model and never visible.
/// Then the remaining exposed faces are greedily merged: coplanar faces of the
/// same color are combined into the largest rectangles possible (greedy
/// meshing). The surface is unchanged, but a solid block exports as six quads
/// instead of one cube per voxel.
///
/// Each emitted quad still carries an explicit flat normal and never shares
/// vertices with another quad, so importers (e.g. Godot) keep the blocky flat
/// shading instead of auto-smoothing across faces.
pub fn export_obj(path: impl AsRef<Path>, chunk: &Chunk, palette: &Palette) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    let mtl_path = path.with_extension("mtl");
    let mtl_name = mtl_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("invalid .mtl path")?
        .to_string();

    // Color of the voxel at integer coords, or None if empty / out of bounds. A
    // face's outward neighbor can sit at -1 or past an edge; both read as empty.
    let solid = |c: [i32; 3]| -> Option<u8> {
        if c.iter().any(|&v| v < 0) {
            return None;
        }
        chunk
            .get(c[0] as usize, c[1] as usize, c[2] as usize)
            .filter(|v| !v.is_empty())
            .map(|v| v.color_index)
    };

    // Occupied bounding box, so the model can be re-origined to its bottom-center.
    let (mut min, mut max) = ([usize::MAX; 3], [usize::MIN; 3]);
    let mut any = false;
    for x in 0..chunk.width {
        for y in 0..chunk.height {
            for z in 0..chunk.depth {
                if chunk.get(x, y, z).is_some_and(|v| !v.is_empty()) {
                    any = true;
                    for (i, &c) in [x, y, z].iter().enumerate() {
                        min[i] = min[i].min(c);
                        max[i] = max[i].max(c);
                    }
                }
            }
        }
    }

    // Move the origin to the bottom-center: X/Z centered on the occupied volume,
    // Y resting on its lowest layer. Empty models fall back to no shift.
    let origin = if any {
        [
            (min[0] + max[0] + 1) as f32 / 2.0,
            min[1] as f32,
            (min[2] + max[2] + 1) as f32 / 2.0,
        ]
    } else {
        [0.0; 3]
    };

    // World-units emitted per voxel. OBJ is unitless but importers (Godot) treat
    // one unit as one metre, so a unit cube per voxel gives a 1 m block.
    const VOXEL_SIZE_M: f32 = 1.0;

    // The six face directions, in the order their normals are written below.
    // `axis` is the axis the face points along (0=x, 1=y, 2=z) and `sign` the
    // direction; `u`/`v` are the two in-plane axes, ordered so the quad winds
    // counter-clockwise when viewed from outside (its cross product is `normal`).
    struct Dir {
        normal: [i32; 3],
        axis: usize,
        sign: i32,
        u: usize,
        v: usize,
    }
    const DIRS: [Dir; 6] = [
        Dir { normal: [-1, 0, 0], axis: 0, sign: -1, u: 2, v: 1 }, // -X (left)
        Dir { normal: [1, 0, 0],  axis: 0, sign: 1,  u: 1, v: 2 }, // +X (right)
        Dir { normal: [0, -1, 0], axis: 1, sign: -1, u: 0, v: 2 }, // -Y (bottom)
        Dir { normal: [0, 1, 0],  axis: 1, sign: 1,  u: 2, v: 0 }, // +Y (top)
        Dir { normal: [0, 0, -1], axis: 2, sign: -1, u: 1, v: 0 }, // -Z (front)
        Dir { normal: [0, 0, 1],  axis: 2, sign: 1,  u: 0, v: 1 }, // +Z (back)
    ];

    let dims = [chunk.width, chunk.height, chunk.depth];

    // One quad of the simplified mesh: its color, the 1-based normal index, and
    // four corners in grid coordinates (CCW, outward-facing).
    struct Quad {
        color: u8,
        normal_idx: usize,
        corners: [[i32; 3]; 4],
    }
    let mut quads: Vec<Quad> = Vec::new();

    for (ni, d) in DIRS.iter().enumerate() {
        let (na, nu, nv) = (d.axis, d.u, d.v);
        let (du, dv) = (dims[nu], dims[nv]);

        // Sweep each grid layer perpendicular to this face's axis.
        for layer in 0..dims[na] {
            // Exposed-face mask for the layer: Some(color) where the voxel is
            // solid and its outward neighbor (layer + sign) is not — i.e. the
            // face is visible. Indexed [u + v * du].
            let mut mask = vec![None; du * dv];
            for j in 0..dv {
                for i in 0..du {
                    let mut cell = [0i32; 3];
                    cell[na] = layer as i32;
                    cell[nu] = i as i32;
                    cell[nv] = j as i32;
                    let Some(color) = solid(cell) else { continue };
                    let mut neighbor = cell;
                    neighbor[na] += d.sign;
                    if solid(neighbor).is_none() {
                        mask[i + j * du] = Some(color);
                    }
                }
            }

            // Greedily merge equal-color cells into maximal rectangles.
            for j in 0..dv {
                let mut i = 0;
                while i < du {
                    let Some(color) = mask[i + j * du] else {
                        i += 1;
                        continue;
                    };
                    // Widen along u while the color matches.
                    let mut w = 1;
                    while i + w < du && mask[i + w + j * du] == Some(color) {
                        w += 1;
                    }
                    // Grow along v while every cell of the candidate row matches.
                    let mut h = 1;
                    'grow: while j + h < dv {
                        for k in 0..w {
                            if mask[i + k + (j + h) * du] != Some(color) {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    // Consume the rectangle so its cells aren't emitted again.
                    for dj in 0..h {
                        for di in 0..w {
                            mask[i + di + (j + dj) * du] = None;
                        }
                    }

                    // The face plane sits at `layer` for a negative normal and
                    // `layer + 1` for a positive one.
                    let p = layer as i32 + (d.sign > 0) as i32;
                    let corner = |uu: usize, vv: usize| {
                        let mut c = [0i32; 3];
                        c[na] = p;
                        c[nu] = uu as i32;
                        c[nv] = vv as i32;
                        c
                    };
                    quads.push(Quad {
                        color,
                        normal_idx: ni + 1,
                        corners: [
                            corner(i, j),
                            corner(i + w, j),
                            corner(i + w, j + h),
                            corner(i, j + h),
                        ],
                    });

                    i += w;
                }
            }
        }
    }

    // Group quads by color so each material is named once and faces stay valid.
    let mut by_color: std::collections::BTreeMap<u8, Vec<&Quad>> = std::collections::BTreeMap::new();
    for q in &quads {
        by_color.entry(q.color).or_default().push(q);
    }

    let mut obj = String::new();
    writeln!(obj, "# Exported by Voxely")?;
    writeln!(obj, "mtllib {mtl_name}")?;
    for d in &DIRS {
        writeln!(obj, "vn {} {} {}", d.normal[0], d.normal[1], d.normal[2])?;
    }

    let mut vbase = 0u32;
    for (&color, group) in &by_color {
        writeln!(obj, "usemtl mat{color}")?;
        for q in group {
            for c in q.corners {
                writeln!(
                    obj,
                    "v {} {} {}",
                    (c[0] as f32 - origin[0]) * VOXEL_SIZE_M,
                    (c[1] as f32 - origin[1]) * VOXEL_SIZE_M,
                    (c[2] as f32 - origin[2]) * VOXEL_SIZE_M
                )?;
            }
            let n = q.normal_idx; // vn indices are global and 1-based
            writeln!(
                obj,
                "f {0}//{n} {1}//{n} {2}//{n} {3}//{n}",
                vbase + 1, vbase + 2, vbase + 3, vbase + 4
            )?;
            vbase += 4;
        }
    }

    let mut mtl = String::from("# Exported by Voxely\n");
    for &color in by_color.keys() {
        let c = palette.colors.get(color as usize).copied().unwrap_or([255, 255, 255, 255]);
        writeln!(mtl, "newmtl mat{color}")?;
        writeln!(
            mtl,
            "Kd {:.4} {:.4} {:.4}",
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0
        )?;
    }

    std::fs::write(path, obj)?;
    std::fs::write(&mtl_path, mtl)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_culls_hidden_faces_of_a_lone_voxel() {
        let mut chunk = Chunk::new();
        chunk.set(2, 3, 4, Voxel { color_index: 1 });
        let palette = Palette::default();

        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_export.obj");
        export_obj(&obj_path, &chunk, &palette).expect("export should succeed");

        let obj = std::fs::read_to_string(&obj_path).unwrap();
        // An isolated voxel exposes all six faces; merging can't combine them, so
        // we get six quads, each with its own four (unshared) vertices.
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 6, "six exposed faces");
        assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), 24, "6 quads x 4 verts");
        assert!(obj.contains("usemtl mat1"));

        let mtl = std::fs::read_to_string(dir.join("voxely_test_export.mtl")).unwrap();
        assert!(mtl.contains("newmtl mat1"));

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(dir.join("voxely_test_export.mtl"));
    }

    #[test]
    fn export_greedy_merges_and_culls() {
        // A solid 2x2x2 block: every interior face is sealed and culled, and each
        // of the six outward sides is a flat 2x2 surface of one color, so greedy
        // meshing collapses the whole thing to six quads (24 verts) instead of
        // eight cubes (48 faces / 192 verts).
        let mut chunk = Chunk::with_size(4, 4, 4);
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    chunk.set(x, y, z, Voxel { color_index: 1 });
                }
            }
        }

        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_greedy.obj");
        export_obj(&obj_path, &chunk, &Palette::default()).expect("export should succeed");
        let obj = std::fs::read_to_string(&obj_path).unwrap();

        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 6, "2x2x2 block = 6 merged quads");
        assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), 24, "6 quads x 4 verts");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(obj_path.with_extension("mtl"));
    }

    #[test]
    fn export_reorigins_to_bottom_center_with_flat_normals() {
        // Two voxels stacked at x=4,z=6 over y=2..=3. Occupied box is a single
        // column, so X/Z center on the cell centers (4.5, 6.5) and Y rests on
        // the lowest layer (2). The shared face between them is culled and each
        // side merges into one 1x2 quad.
        let mut chunk = Chunk::new();
        chunk.set(4, 2, 6, Voxel { color_index: 1 });
        chunk.set(4, 3, 6, Voxel { color_index: 1 });

        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_origin.obj");
        export_obj(&obj_path, &chunk, &Palette::default()).expect("export should succeed");
        let obj = std::fs::read_to_string(&obj_path).unwrap();

        // Six face normals are declared up front.
        assert_eq!(obj.lines().filter(|l| l.starts_with("vn ")).count(), 6);
        // Faces reference a normal, which forces flat shading on import.
        assert!(obj.lines().any(|l| l.starts_with("f ") && l.contains("//")));

        let verts: Vec<[f32; 3]> = obj
            .lines()
            .filter(|l| l.starts_with("v "))
            .map(|l| {
                let mut it = l.split_whitespace().skip(1).map(|n| n.parse::<f32>().unwrap());
                [it.next().unwrap(), it.next().unwrap(), it.next().unwrap()]
            })
            .collect();
        let min_x = verts.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
        let max_x = verts.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_y = verts.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min);
        let min_z = verts.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = verts.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);

        assert!((min_x + max_x).abs() < 1e-5, "X should straddle the origin");
        assert!((min_z + max_z).abs() < 1e-5, "Z should straddle the origin");
        assert!(min_y.abs() < 1e-5, "the model should rest on y=0");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(dir.join("voxely_test_origin.mtl"));
    }

    #[test]
    fn save_then_load_vox_round_trips() {
        let mut chunk = Chunk::with_size(8, 8, 8);
        chunk.set(1, 2, 3, Voxel { color_index: 7 });
        chunk.set(5, 0, 6, Voxel { color_index: 3 });
        let palette = Palette::default();

        let path = std::env::temp_dir().join("voxely_test_project.vox");
        save_vox(&path, &chunk, &palette).expect("save should succeed");
        let loaded = load_vox(&path).expect("load should succeed");

        assert_eq!(loaded.chunk.get(1, 2, 3).unwrap().color_index, 7);
        assert_eq!(loaded.chunk.get(5, 0, 6).unwrap().color_index, 3);
        // The palette colors for the used indices survive the round-trip.
        assert_eq!(loaded.palette.colors[7], palette.colors[7]);
        assert_eq!(loaded.palette.colors[3], palette.colors[3]);
        let _ = std::fs::remove_file(&path);
    }
}
