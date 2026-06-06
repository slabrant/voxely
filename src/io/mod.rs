use std::collections::HashMap;
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

/// Open a model file, dispatching on its extension. `.vox` is the native,
/// lossless format (voxel grid + palette); `.obj` is a best-effort import of a
/// mesh we previously exported (one cube per voxel).
pub fn open(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("vox") => load_vox(path),
        Some("obj") => load_obj(path),
        other => Err(format!("unsupported file type: {}", other.unwrap_or("(none)")).into()),
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

/// Best-effort import of a Wavefront `.obj` we previously exported: one cube
/// per voxel, eight vertices each, grouped by `usemtl matN`. Foreign meshes may
/// not reconstruct cleanly — `.vox` is the format for round-tripping.
///
/// Colors come from the companion `.mtl` (`newmtl matN` + `Kd`), so the
/// original palette indices are recovered when present.
pub fn load_obj(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;

    // material name -> palette index, from the companion .mtl. Falls back to
    // sequential indices for any material we can't map.
    let mtl_colors = read_mtl(&path.with_extension("mtl")).unwrap_or_default();
    let mut palette = Palette::default();
    for (&idx, &color) in &mtl_colors {
        if let Some(slot) = palette.colors.get_mut(idx as usize) {
            *slot = color;
        }
    }

    // Walk the file in order, tracking the active material, collecting vertices.
    // Every eight vertices form one exported cube.
    let mut cur_color = 1u8;
    let mut next_seq = 1u8; // for materials that aren't named "matN"
    let mut seq_map: HashMap<String, u8> = HashMap::new();
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut cubes: Vec<([f32; 3], u8)> = Vec::new(); // (min corner, color index)

    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("usemtl") => {
                let name = it.next().unwrap_or("");
                cur_color = name
                    .strip_prefix("mat")
                    .and_then(|n| n.parse::<u8>().ok())
                    .unwrap_or_else(|| {
                        *seq_map.entry(name.to_string()).or_insert_with(|| {
                            let c = next_seq;
                            next_seq = next_seq.saturating_add(1);
                            c
                        })
                    });
            }
            Some("v") => {
                let coords: Vec<f32> = it.filter_map(|t| t.parse().ok()).collect();
                if coords.len() >= 3 {
                    verts.push([coords[0], coords[1], coords[2]]);
                    if verts.len() == 8 {
                        let min = verts.iter().fold([f32::INFINITY; 3], |a, v| {
                            [a[0].min(v[0]), a[1].min(v[1]), a[2].min(v[2])]
                        });
                        cubes.push((min, cur_color));
                        verts.clear();
                    }
                }
            }
            _ => {}
        }
    }

    if cubes.is_empty() {
        return Err("no voxels could be reconstructed from the .obj".into());
    }

    // Re-anchor to a non-negative integer grid: subtract the global minimum
    // corner and round. (Export re-origins to the bottom-center, so corners can
    // be fractional and negative.)
    let gmin = cubes.iter().fold([f32::INFINITY; 3], |a, (m, _)| {
        [a[0].min(m[0]), a[1].min(m[1]), a[2].min(m[2])]
    });
    let cells: Vec<([usize; 3], u8)> = cubes
        .iter()
        .map(|(m, c)| {
            (
                [
                    (m[0] - gmin[0]).round().max(0.0) as usize,
                    (m[1] - gmin[1]).round().max(0.0) as usize,
                    (m[2] - gmin[2]).round().max(0.0) as usize,
                ],
                *c,
            )
        })
        .collect();

    let dims = cells.iter().fold([0usize; 3], |a, (p, _)| {
        [a[0].max(p[0] + 1), a[1].max(p[1] + 1), a[2].max(p[2] + 1)]
    });
    let mut chunk = Chunk::with_size(dims[0], dims[1], dims[2]);
    for (p, color_index) in cells {
        chunk.set(p[0], p[1], p[2], Voxel { color_index });
    }

    Ok(Project { chunk, palette })
}

/// Parse `newmtl matN` / `Kd r g b` pairs into a map of palette index -> color.
/// Only materials named `matN` are mapped (that's how [`export_obj`] names them).
fn read_mtl(path: &Path) -> Option<HashMap<u8, [u8; 4]>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut map = HashMap::new();
    let mut cur: Option<u8> = None;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("newmtl") => {
                cur = it.next().and_then(|n| n.strip_prefix("mat")).and_then(|n| n.parse().ok());
            }
            Some("Kd") => {
                if let Some(idx) = cur {
                    let c: Vec<f32> = it.filter_map(|t| t.parse().ok()).collect();
                    if c.len() >= 3 {
                        map.insert(idx, [
                            (c[0] * 255.0).round() as u8,
                            (c[1] * 255.0).round() as u8,
                            (c[2] * 255.0).round() as u8,
                            255,
                        ]);
                    }
                }
            }
            _ => {}
        }
    }
    Some(map)
}

/// Export the chunk to a Wavefront `.obj` file (plus a companion `.mtl` with one
/// material per used color). Emits one cube per solid voxel — simple, correct,
/// and colored; greedy merging isn't needed for an export of this size.
pub fn export_obj(path: impl AsRef<Path>, chunk: &Chunk, palette: &Palette) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    let mtl_path = path.with_extension("mtl");
    let mtl_name = mtl_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("invalid .mtl path")?
        .to_string();

    // Local corner offsets of a unit cube and its six quad faces (1-based,
    // relative to the cube's first vertex), wound counter-clockwise outward.
    const CORNERS: [[i32; 3]; 8] = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 5, 4], // bottom (-Y)
        [3, 7, 6, 2], // top (+Y)
        [0, 3, 2, 1], // front (-Z)
        [4, 5, 6, 7], // back (+Z)
        [0, 4, 7, 3], // left (-X)
        [1, 2, 6, 5], // right (+X)
    ];
    // One flat normal per face, in the same order as FACES. Emitting explicit
    // normals (and never sharing vertices between faces) keeps importers like
    // Godot from auto-generating *smooth* normals, which is what makes an
    // un-normaled voxel export look gradient-shaded / "blurry".
    const NORMALS: [[i32; 3]; 6] = [
        [0, -1, 0], // bottom
        [0, 1, 0],  // top
        [0, 0, -1], // front
        [0, 0, 1],  // back
        [-1, 0, 0], // left
        [1, 0, 0],  // right
    ];

    // Group by color so each material is referenced once and faces stay valid,
    // and find the occupied bounding box so we can re-origin the model.
    let mut used = std::collections::BTreeSet::new();
    let mut by_color: std::collections::BTreeMap<u8, Vec<(usize, usize, usize)>> = std::collections::BTreeMap::new();
    let (mut min, mut max) = ([usize::MAX; 3], [usize::MIN; 3]);
    for x in 0..chunk.width {
        for y in 0..chunk.height {
            for z in 0..chunk.depth {
                if let Some(v) = chunk.get(x, y, z) {
                    if !v.is_empty() {
                        by_color.entry(v.color_index).or_default().push((x, y, z));
                        for (i, &c) in [x, y, z].iter().enumerate() {
                            min[i] = min[i].min(c);
                            max[i] = max[i].max(c);
                        }
                    }
                }
            }
        }
    }

    // Move the origin to the bottom-center of the model: X/Z centered on the
    // occupied volume, Y resting on its lowest layer. Empty models export
    // nothing meaningful, so fall back to no shift.
    let (ox, oy, oz) = if by_color.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        (
            (min[0] + max[0] + 1) as f32 / 2.0,
            min[1] as f32,
            (min[2] + max[2] + 1) as f32 / 2.0,
        )
    };

    // World-units emitted per voxel. OBJ is unitless but importers (Godot)
    // treat one unit as one metre, so a unit cube per voxel gives a 1 m block.
    const VOXEL_SIZE_M: f32 = 1.0;

    let mut obj = String::new();
    writeln!(obj, "# Exported by Voxely")?;
    writeln!(obj, "mtllib {mtl_name}")?;
    for n in NORMALS {
        writeln!(obj, "vn {} {} {}", n[0], n[1], n[2])?;
    }

    let mut vbase = 0u32;
    for (&color, cells) in &by_color {
        used.insert(color);
        writeln!(obj, "usemtl mat{color}")?;
        for &(x, y, z) in cells {
            for c in CORNERS {
                writeln!(
                    obj,
                    "v {} {} {}",
                    ((x as i32 + c[0]) as f32 - ox) * VOXEL_SIZE_M,
                    ((y as i32 + c[1]) as f32 - oy) * VOXEL_SIZE_M,
                    ((z as i32 + c[2]) as f32 - oz) * VOXEL_SIZE_M
                )?;
            }
            for (fi, f) in FACES.iter().enumerate() {
                let n = fi + 1; // vn indices are global and 1-based
                writeln!(
                    obj,
                    "f {0}//{n} {1}//{n} {2}//{n} {3}//{n}",
                    vbase + f[0] as u32 + 1,
                    vbase + f[1] as u32 + 1,
                    vbase + f[2] as u32 + 1,
                    vbase + f[3] as u32 + 1
                )?;
            }
            vbase += 8;
        }
    }

    let mut mtl = String::from("# Exported by Voxely\n");
    for color in used {
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
    fn export_obj_emits_a_cube_per_voxel() {
        let mut chunk = Chunk::new();
        chunk.set(2, 3, 4, Voxel { color_index: 1 });
        let palette = Palette::default();

        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_export.obj");
        export_obj(&obj_path, &chunk, &palette).expect("export should succeed");

        let obj = std::fs::read_to_string(&obj_path).unwrap();
        assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), 8, "one cube = 8 vertices");
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 6, "one cube = 6 faces");
        assert!(obj.contains("usemtl mat1"));

        let mtl = std::fs::read_to_string(dir.join("voxely_test_export.mtl")).unwrap();
        assert!(mtl.contains("newmtl mat1"));

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(dir.join("voxely_test_export.mtl"));
    }

    #[test]
    fn export_reorigins_to_bottom_center_with_flat_normals() {
        // Two voxels stacked at x=4,z=6 over y=2..=3. Occupied box is a single
        // column, so X/Z center on the cell centers (4.5, 6.5) and Y rests on
        // the lowest layer (2).
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

    #[test]
    fn export_then_load_obj_round_trips_shape_and_color() {
        let mut chunk = Chunk::with_size(10, 10, 10);
        chunk.set(2, 0, 1, Voxel { color_index: 4 });
        chunk.set(3, 0, 1, Voxel { color_index: 4 });
        chunk.set(2, 1, 1, Voxel { color_index: 7 });
        let palette = Palette::default();

        let path = std::env::temp_dir().join("voxely_test_obj_roundtrip.obj");
        export_obj(&path, &chunk, &palette).expect("export should succeed");
        let loaded = load_obj(&path).expect("load should succeed");

        // Shape is re-anchored to the origin: the lowest/leftmost voxel moves to
        // (0,0,0), so relative layout is what we check.
        let count = (0..loaded.chunk.width)
            .flat_map(|x| (0..loaded.chunk.height).flat_map(move |y| (0..loaded.chunk.depth).map(move |z| (x, y, z))))
            .filter(|&(x, y, z)| loaded.chunk.get(x, y, z).map(|v| !v.is_empty()).unwrap_or(false))
            .count();
        assert_eq!(count, 3, "all three voxels survive the obj round-trip");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("mtl"));
    }
}
