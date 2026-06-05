use std::error::Error;
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::chunk::CHUNK_SIZE;
use crate::core::{Chunk, Palette, Voxel};

/// Everything persisted in a native `.voxely` project file.
#[derive(Serialize, Deserialize)]
pub struct Project {
    pub chunk: Chunk,
    pub palette: Palette,
}

/// Save the current chunk and palette to a native project file.
pub fn save_project(path: impl AsRef<Path>, chunk: &Chunk, palette: &Palette) -> Result<(), Box<dyn Error>> {
    let project = ProjectRef { chunk, palette };
    let bytes = bincode::serialize(&project)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Load a native project file. Returns an error (rather than panicking) on a
/// missing file or an incompatible/old format.
pub fn load_project(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let project = bincode::deserialize(&bytes)?;
    Ok(project)
}

/// Import a MagicaVoxel `.vox` file into a project, bringing along its real
/// color palette. MagicaVoxel is Z-up, so Y and Z are swapped to match our
/// Y-up engine. Color indices are shifted by one so index 0 stays "empty".
pub fn import_vox(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
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

    let mut chunk = Chunk::new();
    let mut dropped = 0usize;
    for v in &model.voxels {
        let color_index = v.i.saturating_add(1);
        // (x, y, z) MagicaVoxel -> (x, z, y) engine (Z-up to Y-up).
        if !chunk.set(v.x as usize, v.z as usize, v.y as usize, Voxel { color_index }) {
            dropped += 1;
        }
    }
    if dropped > 0 {
        log::warn!("{dropped} voxels were outside the {CHUNK_SIZE}^3 chunk and were dropped on import");
    }

    Ok(Project { chunk, palette })
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

    let mut used = std::collections::BTreeSet::new();
    let mut obj = String::new();
    writeln!(obj, "# Exported by Voxely")?;
    writeln!(obj, "mtllib {mtl_name}")?;

    let mut vbase = 0u32;
    // Group by color so each material is referenced once and faces stay valid.
    let mut by_color: std::collections::BTreeMap<u8, Vec<(usize, usize, usize)>> = std::collections::BTreeMap::new();
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if let Some(v) = chunk.get(x, y, z) {
                    if !v.is_empty() {
                        by_color.entry(v.color_index).or_default().push((x, y, z));
                    }
                }
            }
        }
    }

    for (&color, cells) in &by_color {
        used.insert(color);
        writeln!(obj, "usemtl mat{color}")?;
        for &(x, y, z) in cells {
            for c in CORNERS {
                writeln!(
                    obj,
                    "v {} {} {}",
                    x as i32 + c[0],
                    y as i32 + c[1],
                    z as i32 + c[2]
                )?;
            }
            for f in FACES {
                writeln!(
                    obj,
                    "f {} {} {} {}",
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

/// Borrowed mirror of [`Project`] so saving doesn't need to clone the chunk.
#[derive(Serialize)]
struct ProjectRef<'a> {
    chunk: &'a Chunk,
    palette: &'a Palette,
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
    fn save_then_load_round_trips() {
        let mut chunk = Chunk::new();
        chunk.set(1, 1, 1, Voxel { color_index: 7 });
        let palette = Palette::default();

        let path = std::env::temp_dir().join("voxely_test_project.voxely");
        save_project(&path, &chunk, &palette).expect("save should succeed");
        let loaded = load_project(&path).expect("load should succeed");

        assert_eq!(loaded.chunk.get(1, 1, 1).unwrap().color_index, 7);
        let _ = std::fs::remove_file(&path);
    }
}
