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

/// Open a model file. Only `.obj` is supported, and only when it was produced by
/// [`export_obj`] (its companion `.mtl` must sit alongside it).
pub fn open(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("obj") => load_obj(path),
        other => Err(format!(
            "unsupported file type: {} (only .obj can be opened)",
            other.unwrap_or("(none)")
        )
        .into()),
    }
}

/// Save a model as a Wavefront `.obj` mesh (plus a companion `.mtl`). Mirrors
/// [`open`] so Save needs no separate "export" step.
pub fn save(path: impl AsRef<Path>, chunk: &Chunk, palette: &Palette) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("obj") => export_obj(path, chunk, palette),
        other => Err(format!("unsupported file type: {} (save as .obj)", other.unwrap_or("(none)")).into()),
    }
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
    for (slot, (&_color, group)) in by_color.iter().enumerate() {
        let mtl_name_local = format!("mtl{}", slot + 1);
        writeln!(obj, "usemtl {mtl_name_local}")?;
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
    for (slot, &color) in by_color.keys().enumerate() {
        let c = palette.colors.get(color as usize).copied().unwrap_or([255, 255, 255, 255]);
        writeln!(mtl, "newmtl mtl{}", slot + 1)?;
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

/// Import a Wavefront `.obj` produced by [`export_obj`] back into voxels.
///
/// The export keeps only the *surface* of the model: every quad is an
/// axis-aligned, greedy-merged rectangle of one color, with an explicit flat
/// normal telling us which way the face points. From each face we know exactly
/// which voxel cell sits behind it (the one on the solid side of the normal) and
/// its color, so the visible "crust" is reconstructed directly. Interior voxels
/// were culled on export, so any empty cell that the surface fully seals off
/// (unreachable from outside the bounding box) is filled back in as solid — a
/// solid block round-trips as a solid block rather than a hollow shell.
///
/// The format is strict: this only accepts the structure Voxely writes (quads
/// with `v//vn` faces, a `mtllib` whose `.mtl` sits alongside, and `usemtl`
/// materials carrying `Kd`). Anything else is reported as an error.
pub fn load_obj(path: impl AsRef<Path>) -> Result<Project, Box<dyn Error>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    // Colors keyed by material name, loaded from the `.mtl` named by `mtllib`.
    // Resolved lazily on the first `usemtl` so a file with no faces is fine.
    let mut mtl_colors: Option<std::collections::HashMap<String, [u8; 4]>> = None;

    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[i32; 3]> = Vec::new();

    // A reconstructed face: four 1-based vertex indices, a 1-based normal index,
    // and the palette color index of its material.
    struct Face {
        verts: [usize; 4],
        normal: usize,
        color: u8,
    }
    let mut faces: Vec<Face> = Vec::new();

    // Materials are renamed to mtl1, mtl2, ... on export with no original index
    // preserved, so we hand out fresh palette slots 1.. in first-use order and
    // remember the mapping for the rest of the file.
    let mut palette = Palette::default();
    let mut mat_index: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
    let mut next_color: u8 = 1;
    let mut current_color: Option<u8> = None;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        let Some(kind) = tok.next() else { continue };
        let err = |msg: &str| -> Box<dyn Error> {
            format!("{}:{}: {msg}", path.display(), lineno + 1).into()
        };
        match kind {
            "mtllib" => {
                // The filename can contain spaces (e.g. "data diel.mtl"), so take
                // the whole remainder of the line rather than a single token.
                let name = line["mtllib".len()..].trim();
                if name.is_empty() {
                    return Err(err("mtllib is missing a filename"));
                }
                let mtl_path = path.with_file_name(name);
                let mtl_text = std::fs::read_to_string(&mtl_path).map_err(|e| {
                    err(&format!("could not read companion material file {}: {e}", mtl_path.display()))
                })?;
                mtl_colors = Some(parse_mtl(&mtl_text)?);
            }
            "v" => {
                let c = parse_vec3(&mut tok).ok_or_else(|| err("vertex needs three numbers"))?;
                vertices.push(c);
            }
            "vn" => {
                let c = parse_vec3(&mut tok).ok_or_else(|| err("normal needs three numbers"))?;
                // Export only ever writes the six unit axis normals; round so a
                // value like 1.0 lands cleanly on an integer axis direction.
                normals.push([c[0].round() as i32, c[1].round() as i32, c[2].round() as i32]);
            }
            "usemtl" => {
                let name = tok.next().ok_or_else(|| err("usemtl is missing a name"))?;
                let color = match mat_index.get(name) {
                    Some(&c) => c,
                    None => {
                        let colors = mtl_colors
                            .as_ref()
                            .ok_or_else(|| err("usemtl appears before any mtllib"))?;
                        let rgba = *colors
                            .get(name)
                            .ok_or_else(|| err(&format!("material '{name}' is not defined in the .mtl")))?;
                        if next_color == u8::MAX {
                            return Err(err("too many materials (max 254)"));
                        }
                        let slot = next_color;
                        palette.colors[slot as usize] = rgba;
                        mat_index.insert(name.to_string(), slot);
                        next_color += 1;
                        slot
                    }
                };
                current_color = Some(color);
            }
            "f" => {
                let color = current_color.ok_or_else(|| err("face appears before any usemtl"))?;
                let mut verts = [0usize; 4];
                let mut normal = 0usize;
                let mut count = 0;
                for vtok in tok {
                    if count == 4 {
                        return Err(err("faces must be quads (Voxely exports four-sided faces)"));
                    }
                    // Each reference is `v//vn`; the middle texture slot is empty.
                    let mut parts = vtok.split('/');
                    let vi: usize = parts
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| err("face vertex index is malformed"))?;
                    let _ = parts.next();
                    let ni: usize = parts
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| err("faces must reference a normal (v//vn)"))?;
                    if vi == 0 || vi > vertices.len() {
                        return Err(err("face references an out-of-range vertex"));
                    }
                    if ni == 0 || ni > normals.len() {
                        return Err(err("face references an out-of-range normal"));
                    }
                    verts[count] = vi;
                    normal = ni;
                    count += 1;
                }
                if count != 4 {
                    return Err(err("faces must be quads (Voxely exports four-sided faces)"));
                }
                faces.push(Face { verts, normal, color });
            }
            // Texture coords and smoothing groups are never written by Voxely;
            // ignore the handful of harmless keywords rather than erroring.
            "vt" | "s" | "o" | "g" => {}
            other => return Err(err(&format!("unexpected '{other}' directive"))),
        }
    }

    // An empty export (no geometry) opens as a fresh default canvas.
    if faces.is_empty() {
        return Ok(Project { chunk: Chunk::new(), palette });
    }

    // Vertices are `grid_corner - origin` with a constant per-axis offset and
    // exactly one unit between cells, so shifting by the per-axis minimum and
    // rounding recovers non-negative integer grid-corner coordinates.
    let mut vmin = [f32::INFINITY; 3];
    for v in &vertices {
        for a in 0..3 {
            vmin[a] = vmin[a].min(v[a]);
        }
    }
    let grid = |vi: usize| -> [i32; 3] {
        let v = vertices[vi - 1];
        [
            (v[0] - vmin[0]).round() as i32,
            (v[1] - vmin[1]).round() as i32,
            (v[2] - vmin[2]).round() as i32,
        ]
    };

    // The largest corner index along an axis equals the cell count there.
    let mut dims = [0i32; 3];
    for v in 1..=vertices.len() {
        let g = grid(v);
        for a in 0..3 {
            dims[a] = dims[a].max(g[a]);
        }
    }
    let dims = [dims[0].max(1) as usize, dims[1].max(1) as usize, dims[2].max(1) as usize];
    for (axis, &d) in ["X", "Y", "Z"].iter().zip(&dims) {
        if d > crate::core::chunk::MAX_CHUNK_SIZE {
            return Err(format!(
                "model is {d} voxels on {axis} (max {})",
                crate::core::chunk::MAX_CHUNK_SIZE
            )
            .into());
        }
    }

    let mut chunk = Chunk::with_size(dims[0], dims[1], dims[2]);

    // Reconstruct the crust: each face's normal points away from the solid, so
    // the solid cell sits at the face plane for a negative normal and one cell
    // back for a positive one.
    for f in &faces {
        let n = normals[f.normal - 1];
        let na = (0..3).find(|&a| n[a] != 0).ok_or("a face normal is zero")?;
        let s = n[na];
        let others: Vec<usize> = (0..3).filter(|&a| a != na).collect();
        let (ua, va) = (others[0], others[1]);

        let corners: Vec<[i32; 3]> = f.verts.iter().map(|&vi| grid(vi)).collect();
        let p = corners[0][na];
        let u0 = corners.iter().map(|c| c[ua]).min().unwrap();
        let u1 = corners.iter().map(|c| c[ua]).max().unwrap();
        let v0 = corners.iter().map(|c| c[va]).min().unwrap();
        let v1 = corners.iter().map(|c| c[va]).max().unwrap();
        let cell_na = if s < 0 { p } else { p - 1 };
        if cell_na < 0 {
            continue;
        }
        for ui in u0..u1 {
            for vj in v0..v1 {
                let mut c = [0usize; 3];
                c[na] = cell_na as usize;
                c[ua] = ui as usize;
                c[va] = vj as usize;
                chunk.set(c[0], c[1], c[2], Voxel { color_index: f.color });
            }
        }
    }

    fill_enclosed_interior(&mut chunk);

    Ok(Project { chunk, palette })
}

/// Parse a Voxely-exported `.mtl` into `name -> RGBA`. Only `newmtl`/`Kd` are
/// read; `Kd` is sRGB 0..1 (alpha is not exported, so it defaults to opaque).
fn parse_mtl(text: &str) -> Result<std::collections::HashMap<String, [u8; 4]>, Box<dyn Error>> {
    let mut colors = std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("newmtl") => {
                let name = tok.next().ok_or("newmtl is missing a name")?;
                current = Some(name.to_string());
                colors.entry(name.to_string()).or_insert([255, 255, 255, 255]);
            }
            Some("Kd") => {
                let name = current.as_ref().ok_or("Kd appears before any newmtl")?;
                let rgb = parse_vec3(&mut tok).ok_or("Kd needs three numbers")?;
                let to_byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
                colors.insert(name.clone(), [to_byte(rgb[0]), to_byte(rgb[1]), to_byte(rgb[2]), 255]);
            }
            _ => {}
        }
    }
    Ok(colors)
}

/// Parse the next three whitespace-separated floats from `tok`.
fn parse_vec3<'a>(tok: &mut impl Iterator<Item = &'a str>) -> Option<[f32; 3]> {
    let x = tok.next()?.parse().ok()?;
    let y = tok.next()?.parse().ok()?;
    let z = tok.next()?.parse().ok()?;
    Some([x, y, z])
}

/// Flood empty space inward from the bounding-box border (6-connected) and mark
/// everything it can't reach as solid. The reconstructed crust is a closed shell
/// around the model's hidden core, so any empty cell the flood never touches was
/// interior and is filled with the model's most common surface color.
fn fill_enclosed_interior(chunk: &mut Chunk) {
    let (w, h, d) = (chunk.width, chunk.height, chunk.depth);
    let idx = |x: usize, y: usize, z: usize| x + y * w + z * w * h;

    // Snapshot solidity and tally crust colors in one pass. The most common
    // surface color fills interior voxels (invisible anyway) so they blend in if
    // ever exposed by later editing.
    let mut solid = vec![false; w * h * d];
    let mut counts = [0u32; 256];
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if let Some(v) = chunk.get(x, y, z).filter(|v| !v.is_empty()) {
                    solid[idx(x, y, z)] = true;
                    counts[v.color_index as usize] += 1;
                }
            }
        }
    }
    let fill_color = counts
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|&(_, &n)| n)
        .map(|(i, _)| i as u8)
        .unwrap_or(1);

    // Flood empty air inward from the six bounding faces (6-connected). Whatever
    // it can't reach is sealed interior.
    let mut outside = vec![false; w * h * d];
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    let on_border = |x: usize, y: usize, z: usize| {
        x == 0 || y == 0 || z == 0 || x == w - 1 || y == h - 1 || z == d - 1
    };
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if on_border(x, y, z) && !solid[idx(x, y, z)] && !outside[idx(x, y, z)] {
                    outside[idx(x, y, z)] = true;
                    stack.push((x, y, z));
                }
            }
        }
    }
    while let Some((x, y, z)) = stack.pop() {
        let neighbors = [
            (x.wrapping_sub(1), y, z, x > 0),
            (x + 1, y, z, x + 1 < w),
            (x, y.wrapping_sub(1), z, y > 0),
            (x, y + 1, z, y + 1 < h),
            (x, y, z.wrapping_sub(1), z > 0),
            (x, y, z + 1, z + 1 < d),
        ];
        for (nx, ny, nz, in_bounds) in neighbors {
            if in_bounds && !solid[idx(nx, ny, nz)] && !outside[idx(nx, ny, nz)] {
                outside[idx(nx, ny, nz)] = true;
                stack.push((nx, ny, nz));
            }
        }
    }

    // Empty and never reached from outside => sealed interior => fill it.
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if !solid[idx(x, y, z)] && !outside[idx(x, y, z)] {
                    chunk.set(x, y, z, Voxel { color_index: fill_color });
                }
            }
        }
    }
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
        assert!(obj.contains("usemtl mtl1"));

        let mtl = std::fs::read_to_string(dir.join("voxely_test_export.mtl")).unwrap();
        assert!(mtl.contains("newmtl mtl1"));

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
    fn export_then_import_obj_round_trips_shape_and_color() {
        // An L of two differently-colored voxels: shape, relative position, and
        // each voxel's color must survive an export/import round-trip.
        let mut chunk = Chunk::with_size(8, 8, 8);
        chunk.set(2, 0, 3, Voxel { color_index: 1 }); // red
        chunk.set(3, 0, 3, Voxel { color_index: 3 }); // blue
        let palette = Palette::default();

        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_import.obj");
        export_obj(&obj_path, &chunk, &palette).expect("export should succeed");

        let project = load_obj(&obj_path).expect("import should succeed");
        let imported = &project.chunk;

        // Exactly two solid voxels, side by side along X on the bottom layer.
        let mut solids = Vec::new();
        for x in 0..imported.width {
            for y in 0..imported.height {
                for z in 0..imported.depth {
                    if imported.get(x, y, z).is_some_and(|v| !v.is_empty()) {
                        solids.push((x, y, z, imported.get(x, y, z).unwrap().color_index));
                    }
                }
            }
        }
        assert_eq!(solids.len(), 2, "two voxels survive the round-trip");
        solids.sort();
        let (lx, ly, lz, lc) = solids[0];
        let (rx, ry, rz, rc) = solids[1];
        assert_eq!((ry, rz), (ly, lz), "both rest on the same layer/row");
        assert_eq!(rx, lx + 1, "the two voxels are adjacent along X");

        // Colors come back via the .mtl: the imported palette slots match the
        // original red and blue, even though indices were renumbered on export.
        assert_eq!(project.palette.colors[lc as usize], palette.colors[1], "left stays red");
        assert_eq!(project.palette.colors[rc as usize], palette.colors[3], "right stays blue");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(obj_path.with_extension("mtl"));
    }

    #[test]
    fn import_obj_fills_sealed_interior_solid() {
        // A solid 3x3x3 block exports as just its surface (the center cell has no
        // exposed face). Import must flood-fill that sealed cell back to solid.
        let mut chunk = Chunk::with_size(8, 8, 8);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    chunk.set(x, y, z, Voxel { color_index: 2 });
                }
            }
        }
        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_fill.obj");
        export_obj(&obj_path, &chunk, &Palette::default()).expect("export should succeed");

        let project = load_obj(&obj_path).expect("import should succeed");
        let n = (0..project.chunk.width)
            .flat_map(|x| (0..project.chunk.height).flat_map(move |y| (0..project.chunk.depth).map(move |z| (x, y, z))))
            .filter(|&(x, y, z)| project.chunk.get(x, y, z).is_some_and(|v| !v.is_empty()))
            .count();
        assert_eq!(n, 27, "the sealed center voxel is filled back in");

        let _ = std::fs::remove_file(&obj_path);
        let _ = std::fs::remove_file(obj_path.with_extension("mtl"));
    }

    #[test]
    fn import_obj_errors_without_mtl() {
        // Export, then delete the companion .mtl: import must fail rather than
        // silently dropping colors.
        let mut chunk = Chunk::new();
        chunk.set(1, 1, 1, Voxel { color_index: 1 });
        let dir = std::env::temp_dir();
        let obj_path = dir.join("voxely_test_nomtl.obj");
        export_obj(&obj_path, &chunk, &Palette::default()).expect("export should succeed");
        let _ = std::fs::remove_file(obj_path.with_extension("mtl"));

        assert!(load_obj(&obj_path).is_err(), "missing .mtl is an error");
        let _ = std::fs::remove_file(&obj_path);
    }
}
