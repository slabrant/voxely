# Voxely — Software Description

**Voxely** is a desktop voxel editor for Linux (Debian), written in Rust. It lets users build, sculpt, and color 3D models out of voxels (3D pixels) in an interactive viewport, in the same spirit as **MagicaVoxel** and **Qubicle**.

---

## 1. Purpose & Vision

Voxely aims to be a fast, lightweight, and modern voxel modeling tool that runs natively on Debian-based Linux systems. It focuses on a smooth real-time editing experience, an intuitive interface, and clean export options for use in games, rendering, and 3D printing pipelines.

---

## 2. Target Platform

| Aspect | Detail |
|--------|--------|
| **OS** | Debian (and Debian-based distros like Ubuntu) |
| **Architecture** | x86_64 (primary), with potential for ARM64 later |
| **Language** | Rust (edition 2024) |
| **Graphics** | GPU-accelerated rendering via Vulkan/OpenGL backends |
| **Distribution** | `.deb` package and/or standalone binary |

---

## 3. Core Features

### Editing
- **Voxel placement & removal** with a brush, in a 3D grid.
- **Box / line / fill tools** for fast bulk editing.
- **Color palette** with per-voxel color assignment.
- **Eyedropper / paint** tools for recoloring.
- **Selection & transform** (move, mirror, rotate regions).
- **Undo / redo** history.

### Viewport
- **Real-time 3D view** with orbit, pan, and zoom camera controls.
- **Grid and ground-plane guides** for orientation.
- **Lighting / shading** to visualize the model in 3D.
- Optional **multi-view** (top / front / side) editing.

### Project & Data
- **Scene model** holding one or more voxel volumes/objects.
- **Native project format** will save and load from common voxel and mesh formats (e.g. `.vox`, `.obj`, `.ply`).

---

## 4. Suggested Architecture (High Level)

A modular design keeps the editor maintainable and testable:

- **`core`** — voxel data structures (chunks, sparse structures like SVOs), the color palette, and edit operations. Pure logic, no rendering.
- **`render`** — turns voxel volumes into optimized meshes (e.g., via **Greedy Meshing**) and draws them via the GPU.
- **`editor`** — tools, selection, undo/redo command stack, camera control.
- **`ui`** — windows, panels, toolbars, and palette widgets.
- **`io`** — project save/load and external formats.
- **`app`** — wires everything together: window creation, the main loop, and input dispatch.

---

## 5. Likely Rust Crate Choices

| Concern | Candidate Crates |
|---------|------------------|
| Windowing / event loop | `winit` |
| GPU rendering | `wgpu` (cross-platform, Vulkan/GL backends) |
| Immediate-mode UI | `egui` + `egui-wgpu` |
| Math (vectors/matrices) | `glam` |
| Serialization (project files) | `serde` + `bincode` / `ron` |
| `.vox` format support | `dot_vox` |
| Mesh export (`.obj`) | custom writer or `obj` crates |
| Logging | `log` + `env_logger` |
| Parallelism | `rayon` (for meshing/heavy logic) |
| Debian Packaging | `cargo-deb` |

---

## 6. Initial Milestones

1. **Window + viewport** — open a window, render a single colored voxel cube via `wgpu`.
2. **Voxel grid + camera** — orbiting camera around a chunked voxel volume.
3. **Meshing engine** — implement greedy meshing for efficient rendering.
4. **Place/remove voxels** — basic mouse interaction with the grid; incremental mesh updates.
5. **Palette + coloring** — assign colors to voxels.
6. **Save/Load** — native project format.
7. **Import/Export** — `.vox` and `.obj` support.
8. **Polish** — undo/redo, 3D transformation gizmos, UI panels, Debian packaging.