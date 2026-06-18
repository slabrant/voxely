# Voxely

A small voxel editor written in Rust, rendered with [wgpu](https://wgpu.rs/) and
an [egui](https://www.egui.rs/) side panel.

## Features

- Build, paint (recolor), erase, line-drag, rectangle-fill, and flood-fill voxels
- Eyedropper (press `Q`, then click) to sample a voxel's color
- Cursor face highlight showing where the next voxel lands
- 64-color palette with live recoloring, and full undo/redo (one step per action)
- Resizable canvas (W×H×D) from the UI
- Save/Open native MagicaVoxel `.vox` projects (lossless voxel grid + palette)
- Export to Wavefront `.obj` (+ `.mtl`) via a native save dialog — flat-shaded,
  1 m blocks, origin at the model's bottom-center, with hidden faces culled and
  coplanar same-color faces merged (greedy meshing) to keep the triangle count low
- Import a Voxely-exported `.obj` (with its `.mtl`) back into voxels, refilling
  any interior that face-culling removed on export
- Open files from the OS ("Open With…") or by dragging them onto the window

## Requirements

- **Rust 1.85+** (uses the 2024 edition) — install via [rustup](https://rustup.rs/)
- A GPU with a Vulkan/Metal/DX12 backend

On Debian/Ubuntu you'll also need the windowing/graphics dev libraries:

```sh
sudo apt install build-essential pkg-config \
  libxkbcommon-dev libwayland-dev libx11-dev \
  libvulkan1 mesa-vulkan-drivers
```

## Build & Run

```sh
cargo run --release      # build and launch
cargo build --release    # build only (binary in target/release/)
cargo test               # run the test suite
```

### Cross-compiling to Windows (from Linux)

`Cargo.toml` is preconfigured for the `mingw-w64` toolchain:

```sh
sudo apt install mingw-w64
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## Controls

| Action | Input |
| --- | --- |
| Build / paint voxel (active tool) | Left-click / left-drag |
| Erase voxel | Shift + left-click / drag |
| Eyedropper (sample color) | Press `Q`, then left-click (or Alt + left-click) |
| Cycle tool (Build → Paint → Bucket → Extrude) | `Tab` / `Shift`+`Tab` (reverse) |
| Fill rectangle | Ctrl + left-drag (Build) |
| Erase rectangle | Ctrl + Shift + left-drag |
| Paint bucket | Bucket tool, click to fill — Shift + left-click to erase region |
| Pick color | `1`–`9`, or click a palette swatch |
| Orbit / pan / zoom | Right-drag / middle-drag / scroll |
| Undo / redo | Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) |
| Save / Save As / open | Ctrl+S / Ctrl+Shift+S / Ctrl+O |

File, Edit (canvas extents), and Help live in the top menu bar.

Tools, history, file actions, canvas size, and the palette are also available in
the left-hand panel.
