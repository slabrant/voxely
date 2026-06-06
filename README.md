# Voxely

A small voxel editor written in Rust, rendered with [wgpu](https://wgpu.rs/) and
an [egui](https://www.egui.rs/) side panel.

## Features

- Place, erase, line-drag, rectangle-fill, and flood-fill (paint bucket) voxels
- Cursor face highlight showing where the next voxel lands
- 64-color palette with live recoloring, and full undo/redo (one step per action)
- Resizable canvas (W×H×D) from the UI
- Save/load native `.voxely` projects, import MagicaVoxel `.vox`
- Export to Wavefront `.obj` (+ `.mtl`) via a native save dialog — flat-shaded,
  1 m blocks, origin at the model's bottom-center

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
| Place voxel | Left-click / left-drag |
| Remove voxel | Shift + left-click / drag |
| Fill rectangle | Ctrl + left-drag |
| Erase rectangle | Ctrl + Shift + left-drag |
| Paint bucket (fill region) | `B` to toggle, then click — Shift to erase |
| Pick color | `1`–`9`, or click a palette swatch |
| Orbit / pan / zoom | Right-drag / middle-drag / scroll |
| Undo / redo | Ctrl+Z / Ctrl+Y |
| Save / load | Ctrl+S / Ctrl+L |
| Import `.vox` / export `.obj` | Ctrl+I / Ctrl+E |

Tools, history, file actions, canvas size, and the palette are also available in
the left-hand panel.
