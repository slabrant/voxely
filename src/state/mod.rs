//! The editor's whole mutable world: GPU resources, the voxel chunk, the active
//! gesture, and the UI that drives them.
//!
//! `State` is one struct because every part of a frame touches most of it — a
//! click reads the camera, walks the chunk, writes history and dirties the
//! mesh. Rather than split the *data*, this splits the *behaviour*: each
//! submodule adds an `impl State` block for one concern. They are children of
//! this module, so they still reach `State`'s private fields directly.

mod edit;
mod file;
mod geometry;
mod gpu;
mod overlay;
mod picker;
mod ui;

pub use gpu::{Texture, Vertex};

use geometry::bounding_box_lines;
use gpu::upload_growable;

use crate::camera::{Camera, CameraUniform, CameraController};
use crate::core::{Palette, Voxel};
use crate::editor::{EditHistory, VoxelEdit};
use glam::Vec4Swizzles;
use std::sync::Arc;
use winit::{
    event::WindowEvent,
    keyboard::ModifiersState,
    window::Window,
};

/// Default file name suggested in the "Save" dialog. Models save as a Wavefront
/// `.obj` mesh (plus a companion `.mtl`); see `crate::io::save`.
const OBJ_PATH: &str = "model.obj";

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    xray_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    line_vertex_buffer: wgpu::Buffer,
    overlay_vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    num_line_vertices: u32,
    num_overlay_vertices: u32,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    depth_texture: Texture,
    chunk: crate::core::Chunk,
    palette: Palette,
    history: EditHistory,
    cursor_position: winit::dpi::PhysicalPosition<f64>,
    current_color_index: u8,
    modifiers: ModifiersState,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    is_left_mouse_pressed: bool,
    /// Latched erase mode for the in-progress left-button gesture (Shift held
    /// at press). Kept stable for the whole gesture so releasing Shift mid-drag
    /// doesn't flip a stroke between place and erase.
    drag_erase: bool,
    /// True while an eyedropper click is in progress (Alt+Left, or a click
    /// while armed), so cursor motion isn't mistaken for a paint/build drag.
    is_eyedropping: bool,
    /// Set by tapping `Q`: the next left-click samples the color under the
    /// cursor instead of editing, then disarms. `Esc` cancels.
    eyedropper_armed: bool,
    /// True for the duration of a freehand (Build/Paint) erase stroke. While
    /// set, [`raycast`](Self::raycast) treats `erased_cells` as solid so the
    /// stroke stops at the surface it's carving instead of tunnelling deeper.
    is_erasing_gesture: bool,
    /// Cells cleared so far by the current freehand erase stroke. Reset when
    /// the gesture ends.
    erased_cells: std::collections::HashSet<[i32; 3]>,
    last_grid_coord: Option<[i32; 3]>,
    extrude_start: Option<(Vec<[i32; 3]>, [i32; 3], u8)>, // (positions, normal, color_index)
    /// In-progress Move gesture: the grabbed object plus the axis it slides
    /// along. `None` when no move is underway.
    move_start: Option<MoveDrag>,
    /// Step count the Move overlay was last drawn for, so cursor motion that
    /// doesn't change the snapped offset skips a rebuild.
    move_steps: Option<i32>,
    /// Voxels placed during the current gesture, excluded from raycasting so a
    /// stroke doesn't climb its own output. A set, not a list: `raycast` probes
    /// it on every hit, and a rectangle fill can put tens of thousands in it.
    drag_start_voxels: std::collections::HashSet<[i32; 3]>,
    rect_drag: Option<RectDrag>,
    extrude_steps: Option<i32>,
    tool: Tool,
    /// Set by [`remesh`](State::remesh); the mesh is rebuilt at most once per
    /// frame in [`flush_mesh`](State::flush_mesh).
    mesh_dirty: bool,
    /// Canvas dimensions edited in the UI, applied on "Resize".
    pending_size: [usize; 3],
    /// Text buffers backing the canvas-extent fields. Only authoritative while
    /// a field has focus; the moment it doesn't, it is rewritten from
    /// `pending_size`. That single rule makes an external change (New/Open),
    /// a clamped value, and invalid input all resolve on their own.
    size_text: [String; 3],
    /// Text buffer backing the active colour's hex field, under the same
    /// focus-gated sync rule as `size_text`.
    hex_text: String,
    /// Path of the current `.obj` model, if it has been saved/opened. `Save`
    /// writes here silently; `Save As` (or saving when this is `None`) prompts.
    current_path: Option<std::path::PathBuf>,
    /// Transient status banner (message, is-error, shown-at). Set by file
    /// open/save so the user sees the outcome in-app; auto-dismisses.
    status: Option<(String, bool, std::time::Instant)>,
    /// Directory of the last file opened or saved, so the next file dialog
    /// starts there instead of the home folder.
    last_dir: Option<std::path::PathBuf>,
}

/// The active editing tool, chosen from the UI or with a hotkey.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    /// Add voxels: left-click/drag places (and ctrl-drag fills a rectangle).
    /// Geometry only — erasing lives on Shift+Left-click for every tool.
    Build,
    /// Recolor existing voxels to the active color. Never adds or removes
    /// geometry: clicks/drags over empty space do nothing.
    Paint,
    /// Flood-fill a connected region of one color (click to apply).
    Bucket,
    /// Pull a face along its normal (click and drag).
    Extrude,
    /// Slide a whole connected object along the clicked face's normal (click and
    /// drag). The grabbed object is every voxel reachable from the hit one
    /// through shared faces, regardless of color.
    Move,
}

/// State for an in-progress ctrl+left-drag rectangle fill.
///
/// The plane is locked when the drag starts: `axis` is the face's normal axis,
/// `face_world` is the world coordinate of that face's plane (used for preview
/// rendering and for projecting the cursor ray back onto the plane), and
/// `place_value` is the cell index along `axis` where voxels are written.
/// The rectangle spans `start`..`cur` across the two in-plane axes.
#[derive(Clone, Copy)]
struct RectDrag {
    axis: usize,
    u_axis: usize,
    v_axis: usize,
    face_world: f32,
    normal: [f32; 3],
    place_value: i32,
    start_u: i32,
    start_v: i32,
    cur_u: i32,
    cur_v: i32,
    remove: bool,
}

/// State for an in-progress Move drag (the [`Tool::Move`] gesture).
///
/// `voxels` is the grabbed connected object captured at grab time — each cell's
/// position and color — so the move is independent of edits made by the slide
/// itself. `normal` is the unit integer axis the object slides along (the
/// clicked face's normal), and `anchor_center` is the world-space center of the
/// clicked voxel, the reference point for mapping the cursor to a step count.
struct MoveDrag {
    voxels: Vec<([i32; 3], u8)>,
    normal: [i32; 3],
    anchor_center: glam::Vec3,
}

impl State {
    /// Remember `path`'s parent directory as the starting point for the next
    /// file dialog.
    pub(super) fn remember_dir(&mut self, path: &std::path::Path) {
        if let Some(dir) = path.parent() {
            self.last_dir = Some(dir.to_path_buf());
        }
    }

    /// Show a transient status banner (green for success, red for error).
    pub(super) fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status = Some((msg.into(), is_error, std::time::Instant::now()));
    }


    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        }
    }

    /// Switches to the next tool, or the previous one when `reverse` is set
    /// (Shift+Tab). Order: Build → Paint → Bucket → Extrude → Move → Build.
    pub(super) fn cycle_tool(&mut self, reverse: bool) {
        self.tool = if reverse {
            match self.tool {
                Tool::Build => Tool::Move,
                Tool::Paint => Tool::Build,
                Tool::Bucket => Tool::Paint,
                Tool::Extrude => Tool::Bucket,
                Tool::Move => Tool::Extrude,
            }
        } else {
            match self.tool {
                Tool::Build => Tool::Paint,
                Tool::Paint => Tool::Bucket,
                Tool::Bucket => Tool::Extrude,
                Tool::Extrude => Tool::Move,
                Tool::Move => Tool::Build,
            }
        };
    }

    pub(super) fn undo(&mut self) {
        if let Some(group) = self.history.undo() {
            // Reverse order so coordinates touched more than once in a single
            // gesture land back on their original value.
            for e in group.iter().rev() {
                self.chunk.set(e.x as usize, e.y as usize, e.z as usize, e.old);
            }
            self.remesh();
            println!("Undo");
        }
    }

    pub(super) fn redo(&mut self) {
        if let Some(group) = self.history.redo() {
            for e in group.iter() {
                self.chunk.set(e.x as usize, e.y as usize, e.z as usize, e.new);
            }
            self.remesh();
            println!("Redo");
        }
    }

    /// Resizes the canvas, keeping voxels that still fit (anchored at the
    /// origin). Clears history since old coordinates may no longer be valid,
    /// then rebuilds the mesh and bounding-box wireframe.
    pub(super) fn resize_canvas(&mut self, width: usize, height: usize, depth: usize) {
        use crate::core::chunk::MAX_CHUNK_SIZE;
        let width = width.clamp(1, MAX_CHUNK_SIZE);
        let height = height.clamp(1, MAX_CHUNK_SIZE);
        let depth = depth.clamp(1, MAX_CHUNK_SIZE);
        if (width, height, depth) == (self.chunk.width, self.chunk.height, self.chunk.depth) {
            return;
        }
        self.chunk = self.chunk.resized(width, height, depth);
        self.history.clear();
        self.last_grid_coord = None;
        self.rect_drag = None;
        self.sync_to_chunk();
        self.frame_camera_to_chunk();
    }

    /// Points the camera at the centre of the current canvas and pulls the eye
    /// back far enough to see the whole bounding box, so a resize is immediately
    /// visible. The orbit/pan controller only applies deltas, so this sticks.
    pub(super) fn frame_camera_to_chunk(&mut self) {
        let (w, h, d) = (self.chunk.width as f32, self.chunk.height as f32, self.chunk.depth as f32);
        let center = glam::Vec3::new(w * 0.5, h * 0.5, d * 0.5);
        let radius = center.length();
        // Keep the current viewing direction; just re-distance from the centre.
        let dir = (self.camera.eye - self.camera.target).normalize_or_zero();
        let dir = if dir == glam::Vec3::ZERO { glam::Vec3::new(1.0, 0.8, 1.0).normalize() } else { dir };
        self.camera.target = center;
        self.camera.eye = center + dir * (radius * 2.5).max(3.0);
    }

    /// Rebuilds canvas-dependent GPU state (mesh + bounding box) and resets the
    /// pending UI size to match the current chunk. Call after the chunk's
    /// dimensions change (resize, load, import).
    pub(super) fn sync_to_chunk(&mut self) {
        self.pending_size = [self.chunk.width, self.chunk.height, self.chunk.depth];

        use wgpu::util::DeviceExt;
        let line_vertices = bounding_box_lines(&self.chunk);
        self.num_line_vertices = line_vertices.len() as u32;
        self.line_vertex_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        );
        self.remesh();
    }

    /// Marks the mesh out of date. The rebuild itself happens once per frame in
    /// [`flush_mesh`](Self::flush_mesh) — a fast drag delivers several cursor
    /// events per frame, and re-meshing the whole canvas on each of them is what
    /// made large canvases unusable.
    pub(super) fn remesh(&mut self) {
        self.mesh_dirty = true;
    }

    /// Rebuilds the voxel mesh if anything has changed since the last frame.
    pub(super) fn flush_mesh(&mut self) {
        if !self.mesh_dirty {
            return;
        }
        self.mesh_dirty = false;
        let (vertices, indices) = crate::render::mesh_chunk(&self.chunk, &self.palette);
        self.num_indices = indices.len() as u32;
        upload_growable(
            &self.device,
            &self.queue,
            &mut self.vertex_buffer,
            bytemuck::cast_slice(&vertices),
            wgpu::BufferUsages::VERTEX,
            "Vertex Buffer",
        );
        upload_growable(
            &self.device,
            &self.queue,
            &mut self.index_buffer,
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
            "Index Buffer",
        );
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

}
