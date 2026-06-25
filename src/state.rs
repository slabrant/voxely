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
    num_line_indices: u32,
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
    drag_start_voxels: Vec<[i32; 3]>,
    rect_drag: Option<RectDrag>,
    extrude_steps: Option<i32>,
    tool: Tool,
    /// Canvas dimensions edited in the UI, applied on "Resize".
    pending_size: [usize; 3],
    /// Path of the current `.obj` model, if it has been saved/opened. `Save`
    /// writes here silently; `Save As` (or saving when this is `None`) prompts.
    current_path: Option<std::path::PathBuf>,
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

pub struct Texture {
    pub view: wgpu::TextureView,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let _sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self { view }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}


impl State {
    pub async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // The surface must not outlive the window it was created from. We hand
        // `create_surface` an owned `Arc<Window>` clone, so the surface keeps the
        // window alive for as long as it lives.
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: wgpu::Limits::default(),
            },
            None, // trace_path
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non-sRGB
        // surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let camera = Camera {
            eye: (20.0, 20.0, 20.0).into(),
            target: (8.0, 0.0, 8.0).into(),
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_projection(&camera);

        use wgpu::util::DeviceExt;
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_line",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Translucent overlay drawn on top of the scene for the hover-face
        // highlight and the rectangle preview. Depth-tested (LessEqual) so it
        // sits on the surface but never writes depth, and alpha-blended.
        let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Highlight Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_highlight",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // X-ray pipeline for parts of the overlay that are occluded by geometry.
        // Uses Greater depth comparison and a dimmer fragment shader.
        let xray_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("X-ray Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_xray",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let camera_controller = CameraController::new();

        // egui: immediate-mode UI overlaid on top of the scene.
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1);

        let palette = Palette::default();

        let chunk = crate::core::Chunk::new();
        let (chunk_w, chunk_h, chunk_d) = (chunk.width, chunk.height, chunk.depth);

        let (mesh_vertices, mesh_indices) = crate::render::mesh_chunk(&chunk, &palette);

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh_indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        let num_indices = mesh_indices.len() as u32;

        let line_vertices = bounding_box_lines(&chunk);
        let num_line_indices = line_vertices.len() as u32;

        let line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        // Dynamic overlay buffer (hover highlight / rect preview). Seeded with a
        // single dummy vertex so the buffer is always valid; the actual contents
        // are rebuilt as the cursor moves. `num_overlay_vertices` of 0 means
        // "nothing to draw".
        let overlay_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Overlay Vertex Buffer"),
                contents: bytemuck::cast_slice(&[Vertex {
                    position: [0.0; 3],
                    color: [0.0; 3],
                    normal: [0.0; 3],
                }]),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            line_pipeline,
            highlight_pipeline,
            xray_pipeline,
            vertex_buffer,
            index_buffer,
            line_vertex_buffer,
            overlay_vertex_buffer,
            num_indices,
            num_line_indices,
            num_overlay_vertices: 0,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            depth_texture,
            chunk,
            palette,
            history: EditHistory::default(),
            cursor_position: winit::dpi::PhysicalPosition::new(0.0, 0.0),
            current_color_index: 1,
            modifiers: ModifiersState::empty(),
            egui_ctx,
            egui_state,
            egui_renderer,
            is_left_mouse_pressed: false,
            drag_erase: false,
            is_eyedropping: false,
            eyedropper_armed: false,
            is_erasing_gesture: false,
            erased_cells: std::collections::HashSet::new(),
            last_grid_coord: None,
            extrude_start: None,
            move_start: None,
            move_steps: None,
            drag_start_voxels: Vec::new(),
            rect_drag: None,
            extrude_steps: None,
            tool: Tool::Build,
            pending_size: [chunk_w, chunk_h, chunk_d],
            current_path: None,
        }
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

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        // A file dropped onto the window opens/imports it. Handle this before
        // egui so the drop always loads the model regardless of cursor position.
        if let WindowEvent::DroppedFile(path) = event {
            self.load_path(path);
            return true;
        }

        // Tab cycles tools (Shift+Tab reverse). Intercept it *before* egui,
        // which would otherwise consume Tab to move focus between panel
        // widgets. While egui is capturing the keyboard (e.g. editing a
        // canvas-size field) we defer, so Tab still moves between fields there.
        if let WindowEvent::KeyboardInput {
            event:
                winit::event::KeyEvent {
                    state: winit::event::ElementState::Pressed,
                    physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Tab),
                    repeat: false,
                    ..
                },
            ..
        } = event
        {
            if !self.egui_ctx.wants_keyboard_input() {
                self.cycle_tool(self.modifiers.shift_key());
                return true;
            }
        }

        // Let egui handle the event first; if it's consumed (e.g. a click on a
        // panel), don't also treat it as a scene/camera interaction.
        if self.egui_state.on_window_event(&self.window, event).consumed {
            return true;
        }
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - self.cursor_position.x) as f32;
                let dy = (position.y - self.cursor_position.y) as f32;
                self.camera_controller.handle_mouse_motion(dx, dy);
                self.cursor_position = *position;
                if self.is_left_mouse_pressed && !self.is_eyedropping {
                    let erase = self.drag_erase;
                    if self.rect_drag.is_some() {
                        self.update_rect();
                    } else if self.tool == Tool::Extrude || self.tool == Tool::Move {
                        // Both are click-and-drag along an axis; the live preview
                        // is rebuilt in `update_overlay`, so nothing to do here.
                        self.handle_extrude_drag(erase);
                    } else if self.tool != Tool::Bucket {
                        // Build and Paint both stroke along the drag; Bucket is
                        // click-only.
                        self.handle_drag(erase);
                    }
                }
                self.update_overlay();
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == winit::event::ElementState::Pressed;
                match button {
                    // The left button carries every editing gesture, with
                    // modifiers composing on top of the active tool (and the
                    // click samples color when the eyedropper is armed):
                    //   Alt        -> eyedropper (where the WM allows it)
                    //   Shift      -> erase
                    //   Ctrl       -> rectangle (Build)
                    //   Ctrl+Shift -> erase rectangle (Build)
                    // The right button is camera-orbit only.
                    winit::event::MouseButton::Left => {
                        if pressed {
                            let ctrl = self.modifiers.control_key();
                            let shift = self.modifiers.shift_key();
                            let alt = self.modifiers.alt_key();
                            self.is_left_mouse_pressed = true;
                            if self.eyedropper_armed || (alt && !ctrl && !shift) {
                                // Eyedropper: adopt the color under the cursor,
                                // either armed (tapped `I`) or via Alt+Left where
                                // the WM allows it. No edit -> no history.
                                self.pick_color();
                                self.eyedropper_armed = false;
                                self.is_eyedropping = true;
                            } else {
                                let erase = shift;
                                self.drag_erase = erase;
                                self.is_erasing_gesture = false;
                                self.erased_cells.clear();
                                self.drag_start_voxels.clear();
                                self.last_grid_coord = None;
                                // Every gesture (click, drag, rectangle, bucket)
                                // is one undo step; bracket it here, close on release.
                                self.history.begin_group();
                                if self.tool == Tool::Bucket {
                                    self.handle_bucket(erase);
                                } else if self.tool == Tool::Extrude {
                                    self.handle_extrude_click(erase);
                                } else if self.tool == Tool::Move {
                                    // Grab the whole connected object under the
                                    // cursor; the drag slides it (commit on release).
                                    self.handle_move_click();
                                } else if self.tool == Tool::Build && ctrl {
                                    // Ctrl+Left-drag fills a plane-locked
                                    // rectangle; +Shift erases it instead.
                                    self.begin_rect(erase);
                                } else {
                                    // Freehand stroke. An erase stroke carves the
                                    // visible surface without drilling through:
                                    // see `is_erasing_gesture` / `raycast`.
                                    self.is_erasing_gesture = erase;
                                    self.handle_click(erase);
                                }
                            }
                        } else {
                            self.is_left_mouse_pressed = false;
                            if self.is_eyedropping {
                                self.is_eyedropping = false;
                                self.last_grid_coord = None;
                            } else {
                                // Releasing the button commits any in-progress rectangle.
                                if self.rect_drag.is_some() {
                                    self.commit_rect();
                                }
                                if self.extrude_start.is_some() {
                                    self.commit_extrude();
                                }
                                if self.move_start.is_some() {
                                    self.commit_move();
                                }
                                self.extrude_start = None;
                                self.extrude_steps = None;
                                self.move_start = None;
                                self.move_steps = None;
                                self.history.end_group();
                                self.last_grid_coord = None;
                                self.drag_start_voxels.clear();
                                self.is_erasing_gesture = false;
                                self.erased_cells.clear();
                            }
                        }
                        self.update_overlay();
                        self.camera_controller.process_events(event)
                    }
                    _ => self.camera_controller.process_events(event),
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                false
            }
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    state,
                    physical_key: winit::keyboard::PhysicalKey::Code(key),
                    repeat,
                    ..
                },
                ..
            } => {
                use winit::keyboard::KeyCode;
                let pressed = *state == winit::event::ElementState::Pressed;
                let ctrl = self.modifiers.control_key();

                if pressed {
                    match key {
                        KeyCode::Digit1 if !*repeat => { self.current_color_index = 1; return true; }
                        KeyCode::Digit2 if !*repeat => { self.current_color_index = 2; return true; }
                        KeyCode::Digit3 if !*repeat => { self.current_color_index = 3; return true; }
                        KeyCode::Digit4 if !*repeat => { self.current_color_index = 4; return true; }
                        KeyCode::Digit5 if !*repeat => { self.current_color_index = 5; return true; }
                        KeyCode::Digit6 if !*repeat => { self.current_color_index = 6; return true; }
                        KeyCode::Digit7 if !*repeat => { self.current_color_index = 7; return true; }
                        KeyCode::Digit8 if !*repeat => { self.current_color_index = 8; return true; }
                        KeyCode::Digit9 if !*repeat => { self.current_color_index = 9; return true; }
                        KeyCode::KeyQ if !ctrl && !*repeat => {
                            // Arm the eyedropper: the next left-click samples the
                            // color under the cursor. Tap again to cancel. Q sits
                            // just below Tab, in the tool/color key cluster.
                            self.eyedropper_armed = !self.eyedropper_armed;
                            return true;
                        }
                        KeyCode::Escape if !*repeat && self.eyedropper_armed => {
                            self.eyedropper_armed = false;
                            return true;
                        }
                        KeyCode::KeyS if ctrl && self.modifiers.shift_key() && !*repeat => { self.save_project_as(); return true; }
                        KeyCode::KeyS if ctrl && !*repeat => { self.save_project(); return true; }
                        KeyCode::KeyO if ctrl && !*repeat => { self.open_file(); return true; }
                        // One undo/redo per press: the `!*repeat` guard ignores
                        // the OS key-repeat stream while the key is held. The
                        // Ctrl+Shift+Z redo arm must precede the plain Ctrl+Z.
                        KeyCode::KeyZ if ctrl && self.modifiers.shift_key() && !*repeat => { self.redo(); return true; }
                        KeyCode::KeyZ if ctrl && !*repeat => { self.undo(); return true; }
                        KeyCode::KeyY if ctrl && !*repeat => { self.redo(); return true; }
                        _ => {}
                    }
                }
                // Forward every press AND release to the camera controller so
                // held movement keys don't get stuck on (releases must arrive).
                self.camera_controller.process_events(event)
            }
            WindowEvent::MouseWheel { .. } => {
                self.camera_controller.process_events(event)
            }
            _ => false,
        }
    }

    /// Switches to the next tool, or the previous one when `reverse` is set
    /// (Shift+Tab). Order: Build → Paint → Bucket → Extrude → Move → Build.
    fn cycle_tool(&mut self, reverse: bool) {
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

    /// Applies one click. `erase` (Shift held) clears the hit voxel; otherwise
    /// the active tool decides: Build places against the hit face, Paint
    /// recolors the hit voxel in place (never adds *or* removes geometry).
    fn handle_click(&mut self, erase: bool) {
        // Paint only ever recolors, so erasing in Paint mode is a no-op; switch
        // to Build to remove geometry.
        if erase && self.tool == Tool::Paint {
            return;
        }
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, is_drag_voxel)) = self.raycast(ray_origin, ray_dir) {
            let paint = self.tool == Tool::Paint;
            let coord = if erase || paint || is_drag_voxel {
                // Erase and paint act on the hit voxel itself; so does a hit on
                // a voxel from this same drag session (prevents staircasing).
                pos
            } else {
                // Place a new voxel against the hit face, one cell along the normal.
                [
                    pos[0] + normal[0] as i32,
                    pos[1] + normal[1] as i32,
                    pos[2] + normal[2] as i32,
                ]
            };
            if Some(coord) == self.last_grid_coord {
                return;
            }
            // Paint only recolors solid voxels; it never fills empty space.
            if paint && !erase && !self.solid_at(coord) {
                return;
            }
            let color_index = if erase { 0 } else { self.current_color_index };
            if self.set_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, false) {
                self.last_grid_coord = Some(coord);
            }
        }
    }

    /// Continues a stroke from the last grid cell to the one under the cursor,
    /// filling the gap with a 3D line so fast drags don't leave holes. `erase`
    /// and the active tool mean the same thing as in [`handle_click`].
    fn handle_drag(&mut self, erase: bool) {
        if erase && self.tool == Tool::Paint {
            return;
        }
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, is_drag_voxel)) = self.raycast(ray_origin, ray_dir) {
            let paint = self.tool == Tool::Paint;
            let current_coord = if erase || paint || is_drag_voxel {
                pos
            } else {
                [
                    pos[0] + normal[0] as i32,
                    pos[1] + normal[1] as i32,
                    pos[2] + normal[2] as i32,
                ]
            };
            let color_index = if erase { 0 } else { self.current_color_index };

            // Bulk-write the stroke and remesh once at the end (see `write_voxel`).
            let mut changed = false;
            if let Some(last_coord) = self.last_grid_coord {
                if current_coord != last_coord {
                    // 3D DDA Line Algorithm
                    let x1 = last_coord[0] as f32;
                    let y1 = last_coord[1] as f32;
                    let z1 = last_coord[2] as f32;
                    let x2 = current_coord[0] as f32;
                    let y2 = current_coord[1] as f32;
                    let z2 = current_coord[2] as f32;

                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let dz = z2 - z1;

                    let steps = dx.abs().max(dy.abs()).max(dz.abs());
                    if steps > 0.0 {
                        let x_inc = dx / steps;
                        let y_inc = dy / steps;
                        let z_inc = dz / steps;

                        let mut cx = x1;
                        let mut cy = y1;
                        let mut cz = z1;

                        for i in 0..=steps as i32 {
                            let coord = [cx.round() as i32, cy.round() as i32, cz.round() as i32];
                            cx += x_inc;
                            cy += y_inc;
                            cz += z_inc;
                            // Skip the first coordinate if it's the one we just placed in handle_click
                            // or the last one from handle_drag.
                            if i == 0 && Some(coord) == self.last_grid_coord {
                                continue;
                            }
                            // Paint skips empty cells along the line so a stroke
                            // recolors only the voxels it actually crosses.
                            if paint && !erase && !self.solid_at(coord) {
                                continue;
                            }
                            changed |= self.write_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, true);
                        }
                    }
                    self.last_grid_coord = Some(current_coord);
                }
            } else {
                if !(paint && !erase && !self.solid_at(current_coord)) {
                    changed |= self.write_voxel(current_coord[0], current_coord[1], current_coord[2], Voxel { color_index }, false);
                }
                self.last_grid_coord = Some(current_coord);
            }
            if changed {
                self.remesh();
            }
        }
    }

    /// Casts the cursor ray and starts a rectangle drag, locking the fill plane
    /// to the face that was hit. `remove` chooses fill vs. erase. Does nothing
    /// if the ray misses the world.
    fn begin_rect(&mut self, remove: bool) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, _)) = self.raycast(ray_origin, ray_dir) {
            let axis = if normal[0] != 0.0 {
                0
            } else if normal[1] != 0.0 {
                1
            } else {
                2
            };
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            // Erase acts on the hit voxel's own plane; place acts on the cell
            // one step along the normal.
            let place_value = if remove { pos[axis] } else { pos[axis] + normal[axis] as i32 };
            // World plane of the hit face: at the high side of the cell for a
            // positive normal, the low side for a negative one.
            let face_world = if normal[axis] > 0.0 {
                (pos[axis] + 1) as f32
            } else {
                pos[axis] as f32
            };

            let (u, v) = self.project_to_plane(ray_origin, ray_dir, axis, u_axis, v_axis, face_world);
            self.rect_drag = Some(RectDrag {
                axis,
                u_axis,
                v_axis,
                face_world,
                normal,
                place_value,
                start_u: u,
                start_v: v,
                cur_u: u,
                cur_v: v,
                remove,
            });
        }
    }

    /// Updates the moving corner of an in-progress rectangle by projecting the
    /// cursor ray onto the locked fill plane.
    fn update_rect(&mut self) {
        let Some(rd) = self.rect_drag else { return };
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;
        let (u, v) = self.project_to_plane(ray_origin, ray_dir, rd.axis, rd.u_axis, rd.v_axis, rd.face_world);
        if let Some(rd) = &mut self.rect_drag {
            rd.cur_u = u;
            rd.cur_v = v;
        }
    }

    /// Intersects a ray with the axis-aligned plane `coord[axis] == face_world`
    /// and returns the in-plane cell indices (clamped to the chunk bounds).
    fn project_to_plane(
        &self,
        origin: glam::Vec3,
        dir: glam::Vec3,
        axis: usize,
        u_axis: usize,
        v_axis: usize,
        face_world: f32,
    ) -> (i32, i32) {
        let dims = [self.chunk.width as i32, self.chunk.height as i32, self.chunk.depth as i32];
        let o = origin.to_array();
        let d = dir.to_array();
        if d[axis].abs() < 1e-6 {
            return (0, 0);
        }
        let t = (face_world - o[axis]) / d[axis];
        let p = origin + dir * t;
        let p = p.to_array();
        let u = (p[u_axis].floor() as i32).clamp(0, dims[u_axis] - 1);
        let v = (p[v_axis].floor() as i32).clamp(0, dims[v_axis] - 1);
        (u, v)
    }

    /// Writes every cell of the finished rectangle, then clears the drag state.
    fn commit_rect(&mut self) {
        let Some(rd) = self.rect_drag.take() else { return };
        let (u0, u1) = (rd.start_u.min(rd.cur_u), rd.start_u.max(rd.cur_u));
        let (v0, v1) = (rd.start_v.min(rd.cur_v), rd.start_v.max(rd.cur_v));
        let color_index = if rd.remove { 0 } else { self.current_color_index };
        let mut coord = [0i32; 3];
        coord[rd.axis] = rd.place_value;
        let mut changed = false;
        for u in u0..=u1 {
            for v in v0..=v1 {
                coord[rd.u_axis] = u;
                coord[rd.v_axis] = v;
                changed |= self.write_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, true);
            }
        }
        if changed {
            self.remesh();
        }
    }

    /// Paint-bucket flood fill: recolors the connected region (6-adjacency) of
    /// voxels sharing the clicked voxel's color. `erase` (Shift held) clears the
    /// region instead. The whole flood is recorded as a single undo group by the
    /// caller. Does nothing if the cursor isn't over a solid voxel.
    fn handle_bucket(&mut self, erase: bool) {
        let (cw, ch, cd) = (self.chunk.width, self.chunk.height, self.chunk.depth);
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        let Some((pos, _normal, _)) = self.raycast(ray_origin, ray_dir) else { return };
        if pos[0] < 0 || pos[1] < 0 || pos[2] < 0 {
            return; // boundary-only hit, no voxel to fill
        }
        let (sx, sy, sz) = (pos[0] as usize, pos[1] as usize, pos[2] as usize);
        let target = match self.chunk.get(sx, sy, sz) {
            Some(v) if !v.is_empty() => v.color_index,
            _ => return,
        };
        let new_color = if erase { 0 } else { self.current_color_index };
        if new_color == target {
            return;
        }

        // Iterative flood fill. Mutate the chunk directly and record each change
        // so we can remesh just once at the end.
        let mut stack = vec![[sx, sy, sz]];
        let mut changed = false;
        while let Some([x, y, z]) = stack.pop() {
            let Some(v) = self.chunk.get(x, y, z) else { continue };
            if v.is_empty() || v.color_index != target {
                continue;
            }
            let old = *v;
            self.chunk.set(x, y, z, Voxel { color_index: new_color });
            self.history.record(VoxelEdit {
                x,
                y,
                z,
                old,
                new: Voxel { color_index: new_color },
            });
            changed = true;

            if x + 1 < cw { stack.push([x + 1, y, z]); }
            if x > 0 { stack.push([x - 1, y, z]); }
            if y + 1 < ch { stack.push([x, y + 1, z]); }
            if y > 0 { stack.push([x, y - 1, z]); }
            if z + 1 < cd { stack.push([x, y, z + 1]); }
            if z > 0 { stack.push([x, y, z - 1]); }
        }

        if changed {
            self.remesh();
        }
    }

    fn handle_extrude_click(&mut self, erase: bool) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, _)) = self.raycast(ray_origin, ray_dir) {
            if let Some(v) = self.chunk.get(pos[0] as usize, pos[1] as usize, pos[2] as usize) {
                if !v.is_empty() {
                    let normal_i = [normal[0] as i32, normal[1] as i32, normal[2] as i32];
                    let target_color = v.color_index;
                    
                    // Flood fill on the surface to find all connected voxels of the same color
                    // that have the same exposed face.
                    let mut surface_voxels = Vec::new();
                    let mut stack = vec![pos];
                    let mut visited = std::collections::HashSet::new();
                    visited.insert(pos);
                    
                    let axis = if normal_i[0] != 0 { 0 } else if normal_i[1] != 0 { 1 } else { 2 };
                    let u_axis = (axis + 1) % 3;
                    let v_axis = (axis + 2) % 3;

                    while let Some(curr) = stack.pop() {
                        surface_voxels.push(curr);
                        
                        // Check 4 neighbors on the same plane
                        for (du, dv) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                            let mut next = curr;
                            next[u_axis] += du;
                            next[v_axis] += dv;
                            
                            if !visited.contains(&next) {
                                if let Some(nv) = self.chunk.get(next[0] as usize, next[1] as usize, next[2] as usize) {
                                    if nv.color_index == target_color {
                                        // Check if this voxel also has an exposed face in the same direction.
                                        // A face is exposed if the neighbor in the normal direction is empty.
                                        let neighbor_pos = [
                                            next[0] + normal_i[0],
                                            next[1] + normal_i[1],
                                            next[2] + normal_i[2],
                                        ];
                                        let is_exposed = self.chunk.get(
                                            neighbor_pos[0] as i32 as usize,
                                            neighbor_pos[1] as i32 as usize,
                                            neighbor_pos[2] as i32 as usize
                                        ).map(|v| v.is_empty()).unwrap_or(true);
                                        
                                        if is_exposed {
                                            visited.insert(next);
                                            stack.push(next);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let color_index = if erase { 0 } else { target_color };
                    self.extrude_start = Some((surface_voxels, normal_i, color_index));
                    self.extrude_steps = Some(0);
                    self.last_grid_coord = Some(pos);
                }
            }
        }
    }

    fn handle_extrude_drag(&mut self, _erase: bool) {
        // No-op during drag; we just use the overlay to show where it will be.
    }

    /// How many voxel steps the cursor currently maps to along an axis. Finds
    /// the point on the axis (the line through `start_center` along the unit
    /// `normal`) closest to the cursor ray, and returns its signed distance from
    /// `start_center` in voxel units, rounded. `fallback` is returned when the
    /// ray sights nearly down the axis (depth ill-defined). Shared by Extrude
    /// (layers pulled) and Move (cells slid).
    ///
    /// This is the standard closest-points-of-two-lines result. Because it
    /// measures displacement *along the axis* rather than projecting the cursor
    /// out to the start voxel's eye-distance (the old approach), screen motion
    /// tracks the distance roughly 1:1 instead of needing a long drag.
    fn steps_along_axis(&self, start_center: glam::Vec3, normal: glam::Vec3, fallback: i32) -> i32 {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let w0 = start_center - self.camera.eye;
        // `normal` and `ray_dir` are both unit length, so a = c = 1.
        let b = normal.dot(ray_dir); // n·d == cos(angle between axis and ray)
        let denom = 1.0 - b * b; // sin²(angle); → 0 when sighting down the axis
        if denom.abs() < 1e-3 {
            // Ray nearly parallel to the axis: depth is ill-defined, so hold the
            // last value rather than jumping.
            return fallback;
        }
        let d = normal.dot(w0);
        let e = ray_dir.dot(w0);
        ((b * e - d) / denom).round() as i32
    }

    fn commit_extrude(&mut self) {
        let (surface_voxels, normal, color_index) = match &self.extrude_start {
            Some(s) => s,
            None => return,
        };
        let surface_voxels = surface_voxels.clone();
        let normal = *normal;
        let color_index = *color_index;

        let start_pos = surface_voxels[0];
        let start_center = glam::Vec3::new(
            start_pos[0] as f32 + 0.5,
            start_pos[1] as f32 + 0.5,
            start_pos[2] as f32 + 0.5,
        );
        let normal_v = glam::Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32);
        let steps = self.steps_along_axis(start_center, normal_v, self.extrude_steps.unwrap_or(0));
        if steps == 0 {
            return;
        }

        // Write every layer, then remesh once: extruding many voxels otherwise
        // rebuilt the whole mesh per voxel and stalled the app.
        let mut changed = false;
        if steps > 0 {
            for i in 1..=steps {
                for &pos in &surface_voxels {
                    let coord = [
                        pos[0] + normal[0] * i,
                        pos[1] + normal[1] * i,
                        pos[2] + normal[2] * i,
                    ];
                    changed |= self.write_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, true);
                }
            }
        } else {
            // Negative steps means unextruding. Erase from i=0 down to steps.
            // i=0 is the current surface.
            for i in 0..steps.abs() {
                for &pos in &surface_voxels {
                    let coord = [
                        pos[0] - normal[0] * i,
                        pos[1] - normal[1] * i,
                        pos[2] - normal[2] * i,
                    ];
                    changed |= self.write_voxel(coord[0], coord[1], coord[2], Voxel { color_index: 0 }, true);
                }
            }
        }
        if changed {
            self.remesh();
        }
    }

    /// Grabs the connected object under the cursor for a Move drag. The object
    /// is every solid voxel reachable from the hit one through shared faces
    /// (6-neighbor flood fill), captured with its colors. The clicked face's
    /// normal becomes the slide axis. Does nothing if the cursor isn't over a
    /// solid voxel.
    fn handle_move_click(&mut self) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        let Some((pos, normal, _)) = self.raycast(ray_origin, ray_dir) else {
            return;
        };
        if !self.solid_at(pos) {
            return;
        }
        let normal_i = [normal[0] as i32, normal[1] as i32, normal[2] as i32];
        if normal_i == [0, 0, 0] {
            // Boundary-only hit with no entered face; no axis to slide along.
            return;
        }

        // Flood-fill the whole connected object across shared faces.
        let mut voxels = Vec::new();
        let mut stack = vec![pos];
        let mut visited = std::collections::HashSet::new();
        visited.insert(pos);
        while let Some(curr) = stack.pop() {
            let color = self
                .chunk
                .get(curr[0] as usize, curr[1] as usize, curr[2] as usize)
                .map(|v| v.color_index)
                .unwrap_or(0);
            voxels.push((curr, color));
            for (axis, delta) in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
                let mut next = curr;
                next[axis] += delta;
                if !visited.contains(&next) && self.solid_at(next) {
                    visited.insert(next);
                    stack.push(next);
                }
            }
        }

        let anchor_center = glam::Vec3::new(
            pos[0] as f32 + 0.5,
            pos[1] as f32 + 0.5,
            pos[2] as f32 + 0.5,
        );
        self.move_start = Some(MoveDrag { voxels, normal: normal_i, anchor_center });
        self.move_steps = Some(0);
        self.last_grid_coord = Some(pos);
    }

    /// Caps a raw Move step count so no voxel of the grabbed object would leave
    /// the canvas: the shape slides until its leading edge meets the wall, then
    /// stops, rather than having edge voxels clipped off. Movement is purely
    /// along the slide axis, so this is a 1-D clamp on the displacement that
    /// keeps the object's extent on that axis within `[0, dim)`.
    fn clamp_move_steps(&self, md: &MoveDrag, steps: i32) -> i32 {
        let axis = if md.normal[0] != 0 {
            0
        } else if md.normal[1] != 0 {
            1
        } else {
            2
        };
        let dim = match axis {
            0 => self.chunk.width,
            1 => self.chunk.height,
            _ => self.chunk.depth,
        } as i32;
        let (mut min_p, mut max_p) = (i32::MAX, i32::MIN);
        for (p, _) in &md.voxels {
            min_p = min_p.min(p[axis]);
            max_p = max_p.max(p[axis]);
        }
        // Displacement along the axis must keep [min_p, max_p] inside [0, dim).
        let s = md.normal[axis]; // +1 or -1
        let delta = (s * steps).clamp(-min_p, (dim - 1) - max_p);
        s * delta
    }

    /// Applies the in-progress Move: clears the object's original cells, then
    /// writes it back shifted along the slide axis. The shift is capped so the
    /// whole object stays in-bounds (see [`clamp_move_steps`]); at the
    /// destination it overwrites whatever was there.
    ///
    /// [`clamp_move_steps`]: Self::clamp_move_steps
    fn commit_move(&mut self) {
        let md = match &self.move_start {
            Some(m) => m,
            None => return,
        };
        let voxels = md.voxels.clone();
        let normal = md.normal;
        let normal_v = glam::Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32);
        let raw = self.steps_along_axis(md.anchor_center, normal_v, self.move_steps.unwrap_or(0));
        let steps = self.clamp_move_steps(md, raw);
        if steps == 0 {
            return;
        }
        let offset = [normal[0] * steps, normal[1] * steps, normal[2] * steps];

        // Clear the originals first so cells the object vacates end up empty even
        // when another part of the same object slides onto them.
        let mut changed = false;
        for (pos, _) in &voxels {
            changed |= self.write_voxel(pos[0], pos[1], pos[2], Voxel { color_index: 0 }, false);
        }
        for (pos, color) in &voxels {
            changed |= self.write_voxel(
                pos[0] + offset[0],
                pos[1] + offset[1],
                pos[2] + offset[2],
                Voxel { color_index: *color },
                false,
            );
        }
        if changed {
            self.remesh();
        }
    }

    /// Rebuilds the overlay vertex buffer for the current frame: the rectangle
    /// preview while dragging, otherwise the hover-face highlight under the
    /// cursor. Empty when the cursor isn't over the world.
    fn update_overlay(&mut self) {
        let mut verts: Vec<Vertex> = Vec::new();

        if let Some(rd) = self.rect_drag {
            let (u0, u1) = (rd.start_u.min(rd.cur_u), rd.start_u.max(rd.cur_u));
            let (v0, v1) = (rd.start_v.min(rd.cur_v), rd.start_v.max(rd.cur_v));
            let color = if rd.remove { [1.0, 0.3, 0.3] } else { [0.4, 0.8, 1.0] };
            let mut base = [0.0f32; 3];
            base[rd.axis] = rd.face_world;
            base[rd.u_axis] = u0 as f32;
            base[rd.v_axis] = v0 as f32;
            let mut du = [0.0f32; 3];
            du[rd.u_axis] = (u1 - u0 + 1) as f32;
            let mut dv = [0.0f32; 3];
            dv[rd.v_axis] = (v1 - v0 + 1) as f32;
            push_overlay_quad(&mut verts, base, du, dv, rd.normal, color);
        } else if let Some((surface_voxels, normal, _)) = &self.extrude_start {
            let start_pos = surface_voxels[0];
            let start_center = glam::Vec3::new(
                start_pos[0] as f32 + 0.5,
                start_pos[1] as f32 + 0.5,
                start_pos[2] as f32 + 0.5,
            );
            let normal_v = glam::Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32);
            let steps = self.steps_along_axis(start_center, normal_v, self.extrude_steps.unwrap_or(0));

            if Some(steps) == self.extrude_steps {
                return;
            }
            self.extrude_steps = Some(steps);
            self.num_overlay_vertices = 0; // Force a rebuild if we didn't return early

            let color = if self.drag_erase { [1.0, 0.3, 0.3] } else { [0.4, 0.8, 1.0] };
            let axis = if normal[0] != 0 { 0 } else if normal[1] != 0 { 1 } else { 2 };
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            let face_normal = [normal[0] as f32, normal[1] as f32, normal[2] as f32];

            let sign = if steps > 0 { 1 } else { -1 };
            let offset = steps.abs() * sign;
            
            for &pos in surface_voxels {
                let mut base = [0.0f32; 3];
                // The face world depends on the normal.
                let face_world = if normal[axis] > 0 {
                    (pos[axis] + 1 + offset) as f32
                } else {
                    (pos[axis] - offset) as f32
                };
                
                base[axis] = face_world;
                base[u_axis] = pos[u_axis] as f32;
                base[v_axis] = pos[v_axis] as f32;
                let mut du = [0.0f32; 3];
                du[u_axis] = 1.0;
                let mut dv = [0.0f32; 3];
                dv[v_axis] = 1.0;
                push_overlay_quad(&mut verts, base, du, dv, face_normal, color);
            }
        } else if let Some(md) = &self.move_start {
            // Preview the grabbed object at its slid position: draw the outer
            // hull (each face whose neighbor in the translated set is empty).
            let normal = md.normal;
            let normal_v = glam::Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32);
            let raw = self.steps_along_axis(md.anchor_center, normal_v, self.move_steps.unwrap_or(0));
            let steps = self.clamp_move_steps(md, raw);

            if Some(steps) == self.move_steps {
                return;
            }
            self.move_steps = Some(steps);
            self.num_overlay_vertices = 0; // Force a rebuild if we didn't return early.

            let offset = [normal[0] * steps, normal[1] * steps, normal[2] * steps];
            let moved: std::collections::HashSet<[i32; 3]> = md
                .voxels
                .iter()
                .map(|(p, _)| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]])
                .collect();
            let color = [1.0, 0.7, 0.2]; // orange, distinct from build/erase/extrude

            for &p in &moved {
                // One quad per exposed face (the six axis-aligned directions).
                for (axis, dir) in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)] {
                    let mut neighbor = p;
                    neighbor[axis] += dir;
                    if moved.contains(&neighbor) {
                        continue;
                    }
                    let u_axis = (axis + 1) % 3;
                    let v_axis = (axis + 2) % 3;
                    let mut base = [0.0f32; 3];
                    base[axis] = if dir > 0 { (p[axis] + 1) as f32 } else { p[axis] as f32 };
                    base[u_axis] = p[u_axis] as f32;
                    base[v_axis] = p[v_axis] as f32;
                    let mut du = [0.0f32; 3];
                    du[u_axis] = 1.0;
                    let mut dv = [0.0f32; 3];
                    dv[v_axis] = 1.0;
                    let mut face_normal = [0.0f32; 3];
                    face_normal[axis] = dir as f32;
                    push_overlay_quad(&mut verts, base, du, dv, face_normal, color);
                }
            }
        } else if !self.is_left_mouse_pressed {
            let ray_dir = self.screen_to_ray(self.cursor_position);
            let ray_origin = self.camera.eye;
            if let Some((pos, normal, _)) = self.raycast(ray_origin, ray_dir) {
                let axis = if normal[0] != 0.0 {
                    0
                } else if normal[1] != 0.0 {
                    1
                } else {
                    2
                };
                let u_axis = (axis + 1) % 3;
                let v_axis = (axis + 2) % 3;
                let face_world = if normal[axis] > 0.0 {
                    (pos[axis] + 1) as f32
                } else {
                    pos[axis] as f32
                };
                // Green marks the voxel the eyedropper would sample (armed, or
                // Alt held); red marks the erase target (Shift).
                let color = if self.eyedropper_armed || self.modifiers.alt_key() {
                    [0.4, 1.0, 0.4]
                } else if self.modifiers.shift_key() {
                    [1.0, 0.4, 0.4]
                } else {
                    [1.0, 1.0, 1.0]
                };
                let mut base = [0.0f32; 3];
                base[axis] = face_world;
                base[u_axis] = pos[u_axis] as f32;
                base[v_axis] = pos[v_axis] as f32;
                let mut du = [0.0f32; 3];
                du[u_axis] = 1.0;
                let mut dv = [0.0f32; 3];
                dv[v_axis] = 1.0;
                push_overlay_quad(&mut verts, base, du, dv, normal, color);
            }
        }

        self.num_overlay_vertices = verts.len() as u32;
        if !verts.is_empty() {
            // Re-allocate the buffer only if the vertex count has grown, 
            // otherwise just update the existing one to avoid allocation lag.
            if verts.len() as u32 > self.overlay_vertex_buffer.size() as u32 / std::mem::size_of::<Vertex>() as u32 {
                use wgpu::util::DeviceExt;
                self.overlay_vertex_buffer = self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Overlay Vertex Buffer"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                );
            } else {
                self.queue.write_buffer(&self.overlay_vertex_buffer, 0, bytemuck::cast_slice(&verts));
            }
        }
    }

    /// Whether `coord` is in-bounds and holds a non-empty voxel.
    fn solid_at(&self, coord: [i32; 3]) -> bool {
        if coord[0] < 0 || coord[1] < 0 || coord[2] < 0 {
            return false;
        }
        self.chunk
            .get(coord[0] as usize, coord[1] as usize, coord[2] as usize)
            .is_some_and(|v| !v.is_empty())
    }

    /// Eyedropper: read the color index of the voxel under the cursor and make
    /// it the active color. Does nothing unless the cursor is over a solid voxel.
    fn pick_color(&mut self) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;
        let Some((pos, _normal, _)) = self.raycast(ray_origin, ray_dir) else {
            return;
        };
        if pos[0] < 0 || pos[1] < 0 || pos[2] < 0 {
            return; // boundary-only hit, no voxel to sample
        }
        if let Some(v) = self
            .chunk
            .get(pos[0] as usize, pos[1] as usize, pos[2] as usize)
            .filter(|v| !v.is_empty())
        {
            self.current_color_index = v.color_index;
        }
    }

    /// Core voxel write: updates the chunk and undo history, returning whether
    /// anything changed. Does *not* remesh, so callers doing bulk edits
    /// (rectangle, extrude, drag strokes) rebuild the mesh once at the end
    /// rather than once per voxel. Coordinates outside the chunk are ignored.
    fn write_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel, is_drag: bool) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        let (xu, yu, zu) = (x as usize, y as usize, z as usize);
        let old = match self.chunk.get(xu, yu, zu) {
            Some(v) => *v,
            None => return false,
        };
        if old == voxel {
            return false;
        }
        self.chunk.set(xu, yu, zu, voxel);

        if self.is_left_mouse_pressed && !voxel.is_empty() {
            // Voxels placed this drag session are excluded from raycasting.
            self.drag_start_voxels.push([x, y, z]);
        } else if self.is_erasing_gesture && voxel.is_empty() {
            // Cells cleared this erase stroke keep blocking the ray (see
            // `raycast`), so the stroke carves the surface without tunnelling.
            self.erased_cells.insert([x, y, z]);
        }

        if is_drag {
            self.history.record_continuous(VoxelEdit { x: xu, y: yu, z: zu, old, new: voxel });
        } else {
            self.history.record(VoxelEdit { x: xu, y: yu, z: zu, old, new: voxel });
        }
        true
    }

    /// Like [`write_voxel`](Self::write_voxel) but remeshes immediately. For
    /// one-off single-voxel edits (a click, the head of a stroke).
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel, is_drag: bool) -> bool {
        let changed = self.write_voxel(x, y, z, voxel, is_drag);
        if changed {
            self.remesh();
        }
        changed
    }

    fn undo(&mut self) {
        if let Some(group) = self.history.undo() {
            // Reverse order so coordinates touched more than once in a single
            // gesture land back on their original value.
            for e in group.iter().rev() {
                self.chunk.set(e.x, e.y, e.z, e.old);
            }
            self.remesh();
            println!("Undo");
        }
    }

    fn redo(&mut self) {
        if let Some(group) = self.history.redo() {
            for e in &group {
                self.chunk.set(e.x, e.y, e.z, e.new);
            }
            self.remesh();
            println!("Redo");
        }
    }

    fn screen_to_ray(&self, position: winit::dpi::PhysicalPosition<f64>) -> glam::Vec3 {
        let x = (2.0 * position.x as f32) / self.size.width as f32 - 1.0;
        let y = 1.0 - (2.0 * position.y as f32) / self.size.height as f32;
        
        let projection = glam::Mat4::perspective_rh(self.camera.fovy.to_radians(), self.camera.aspect, self.camera.znear, self.camera.zfar);
        let view = glam::Mat4::look_at_rh(self.camera.eye, self.camera.target, self.camera.up);
        let inv_vp = (projection * view).inverse();
        
        let clip_pos = glam::Vec4::new(x, y, -1.0, 1.0);
        let world_pos = inv_vp * clip_pos;
        let world_pos = world_pos.xyz() / world_pos.w;
        
        (world_pos - self.camera.eye).normalize()
    }

    fn raycast(&self, origin: glam::Vec3, dir: glam::Vec3) -> Option<([i32; 3], [f32; 3], bool)> {
        let max_dist = 100.0;

        // Amanatides–Woo voxel traversal. Unlike fixed-step sampling, this lands
        // on every cell the ray crosses in order and tracks exactly which face
        // we entered through, so the placement normal is always correct (no more
        // voxels embedded in the surface they were placed against).
        let o = origin.to_array();
        let d = dir.to_array();
        let mut ip = [o[0].floor() as i32, o[1].floor() as i32, o[2].floor() as i32];
        let mut step = [0i32; 3];
        let mut t_max = [f32::INFINITY; 3];
        let mut t_delta = [f32::INFINITY; 3];
        for i in 0..3 {
            if d[i] > 0.0 {
                step[i] = 1;
                t_max[i] = (ip[i] as f32 + 1.0 - o[i]) / d[i];
                t_delta[i] = 1.0 / d[i];
            } else if d[i] < 0.0 {
                step[i] = -1;
                t_max[i] = (ip[i] as f32 - o[i]) / d[i];
                t_delta[i] = -1.0 / d[i];
            }
        }

        // Normal of the face we entered the current cell through. Zero until the
        // first boundary crossing (i.e. while still in the origin's own cell).
        let mut normal = [0.0f32; 3];
        let mut t = 0.0;
        while t < max_dist {
            if ip[0] >= 0 && ip[1] >= 0 && ip[2] >= 0 {
                let solid = self
                    .chunk
                    .get(ip[0] as usize, ip[1] as usize, ip[2] as usize)
                    .is_some_and(|v| !v.is_empty());
                // During an erase stroke, cells already cleared this gesture
                // still stop the ray, so the stroke carves the surface it
                // started on instead of tunnelling through the model.
                let blocked = self.is_erasing_gesture && self.erased_cells.contains(&ip);
                if solid || blocked {
                    let is_drag_voxel = self.drag_start_voxels.contains(&ip);
                    return Some((ip, normal, is_drag_voxel));
                }
            }
            // Advance to the next cell along whichever axis has the nearest
            // boundary; the face we cross faces back the way we came.
            let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
                0
            } else if t_max[1] <= t_max[2] {
                1
            } else {
                2
            };
            ip[axis] += step[axis];
            t = t_max[axis];
            t_max[axis] += t_delta[axis];
            normal = [0.0; 3];
            normal[axis] = -step[axis] as f32;
        }

        // If no voxel was hit, check for intersection with the world's bounding box boundaries.
        // This allows building from the limits when the world is empty.
        let dims = [self.chunk.width as i32, self.chunk.height as i32, self.chunk.depth as i32];

        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;
        let mut hit_normal_near = [0.0; 3];
        let mut hit_normal_far = [0.0; 3];

        let bounds_min = [0.0, 0.0, 0.0];
        let bounds_max = [dims[0] as f32, dims[1] as f32, dims[2] as f32];
        let origin_arr = [origin.x, origin.y, origin.z];
        let dir_arr = [dir.x, dir.y, dir.z];

        for i in 0..3 {
            if dir_arr[i].abs() > 1e-6 {
                let t1 = (bounds_min[i] - origin_arr[i]) / dir_arr[i];
                let t2 = (bounds_max[i] - origin_arr[i]) / dir_arr[i];

                let (t_near, t_far, normal_near, normal_far) = if t1 < t2 { 
                    (t1, t2, -1.0, 1.0) 
                } else { 
                    (t2, t1, 1.0, -1.0) 
                };

                if t_near > t_min {
                    t_min = t_near;
                    hit_normal_near = [0.0; 3];
                    hit_normal_near[i] = normal_near;
                }
                if t_far < t_max {
                    t_max = t_far;
                    hit_normal_far = [0.0; 3];
                    hit_normal_far[i] = normal_far;
                }
            } else if origin_arr[i] < bounds_min[i] || origin_arr[i] > bounds_max[i] {
                return None;
            }
        }

        // Use t_max (farthest) if it's in front of us and within range,
        // otherwise fall back to t_min (nearest).
        let (t_hit, hit_normal) = if t_max > 0.0 && t_max < max_dist {
            (t_max, hit_normal_far)
        } else if t_min > 0.0 && t_min < t_max && t_min < max_dist {
            (t_min, hit_normal_near)
        } else {
            return None;
        };

        let p = origin + dir * t_hit;
        let mut ip = [
            p.x.floor() as i32,
            p.y.floor() as i32,
            p.z.floor() as i32,
        ];

        // Clamp to ensure it's within bounds and adjust for boundary hits
        for i in 0..3 {
            let dim = dims[i];
            if hit_normal[i] < 0.0 {
                ip[i] = 0;
            } else if hit_normal[i] > 0.0 {
                ip[i] = dim - 1;
            } else {
                ip[i] = ip[i].clamp(0, dim - 1);
            }
        }

        let mut coord = [0; 3];
        let mut inward_normal = [0.0; 3];
        for i in 0..3 {
            let dim = dims[i];
            if hit_normal[i] < 0.0 {
                coord[i] = -1;
                inward_normal[i] = 1.0;
            } else if hit_normal[i] > 0.0 {
                coord[i] = dim;
                inward_normal[i] = -1.0;
            } else {
                coord[i] = ip[i];
            }
        }

        return Some((coord, inward_normal, false));
    }

    /// Resizes the canvas, keeping voxels that still fit (anchored at the
    /// origin). Clears history since old coordinates may no longer be valid,
    /// then rebuilds the mesh and bounding-box wireframe.
    fn resize_canvas(&mut self, width: usize, height: usize, depth: usize) {
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
    fn frame_camera_to_chunk(&mut self) {
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
    fn sync_to_chunk(&mut self) {
        self.pending_size = [self.chunk.width, self.chunk.height, self.chunk.depth];

        use wgpu::util::DeviceExt;
        let line_vertices = bounding_box_lines(&self.chunk);
        self.num_line_indices = line_vertices.len() as u32;
        self.line_vertex_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        );
        self.remesh();
    }

    fn remesh(&mut self) {
        let (mesh_vertices, mesh_indices) = crate::render::mesh_chunk(&self.chunk, &self.palette);

        use wgpu::util::DeviceExt;
        self.vertex_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        self.index_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh_indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        self.num_indices = mesh_indices.len() as u32;
    }

    /// Save to the current path without prompting, falling back to `Save As` the
    /// first time.
    fn save_project(&mut self) {
        match self.current_path.clone() {
            Some(path) => self.write_path(&path),
            None => self.save_project_as(),
        }
    }

    /// Save the model via a "save as" dialog as a Wavefront `.obj` mesh (plus a
    /// companion `.mtl`), so export lives here rather than as a separate command.
    /// The path is remembered for subsequent plain `Save`s.
    fn save_project_as(&mut self) {
        let suggested = self
            .current_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or(OBJ_PATH);
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save as")
            .add_filter("Wavefront OBJ", &["obj"])
            .set_file_name(suggested)
            .save_file()
        else {
            return; // user cancelled
        };
        self.write_path(&path);
    }

    /// Write the model to `path` (format chosen by extension) and adopt it as the
    /// current path.
    fn write_path(&mut self, path: &std::path::Path) {
        match crate::io::save(path, &self.chunk, &self.palette) {
            Ok(()) => {
                println!("Saved {}", path.display());
                self.current_path = Some(path.to_path_buf());
            }
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }

    /// Import a Voxely-exported `.obj` via a file picker.
    fn open_file(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Open model")
            .add_filter("Wavefront OBJ", &["obj"])
            .pick_file()
        else {
            return; // user cancelled
        };
        self.load_path(&path);
    }

    /// Load a model from `path` into the editor, replacing the current scene.
    /// Shared by the file picker, the OS "Open With" command-line argument, and
    /// drag-and-drop. The openable format (`.obj`) is also a save
    /// target, so the path is adopted for subsequent plain `Save`s.
    pub fn load_path(&mut self, path: &std::path::Path) {
        match crate::io::open(path) {
            Ok(project) => {
                self.chunk = project.chunk;
                self.palette = project.palette;
                self.history.clear();
                self.sync_to_chunk();
                println!("Opened {}", path.display());
                self.current_path = Some(path.to_path_buf());
            }
            Err(e) => eprintln!("Open failed: {e}"),
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Run the UI first so button actions (undo/redo, recolor) apply to the
        // mesh on this same frame. `egui_ctx` is a cheap clone-able handle, so
        // cloning it lets the closure borrow `self` mutably without aliasing.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            self.build_ui(ctx);
        });
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let paint_jobs =
            egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        let user_buffers = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Scene pass: clears the frame and draws the voxel mesh.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Draw bounding box
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.num_line_indices, 0..1);

            // Draw the hover/rectangle overlay on top of the scene.
            if self.num_overlay_vertices > 0 {
                // First pass: draw parts occluded by geometry (dimmer, visible through walls).
                render_pass.set_pipeline(&self.xray_pipeline);
                render_pass.set_vertex_buffer(0, self.overlay_vertex_buffer.slice(..));
                render_pass.draw(0..self.num_overlay_vertices, 0..1);

                // Second pass: draw visible parts (brighter).
                render_pass.set_pipeline(&self.highlight_pipeline);
                render_pass.set_vertex_buffer(0, self.overlay_vertex_buffer.slice(..));
                render_pass.draw(0..self.num_overlay_vertices, 0..1);
            }
        }

        // UI pass: draws egui on top, preserving the scene (LoadOp::Load).
        {
            let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer
                .render(&mut egui_pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue
            .submit(user_buffers.into_iter().chain(std::iter::once(encoder.finish())));
        output.present();

        Ok(())
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        self.build_menu_bar(ctx);
        egui::SidePanel::left("controls_panel")
            .resizable(false)
            .default_width(210.0)
            .show(ctx, |ui| {
              // Scroll so every control stays reachable even when the window is
              // shorter than the full panel.
              egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Voxely");
                ui.separator();

                ui.label("Tool");
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.tool == Tool::Build, "🔨 Build")
                        .clicked()
                    {
                        self.tool = Tool::Build;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Paint, "🖌 Paint")
                        .clicked()
                    {
                        self.tool = Tool::Paint;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Bucket, "🪣 Bucket")
                        .clicked()
                    {
                        self.tool = Tool::Bucket;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Extrude, "⇗ Extrude")
                        .clicked()
                    {
                        self.tool = Tool::Extrude;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Move, "✥ Move")
                        .clicked()
                    {
                        self.tool = Tool::Move;
                    }
                });

                // Eyedropper is modal: tapping `Q` arms it; the next click samples.
                if ui
                    .selectable_label(self.eyedropper_armed, "💧 Eyedropper (Q)")
                    .clicked()
                {
                    self.eyedropper_armed = !self.eyedropper_armed;
                }
                if self.eyedropper_armed {
                    ui.colored_label(
                        egui::Color32::from_rgb(102, 255, 102),
                        "Click a voxel to sample its color",
                    );
                }

                ui.add_space(6.0);
                ui.label("History");
                ui.horizontal(|ui| {
                    if ui.button("⟲ Undo").clicked() {
                        self.undo();
                    }
                    if ui.button("⟳ Redo").clicked() {
                        self.redo();
                    }
                });

                ui.add_space(6.0);
                ui.label(format!(
                    "Canvas: {}×{}×{}",
                    self.chunk.width, self.chunk.height, self.chunk.depth
                ));

                ui.separator();
                ui.label(format!("Active color: #{}", self.current_color_index));

                // Recolor the selected palette slot; existing voxels of that
                // color update immediately via a remesh. The picker works in
                // sRGB *byte* space (`_srgb`), the same values we store and draw
                // the swatches with — `color_edit_button_rgb` would instead read
                // the floats as linear and show a brighter, mismatched color.
                let idx = self.current_color_index as usize;
                let c = self.palette.colors[idx];
                let mut srgb = [c[0], c[1], c[2]];
                if ui.color_edit_button_srgb(&mut srgb).changed() {
                    self.palette.colors[idx] = [srgb[0], srgb[1], srgb[2], 255];
                    self.remesh();
                }

                ui.add_space(6.0);
                ui.label("Palette (click to pick)");
                ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                let cols = 8;
                for row in 0..8 {
                    ui.horizontal(|ui| {
                        for col in 0..cols {
                            let i = (row * cols + col + 1) as u8; // 1..=64
                            let pc = self.palette.colors[i as usize];
                            let swatch = egui::Color32::from_rgb(pc[0], pc[1], pc[2]);
                            let (rect, resp) = ui
                                .allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
                            ui.painter().rect_filled(rect, 2.0, swatch);
                            if i == self.current_color_index {
                                ui.painter().rect_stroke(
                                    rect,
                                    2.0,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                );
                            }
                            if resp.clicked() {
                                self.current_color_index = i;
                            }
                        }
                    });
                }
              });
            });
    }

    /// The top menu bar: File (open/save), Edit (canvas extents), and Help
    /// (controls reference).
    fn build_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…").clicked() {
                        self.open_file();
                        ui.close_menu();
                    }
                    if ui.button("Save").clicked() {
                        self.save_project();
                        ui.close_menu();
                    }
                    if ui.button("Save As…").clicked() {
                        self.save_project_as();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Edit", |ui| {
                    ui.label("Canvas extents");
                    let max = crate::core::chunk::MAX_CHUNK_SIZE;
                    // Apply when a field is committed (Enter / click-away / end of
                    // a drag), so editing a size takes effect on its own.
                    let mut commit = false;
                    for (i, name) in ["Width", "Height", "Depth"].iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{name}:"));
                            let resp = ui.add(
                                egui::DragValue::new(&mut self.pending_size[i])
                                    .clamp_range(1..=max)
                                    .speed(0.2),
                            );
                            if resp.lost_focus() || resp.drag_stopped() {
                                commit = true;
                            }
                        });
                    }
                    let p = self.pending_size;
                    if commit && p != [self.chunk.width, self.chunk.height, self.chunk.depth] {
                        self.resize_canvas(p[0], p[1], p[2]);
                    }
                });

                ui.menu_button("Help", |ui| {
                    ui.label("Left-click: build / paint (active tool)");
                    ui.label("Shift + Left-click: erase");
                    ui.label("Q, then click: eyedropper (pick color)");
                    ui.label("Right-drag: orbit");
                    ui.label("Middle-drag: pan · Scroll: zoom");
                    ui.label("Ctrl + Left-drag: fill rectangle (Build)");
                    ui.label("Ctrl + Shift + Left-drag: erase rectangle");
                    ui.label("Tab / Shift + Tab: cycle tools");
                    ui.label("Bucket: click fills region · Shift + Left erases it");
                });
            });
        });
    }
}

/// Builds the wireframe edges of the chunk's bounding box as a `LineList`
/// (every pair of consecutive vertices is one segment).
fn bounding_box_lines(chunk: &crate::core::Chunk) -> Vec<Vertex> {
    let (w, h, d) = (chunk.width as f32, chunk.height as f32, chunk.depth as f32);
    let c = [1.0, 1.0, 1.0];
    let n = [0.0, 0.0, 0.0];
    let v = |position: [f32; 3]| Vertex { position, color: c, normal: n };
    // The 8 corners, then the 12 edges as index pairs into them.
    let corners = [
        [0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, 0.0, d], [0.0, 0.0, d],
        [0.0, h, 0.0], [w, h, 0.0], [w, h, d], [0.0, h, d],
    ];
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom square
        (4, 5), (5, 6), (6, 7), (7, 4), // top square
        (0, 4), (1, 5), (2, 6), (3, 7), // pillars
    ];
    let mut out = Vec::with_capacity(EDGES.len() * 2);
    for (a, b) in EDGES {
        out.push(v(corners[a]));
        out.push(v(corners[b]));
    }
    out
}

/// Appends two triangles forming the quad `base + s*du + t*dv` (s,t ∈ [0,1]),
/// nudged `0.02` along `normal` so the translucent overlay sits just in front
/// of the surface and avoids z-fighting.
fn push_overlay_quad(
    verts: &mut Vec<Vertex>,
    base: [f32; 3],
    du: [f32; 3],
    dv: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
) {
    let off = 0.02;
    let b = [
        base[0] + normal[0] * off,
        base[1] + normal[1] * off,
        base[2] + normal[2] * off,
    ];
    let p = |s: f32, t: f32| Vertex {
        position: [
            b[0] + du[0] * s + dv[0] * t,
            b[1] + du[1] * s + dv[1] * t,
            b[2] + du[2] * s + dv[2] * t,
        ],
        color,
        normal: [0.0; 3],
    };
    verts.push(p(0.0, 0.0));
    verts.push(p(1.0, 0.0));
    verts.push(p(1.0, 1.0));
    verts.push(p(0.0, 0.0));
    verts.push(p(1.0, 1.0));
    verts.push(p(0.0, 1.0));
}
