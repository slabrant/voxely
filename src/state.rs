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

const PROJECT_PATH: &str = "project.voxely";
const VOX_PATH: &str = "model.vox";
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
    last_action_time: std::time::Duration,
    is_undo_pressed: bool,
    is_redo_pressed: bool,
    last_grid_coord: Option<[i32; 3]>,
    drag_start_voxels: Vec<[i32; 3]>,
    rect_drag: Option<RectDrag>,
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

        use crate::core::chunk::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH};
        let w = CHUNK_WIDTH as f32;
        let h = CHUNK_HEIGHT as f32;
        let d = CHUNK_DEPTH as f32;
        let line_vertices = [
            // Bottom square
            Vertex { position: [0.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            // Top square
            Vertex { position: [0.0, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            // Pillars
            Vertex { position: [0.0, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, 0.0], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [w, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, 0.0, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
            Vertex { position: [0.0, h, d], color: [1.0, 1.0, 1.0], normal: [0.0, 0.0, 0.0] },
        ];
        let num_line_indices = line_vertices.len() as u32;

        let line_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_vertices),
                usage: wgpu::BufferUsages::VERTEX,
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
            last_action_time: std::time::Duration::ZERO,
            is_undo_pressed: false,
            is_redo_pressed: false,
            last_grid_coord: None,
            drag_start_voxels: Vec::new(),
            rect_drag: None,
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
                if self.is_left_mouse_pressed {
                    if self.rect_drag.is_some() {
                        self.update_rect();
                    } else {
                        self.handle_drag();
                    }
                }
                self.update_overlay();
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left {
                    self.is_left_mouse_pressed = *state == winit::event::ElementState::Pressed;
                    if self.is_left_mouse_pressed {
                        self.drag_start_voxels.clear();
                        if self.modifiers.control_key() {
                            // ctrl(+shift) initiates a plane-locked rectangle fill.
                            self.begin_rect();
                        } else {
                            self.handle_click();
                        }
                    } else {
                        // Releasing the button commits any in-progress rectangle.
                        if self.rect_drag.is_some() {
                            self.commit_rect();
                        }
                        self.last_grid_coord = None;
                        self.drag_start_voxels.clear();
                    }
                    self.update_overlay();
                }
                self.camera_controller.process_events(event)
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
                        KeyCode::KeyS if ctrl && !*repeat => { self.save_project(); return true; }
                        KeyCode::KeyL if ctrl && !*repeat => { self.load_project(); return true; }
                        KeyCode::KeyI if ctrl && !*repeat => { self.import_vox(); return true; }
                        KeyCode::KeyE if ctrl && !*repeat => { self.export_obj(); return true; }
                        KeyCode::KeyZ if ctrl => {
                            if !self.is_undo_pressed {
                                self.undo();
                                self.last_action_time = std::time::Duration::ZERO; // Initial undo immediate
                                self.is_undo_pressed = true;
                            }
                            return true;
                        }
                        KeyCode::KeyY if ctrl => {
                            if !self.is_redo_pressed {
                                self.redo();
                                self.last_action_time = std::time::Duration::ZERO; // Initial redo immediate
                                self.is_redo_pressed = true;
                            }
                            return true;
                        }
                        _ => {}
                    }
                } else {
                    // Released
                    match key {
                        KeyCode::KeyZ => self.is_undo_pressed = false,
                        KeyCode::KeyY => self.is_redo_pressed = false,
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

    fn handle_click(&mut self) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, is_drag_voxel)) = self.raycast(ray_origin, ray_dir) {
            let coord = if self.modifiers.shift_key() {
                // Remove the voxel that was hit.
                pos
            } else if is_drag_voxel {
                // If we hit a voxel from the same drag session, don't add the normal.
                // This prevents stacking (staircases) while still allowing the voxel to block the ray.
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
            let color_index = if self.modifiers.shift_key() { 0 } else { self.current_color_index };
            if self.set_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, false) {
                self.last_grid_coord = Some(coord);
            }
        }
    }

    fn handle_drag(&mut self) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, is_drag_voxel)) = self.raycast(ray_origin, ray_dir) {
            let current_coord = if self.modifiers.shift_key() {
                pos
            } else if is_drag_voxel {
                pos
            } else {
                [
                    pos[0] + normal[0] as i32,
                    pos[1] + normal[1] as i32,
                    pos[2] + normal[2] as i32,
                ]
            };

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
                            // Skip the first coordinate if it's the one we just placed in handle_click
                            // or the last one from handle_drag.
                            if i == 0 && Some(coord) == self.last_grid_coord {
                                cx += x_inc;
                                cy += y_inc;
                                cz += z_inc;
                                continue;
                            }
                            let color_index = if self.modifiers.shift_key() { 0 } else { self.current_color_index };
                            self.set_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, true);
                            cx += x_inc;
                            cy += y_inc;
                            cz += z_inc;
                        }
                    }
                    self.last_grid_coord = Some(current_coord);
                }
            } else {
                let color_index = if self.modifiers.shift_key() { 0 } else { self.current_color_index };
                self.set_voxel(current_coord[0], current_coord[1], current_coord[2], Voxel { color_index }, false);
                self.last_grid_coord = Some(current_coord);
            }
        }
    }

    /// Casts the cursor ray and starts a rectangle drag, locking the fill plane
    /// to the face that was hit. Does nothing if the ray misses the world.
    fn begin_rect(&mut self) {
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
            let remove = self.modifiers.shift_key();
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
        use crate::core::chunk::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH};
        let dims = [CHUNK_WIDTH as i32, CHUNK_HEIGHT as i32, CHUNK_DEPTH as i32];
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
        for u in u0..=u1 {
            for v in v0..=v1 {
                coord[rd.u_axis] = u;
                coord[rd.v_axis] = v;
                self.set_voxel(coord[0], coord[1], coord[2], Voxel { color_index }, true);
            }
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
                let color = if self.modifiers.shift_key() {
                    [1.0, 0.3, 0.3]
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
            use wgpu::util::DeviceExt;
            self.overlay_vertex_buffer = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Overlay Vertex Buffer"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            );
        }
    }

    /// Sets a voxel, recording the change for undo and re-meshing only if
    /// something actually changed. Coordinates outside the chunk are ignored.
    fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel, is_drag: bool) -> bool {
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

        // Record voxels placed during this drag session to exclude them from raycasting
        if self.is_left_mouse_pressed && !voxel.is_empty() {
            self.drag_start_voxels.push([x, y, z]);
        }

        if is_drag {
            self.history.record_continuous(VoxelEdit { x: xu, y: yu, z: zu, old, new: voxel });
        } else {
            self.history.record(VoxelEdit { x: xu, y: yu, z: zu, old, new: voxel });
        }
        self.remesh();
        true
    }

    fn undo(&mut self) {
        if let Some(e) = self.history.undo() {
            self.chunk.set(e.x, e.y, e.z, e.old);
            self.remesh();
            println!("Undo");
        }
    }

    fn redo(&mut self) {
        if let Some(e) = self.history.redo() {
            self.chunk.set(e.x, e.y, e.z, e.new);
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
        // First check for voxel hits
        let mut t = 0.0;
        let max_dist = 100.0;
        let step = 0.1;

        while t < max_dist {
            let p = origin + dir * t;
            let ip = [p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32];
            
            if let Some(voxel) = self.chunk.get(ip[0] as usize, ip[1] as usize, ip[2] as usize) {
                if !voxel.is_empty() {
                    let is_drag_voxel = self.drag_start_voxels.contains(&ip);

                    // Primitive normal detection
                    let mut normal = [0.0; 3];
                    let local_p = p - glam::Vec3::new(ip[0] as f32 + 0.5, ip[1] as f32 + 0.5, ip[2] as f32 + 0.5);
                    let abs_p = local_p.abs();
                    if abs_p.x > abs_p.y && abs_p.x > abs_p.z {
                        normal[0] = local_p.x.signum();
                    } else if abs_p.y > abs_p.z {
                        normal[1] = local_p.y.signum();
                    } else {
                        normal[2] = local_p.z.signum();
                    }
                    return Some((ip, normal, is_drag_voxel));
                }
            }
            t += step;
        }

        // If no voxel was hit, check for intersection with the world's bounding box boundaries.
        // This allows building from the limits when the world is empty.
        use crate::core::chunk::{CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_DEPTH};
        
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;
        let mut hit_normal_near = [0.0; 3];
        let mut hit_normal_far = [0.0; 3];

        let bounds_min = [0.0, 0.0, 0.0];
        let bounds_max = [CHUNK_WIDTH as f32, CHUNK_HEIGHT as f32, CHUNK_DEPTH as f32];
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
            let dim = match i { 0 => CHUNK_WIDTH, 1 => CHUNK_HEIGHT, 2 => CHUNK_DEPTH, _ => unreachable!() } as i32;
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
            let dim = match i { 0 => CHUNK_WIDTH, 1 => CHUNK_HEIGHT, 2 => CHUNK_DEPTH, _ => unreachable!() } as i32;
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

    fn save_project(&self) {
        match crate::io::save_project(PROJECT_PATH, &self.chunk, &self.palette) {
            Ok(()) => println!("Saved {PROJECT_PATH}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }

    fn load_project(&mut self) {
        match crate::io::load_project(PROJECT_PATH) {
            Ok(project) => {
                self.chunk = project.chunk;
                self.palette = project.palette;
                self.history.clear();
                self.remesh();
                println!("Loaded {PROJECT_PATH}");
            }
            Err(e) => eprintln!("Load failed: {e}"),
        }
    }

    fn import_vox(&mut self) {
        match crate::io::import_vox(VOX_PATH) {
            Ok(project) => {
                self.chunk = project.chunk;
                self.palette = project.palette;
                self.history.clear();
                self.remesh();
                println!("Imported {VOX_PATH}");
            }
            Err(e) => eprintln!("Import failed: {e}"),
        }
    }

    fn export_obj(&self) {
        match crate::io::export_obj(OBJ_PATH, &self.chunk, &self.palette) {
            Ok(()) => println!("Exported {OBJ_PATH} (+ .mtl)"),
            Err(e) => eprintln!("Export failed: {e}"),
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        if self.is_undo_pressed && dt - self.last_action_time >= crate::ACTION_REPEAT_DELAY {
            self.undo();
            self.last_action_time = dt;
        }

        if self.is_redo_pressed && dt - self.last_action_time >= crate::ACTION_REPEAT_DELAY {
            self.redo();
            self.last_action_time = dt;
        }

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
        egui::SidePanel::left("controls_panel")
            .resizable(false)
            .default_width(210.0)
            .show(ctx, |ui| {
                ui.heading("Voxely");
                ui.separator();

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
                ui.label("File");
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.save_project();
                    }
                    if ui.button("Load").clicked() {
                        self.load_project();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Import .vox").clicked() {
                        self.import_vox();
                    }
                    if ui.button("Export .obj").clicked() {
                        self.export_obj();
                    }
                });

                ui.separator();
                ui.label(format!("Active color: #{}", self.current_color_index));

                // Recolor the selected palette slot; existing voxels of that
                // color update immediately via a remesh.
                let idx = self.current_color_index as usize;
                let c = self.palette.colors[idx];
                let mut rgb = [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ];
                if ui.color_edit_button_rgb(&mut rgb).changed() {
                    self.palette.colors[idx] = [
                        (rgb[0] * 255.0).round() as u8,
                        (rgb[1] * 255.0).round() as u8,
                        (rgb[2] * 255.0).round() as u8,
                        255,
                    ];
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

                ui.separator();
                ui.label("Left-click: place");
                ui.label("Shift + Left-click: remove");
                ui.label("Right-drag: orbit");
                ui.label("Middle-drag: pan · Scroll: zoom");
                ui.label("Ctrl + Left-drag: fill rectangle");
                ui.label("Ctrl + Shift + Left-drag: erase rectangle");
            });
    }
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
