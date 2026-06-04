use crate::camera::{Camera, CameraUniform, CameraController};
use glam::Vec4Swizzles;
use std::sync::Arc;
use winit::{
    event::WindowEvent,
    window::Window,
};

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    depth_texture: Texture,
    chunk: crate::core::Chunk,
    cursor_position: winit::dpi::PhysicalPosition<f64>,
    current_color_index: u8,
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
        
        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window, so this should be safe.
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
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
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

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let camera_controller = CameraController::new(0.2, 0.005);

        let mut chunk = crate::core::Chunk::new();
        // Create a basic 16x16 floor
        for x in 0..16 {
            for z in 0..16 {
                let color = if (x + z) % 2 == 0 { 4 } else { 1 }; // Checkerboard
                chunk.set(x, 0, z, crate::core::Voxel { color_index: color });
            }
        }
        // Add a small pillar in the middle
        for y in 1..4 {
            chunk.set(8, y, 8, crate::core::Voxel { color_index: 2 });
        }

        let (mesh_vertices, mesh_indices) = crate::render::mesh_chunk(&chunk);

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

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            depth_texture,
            chunk,
            cursor_position: winit::dpi::PhysicalPosition::new(0.0, 0.0),
            current_color_index: 1,
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
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let dx = (position.x - self.cursor_position.x) as f32;
                let dy = (position.y - self.cursor_position.y) as f32;
                self.camera_controller.handle_mouse_motion(dx, dy);
                self.cursor_position = *position;
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left && *state == winit::event::ElementState::Pressed {
                    self.handle_click();
                }
                self.camera_controller.process_events(event)
            }
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    state: winit::event::ElementState::Pressed,
                    physical_key: winit::keyboard::PhysicalKey::Code(key),
                    ..
                },
                ..
            } => {
                match key {
                    winit::keyboard::KeyCode::Digit1 => { self.current_color_index = 1; true }
                    winit::keyboard::KeyCode::Digit2 => { self.current_color_index = 2; true }
                    winit::keyboard::KeyCode::Digit3 => { self.current_color_index = 3; true }
                    winit::keyboard::KeyCode::Digit4 => { self.current_color_index = 4; true }
                    winit::keyboard::KeyCode::KeyS if self.window.clone().inner_size().width > 0 => { // Just use Ctrl+S if possible
                        self.save_project();
                        true
                    }
                    winit::keyboard::KeyCode::KeyL => {
                        self.load_project();
                        true
                    }
                    winit::keyboard::KeyCode::KeyI => {
                        self.import_vox();
                        true
                    }
                    _ => self.camera_controller.process_events(event),
                }
            }
            WindowEvent::MouseWheel { .. } => {
                self.camera_controller.process_events(event)
            }
            _ => false,
        }
    }

    fn handle_click(&mut self) {
        // 1. Ray from camera through cursor
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        // 2. Raycast against voxels
        if let Some((pos, normal)) = self.raycast(ray_origin, ray_dir) {
            // Right click or modifier could be for removal, but for now let's just place
            // Shift + Click for removal?
            let _is_shift_pressed = self.window.clone().inner_size().width > 0; // Dummy check for now
            // Actually, let's just implement placement for now.
            
            let place_pos = [
                (pos[0] as f32 + normal[0] * 0.5).floor() as i32,
                (pos[1] as f32 + normal[1] * 0.5).floor() as i32,
                (pos[2] as f32 + normal[2] * 0.5).floor() as i32,
            ];

            if self.chunk.set(place_pos[0] as usize, place_pos[1] as usize, place_pos[2] as usize, crate::core::Voxel { color_index: self.current_color_index }) {
                self.remesh();
            }
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

    fn raycast(&self, origin: glam::Vec3, dir: glam::Vec3) -> Option<([i32; 3], [f32; 3])> {
        let mut t = 0.0;
        let max_dist = 100.0;
        let step = 0.1;

        while t < max_dist {
            let p = origin + dir * t;
            let ip = [p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32];
            
            if let Some(voxel) = self.chunk.get(ip[0] as usize, ip[1] as usize, ip[2] as usize) {
                if !voxel.is_empty() {
                    // Primitive normal detection
                    let mut normal = [0.0, 0.0, 0.0];
                    let local_p = p - glam::Vec3::new(ip[0] as f32 + 0.5, ip[1] as f32 + 0.5, ip[2] as f32 + 0.5);
                    let abs_p = local_p.abs();
                    if abs_p.x > abs_p.y && abs_p.x > abs_p.z {
                        normal[0] = local_p.x.signum();
                    } else if abs_p.y > abs_p.z {
                        normal[1] = local_p.y.signum();
                    } else {
                        normal[2] = local_p.z.signum();
                    }
                    return Some((ip, normal));
                }
            }
            t += step;
        }
        None
    }

    fn remesh(&mut self) {
        let (mesh_vertices, mesh_indices) = crate::render::mesh_chunk(&self.chunk);
        
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
        let serialized = bincode::serialize(&self.chunk).unwrap();
        std::fs::write("project.voxely", serialized).unwrap();
        println!("Project saved to project.voxely");
    }

    fn load_project(&mut self) {
        if let Ok(data) = std::fs::read("project.voxely") {
            let chunk: crate::core::Chunk = bincode::deserialize(&data).unwrap();
            self.chunk = chunk;
            self.remesh();
            println!("Project loaded from project.voxely");
        }
    }

    fn import_vox(&mut self) {
        if let Ok(data) = dot_vox::load("model.vox") {
            let model = &data.models[0];
            let mut new_chunk = crate::core::Chunk::new();
            for voxel in &model.voxels {
                new_chunk.set(voxel.x as usize, voxel.z as usize, voxel.y as usize, crate::core::Voxel { color_index: voxel.i });
            }
            self.chunk = new_chunk;
            self.remesh();
            println!("Imported model.vox");
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_projection(&self.camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

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
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
