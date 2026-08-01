//! GPU-side setup and the per-frame draw: adapter and pipeline creation, the
//! vertex format, the depth target, and the growable mesh buffers.

use super::*;
use super::geometry::bounding_box_lines;

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
            .copied().find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            // Vsync. `present_modes[0]` is whatever the adapter happens to list
            // first, which on some backends is `Immediate` — combined with the
            // unconditional redraw in `lib.rs` that pins the GPU at 100%.
            present_mode: if surface_caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
                wgpu::PresentMode::Fifo
            } else {
                surface_caps.present_modes[0]
            },
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
            zfar: 2000.0,
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

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));

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

        // Mesh buffers start empty and grow on first use; a fresh canvas has no
        // geometry, and `num_indices == 0` means nothing is drawn from them.
        let vertex_buffer = new_growable_buffer(&device, "Vertex Buffer", wgpu::BufferUsages::VERTEX);
        let index_buffer = new_growable_buffer(&device, "Index Buffer", wgpu::BufferUsages::INDEX);
        let num_indices = 0;

        let line_vertices = bounding_box_lines(&chunk);
        let num_line_vertices = line_vertices.len() as u32;

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
            num_line_vertices,
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
            drag_start_voxels: std::collections::HashSet::new(),
            rect_drag: None,
            extrude_steps: None,
            tool: Tool::Build,
            pending_action: None,
            flooding: false,
            mesh_dirty: true,
            pending_size: [chunk_w, chunk_h, chunk_d],
            size_text: [chunk_w.to_string(), chunk_h.to_string(), chunk_d.to_string()],
            hex_text: String::new(),
            current_path: None,
            status: None,
            last_dir: None,
        }
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
        // After the UI, so a button that recolors or undoes lands on this frame,
        // and before the passes that draw the mesh.
        self.flush_mesh();

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
            render_pass.draw(0..self.num_line_vertices, 0..1);

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

}

/// An empty buffer that [`upload_growable`] will size on first write.
pub(super) fn new_growable_buffer(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 0,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Writes `data` into `buffer`, reallocating only when it no longer fits.
///
/// The mesh buffers used to be recreated from scratch on every edit, which at a
/// 256³ canvas meant discarding and re-uploading ~42 MB per mouse-move. Capacity
/// doubles on growth so a model being built up doesn't reallocate each time.
pub(super) fn upload_growable(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    data: &[u8],
    usage: wgpu::BufferUsages,
    label: &str,
) {
    if data.is_empty() {
        return; // Nothing to draw; the caller's count is already 0.
    }
    if data.len() as u64 > buffer.size() {
        *buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len() as u64 * 2).next_power_of_two(),
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    queue.write_buffer(buffer, 0, data);
}
