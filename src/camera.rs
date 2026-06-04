use glam::{Mat4, Vec3};
use winit::event::*;
use winit::keyboard::KeyCode;

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn update_view_projection(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().to_cols_array_2d();
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_right_mouse_pressed: bool,
    is_middle_mouse_pressed: bool,
    mouse_delta: (f32, f32),
    scroll_delta: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_right_mouse_pressed: false,
            is_middle_mouse_pressed: false,
            mouse_delta: (0.0, 0.0),
            scroll_delta: 0.0,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state,
                    physical_key: winit::keyboard::PhysicalKey::Code(key),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match key {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let is_pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Right => {
                        self.is_right_mouse_pressed = is_pressed;
                        true
                    }
                    MouseButton::Middle => {
                        self.is_middle_mouse_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.scroll_delta += y;
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.scroll_delta += pos.y as f32 * 0.1;
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn handle_mouse_motion(&mut self, dx: f32, dy: f32) {
        if self.is_right_mouse_pressed || self.is_middle_mouse_pressed {
            self.mouse_delta.0 += dx;
            self.mouse_delta.1 += dy;
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();

        // Keyboard zoom
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        // Mouse zoom
        if self.scroll_delta.abs() > 0.0 {
            let zoom_amount = self.scroll_delta * self.speed * 10.0;
            if forward_mag > zoom_amount {
                 camera.eye += forward_norm * zoom_amount;
            }
            self.scroll_delta = 0.0;
        }

        let right = forward_norm.cross(camera.up);
        let up = right.cross(forward_norm);

        // Keyboard orbit
        if self.is_right_pressed {
            let orbit_radius = forward_mag;
            let relative_eye = camera.eye - camera.target;
            let rotated_eye = glam::Quat::from_axis_angle(camera.up, -self.speed * 0.5) * relative_eye;
            camera.eye = camera.target + rotated_eye.normalize() * orbit_radius;
        }
        if self.is_left_pressed {
            let orbit_radius = forward_mag;
            let relative_eye = camera.eye - camera.target;
            let rotated_eye = glam::Quat::from_axis_angle(camera.up, self.speed * 0.5) * relative_eye;
            camera.eye = camera.target + rotated_eye.normalize() * orbit_radius;
        }

        // Mouse Orbit
        if self.is_right_mouse_pressed {
            let orbit_radius = forward_mag;
            let relative_eye = camera.eye - camera.target;
            
            // Horizontal rotation
            let rot_x = glam::Quat::from_axis_angle(camera.up, -self.mouse_delta.0 * self.sensitivity);
            let relative_eye = rot_x * relative_eye;
            
            // Vertical rotation
            let right = (camera.target - (camera.target + relative_eye)).normalize().cross(camera.up).normalize();
            let rot_y = glam::Quat::from_axis_angle(right, -self.mouse_delta.1 * self.sensitivity);
            let relative_eye = rot_y * relative_eye;
            
            camera.eye = camera.target + relative_eye.normalize() * orbit_radius;
        }

        // Mouse Pan
        if self.is_middle_mouse_pressed {
            let pan_x = right * self.mouse_delta.0 * self.sensitivity * forward_mag * 0.5;
            let pan_y = up * self.mouse_delta.1 * self.sensitivity * forward_mag * 0.5;
            camera.eye += pan_x + pan_y;
            camera.target += pan_x + pan_y;
        }

        self.mouse_delta = (0.0, 0.0);
    }
}
