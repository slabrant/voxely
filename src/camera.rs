use glam::{Mat4, Vec3};
use winit::event::*;

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
            is_right_mouse_pressed: false,
            is_middle_mouse_pressed: false,
            mouse_delta: (0.0, 0.0),
            scroll_delta: 0.0,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
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
        // Mouse zoom (scroll): move the eye toward/away from the target.
        if self.scroll_delta.abs() > 0.0 {
            let forward = camera.target - camera.eye;
            let forward_mag = forward.length();
            let forward_norm = forward / forward_mag;
            let zoom_amount = self.scroll_delta * self.speed * 10.0;
            if forward_mag > zoom_amount {
                camera.eye += forward_norm * zoom_amount;
            }
            self.scroll_delta = 0.0;
        }

        // Mouse orbit (right-drag): spherical orbit around the target. Yaw is
        // around world-up and pitch is clamped just short of the poles, so the
        // motion stays stable and can't flip the view upside-down.
        if self.is_right_mouse_pressed && (self.mouse_delta.0 != 0.0 || self.mouse_delta.1 != 0.0) {
            let relative = camera.eye - camera.target;
            let radius = relative.length();
            let mut yaw = relative.z.atan2(relative.x);
            let mut pitch = (relative.y / radius).asin();

            yaw -= self.mouse_delta.0 * self.sensitivity;
            pitch += self.mouse_delta.1 * self.sensitivity;

            let limit = std::f32::consts::FRAC_PI_2 - 0.01;
            pitch = pitch.clamp(-limit, limit);

            camera.eye = camera.target
                + Vec3::new(
                    radius * pitch.cos() * yaw.cos(),
                    radius * pitch.sin(),
                    radius * pitch.cos() * yaw.sin(),
                );
        }

        // Mouse pan (middle-drag): slide both eye and target across the view plane.
        if self.is_middle_mouse_pressed {
            let forward = camera.target - camera.eye;
            let forward_mag = forward.length();
            let forward_norm = forward / forward_mag;
            let right = forward_norm.cross(camera.up);
            let up = right.cross(forward_norm);
            let pan = right * self.mouse_delta.0 * self.sensitivity * forward_mag * 0.5
                + up * self.mouse_delta.1 * self.sensitivity * forward_mag * 0.5;
            camera.eye += pan;
            camera.target += pan;
        }

        self.mouse_delta = (0.0, 0.0);
    }
}
