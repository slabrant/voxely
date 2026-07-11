use glam::{Mat4, Vec3};
use winit::event::*;

// ── Camera tuning ───────────────────────────────────────────────────────────
// Everything that controls how the camera "feels" lives here — this is the one
// place to tweak. Larger values = faster motion.
/// Orbit speed: radians of rotation per pixel of right-drag.
const ORBIT_SENSITIVITY: f32 = 0.005;
/// Pan speed: fraction of the distance-to-target moved per pixel of middle-drag.
/// Now angle-independent, so this is set to match the old ~45-degree feel.
const PAN_SENSITIVITY: f32 = 0.0006;
/// Zoom speed: fraction of the distance-to-target closed per unit of scroll
/// delta. Proportional (multiplicative) so a notch feels the same whether the
/// camera is right up against the model or far out, instead of crawling when
/// zoomed way out.
const ZOOM_SENSITIVITY: f32 = 0.1;
/// Closest the eye may sit to the target, so zoom-in can't cross it.
const MIN_ZOOM_DISTANCE: f32 = 0.5;

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

#[derive(Default)]
pub struct CameraController {
    is_right_mouse_pressed: bool,
    is_middle_mouse_pressed: bool,
    mouse_delta: (f32, f32),
    scroll_delta: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
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
        // Mouse zoom (scroll): scale the eye's distance to the target. Each notch
        // moves a fixed fraction of the current distance, so zooming stays
        // responsive at any range (and never crosses the target).
        if self.scroll_delta.abs() > 0.0 {
            let forward = camera.target - camera.eye;
            let forward_mag = forward.length();
            let forward_norm = forward / forward_mag;
            let new_mag = (forward_mag * (1.0 - self.scroll_delta * ZOOM_SENSITIVITY))
                .max(MIN_ZOOM_DISTANCE);
            camera.eye = camera.target - forward_norm * new_mag;
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

            yaw += self.mouse_delta.0 * ORBIT_SENSITIVITY;
            pitch += self.mouse_delta.1 * ORBIT_SENSITIVITY;

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
            // Normalize the basis so pan speed depends only on distance, not on
            // the view angle. (`forward x up` has length sin(angle-to-vertical),
            // which would make panning crawl top-down and race from the side.)
            let right = forward_norm.cross(camera.up).normalize();
            let up = right.cross(forward_norm);
            // "Grab the world": the scene follows the cursor, so the camera
            // moves opposite the horizontal drag (drag left -> camera goes right).
            let pan = right * -self.mouse_delta.0 * PAN_SENSITIVITY * forward_mag
                + up * self.mouse_delta.1 * PAN_SENSITIVITY * forward_mag;
            camera.eye += pan;
            camera.target += pan;
        }

        self.mouse_delta = (0.0, 0.0);
    }
}
