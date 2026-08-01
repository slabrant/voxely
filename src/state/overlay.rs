//! The translucent overlay drawn over the scene: the hover-face highlight and
//! the live preview of an in-progress rectangle, extrude or move.

use super::*;

impl State {
    /// Rebuilds the overlay vertex buffer for the current frame: the rectangle
    /// preview while dragging, otherwise the hover-face highlight under the
    /// cursor. Empty when the cursor isn't over the world.
    pub(super) fn update_overlay(&mut self) {
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

}

/// Appends two triangles forming the quad `base + s*du + t*dv` (s,t ∈ [0,1]),
/// nudged `0.02` along `normal` so the translucent overlay sits just in front
/// of the surface and avoids z-fighting.
pub(super) fn push_overlay_quad(
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
