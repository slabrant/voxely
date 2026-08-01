//! Cursor-to-voxel geometry: unprojecting a screen position into a world ray,
//! walking that ray through the grid, and the canvas wireframe.

use super::*;

impl State {
    pub(super) fn screen_to_ray(&self, position: winit::dpi::PhysicalPosition<f64>) -> glam::Vec3 {
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

    pub(super) fn raycast(&self, origin: glam::Vec3, dir: glam::Vec3) -> Option<([i32; 3], [f32; 3], bool)> {
        let dims = [self.chunk.width as i32, self.chunk.height as i32, self.chunk.depth as i32];

        // Clip the ray to the canvas box before traversing it. The walk used to
        // start at the eye and give up after a fixed 100 units, so once the
        // camera was zoomed further out than that the ray ran out before it
        // reached the model and the cursor silently vanished. Bounding the walk
        // by the box instead makes it independent of camera distance — and
        // cheaper, since the step count now scales with the canvas instead of
        // with the zoom.
        //
        // A degenerate direction would leave the box unbounded on every axis
        // and spin the traversal forever, where the old fixed reach merely ran
        // out; reject it up front.
        if !dir.is_finite() || dir.length_squared() < 1e-12 {
            return None;
        }
        let (t_enter, t_exit, normal_enter, normal_exit) = ray_box(origin, dir, dims)?;
        if t_exit < 0.0 {
            return None; // canvas is entirely behind the camera
        }
        // Enter at the box, or start where we are if the eye is already inside
        // it. The nudge keeps the first `floor` off an exact face plane.
        let t_start = t_enter.max(0.0) + 1e-4;
        let max_t = (t_exit - t_start).max(0.0);

        // Amanatides–Woo voxel traversal. Unlike fixed-step sampling, this lands
        // on every cell the ray crosses in order and tracks exactly which face
        // we entered through, so the placement normal is always correct (no more
        // voxels embedded in the surface they were placed against).
        let o = (origin + dir * t_start).to_array();
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

        // Normal of the face we entered the current cell through: the box face
        // we came in by, or zero when the eye was already inside the box and we
        // are still in its own cell.
        let mut normal = if t_enter > 0.0 { normal_enter } else { [0.0f32; 3] };
        let mut t = 0.0;
        while t <= max_t {
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

        // No voxel along the way, so fall back to the canvas boundary itself —
        // that is what lets you build from the limits while the world is empty.
        // The far face is used so the first voxel lands on the inside of the
        // wall being looked at rather than on the pane nearest the camera.
        let (t_hit, hit_normal) = (t_exit, normal_exit);

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

        Some((coord, inward_normal, false))
    }

}

/// Slab test of a ray against the canvas's axis-aligned box, which spans
/// `0..dims` on each axis.
///
/// Returns `(t_enter, t_exit, normal_enter, normal_exit)`: the distances along
/// `dir` to the near and far faces, and each face's outward normal. Both `t`s
/// may be negative when the box is behind the ray. `None` if the ray misses the
/// box altogether.
pub(super) fn ray_box(
    origin: glam::Vec3,
    dir: glam::Vec3,
    dims: [i32; 3],
) -> Option<(f32, f32, [f32; 3], [f32; 3])> {
    let o = origin.to_array();
    let d = dir.to_array();
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    let mut normal_enter = [0.0f32; 3];
    let mut normal_exit = [0.0f32; 3];

    for i in 0..3 {
        let (lo, hi) = (0.0, dims[i] as f32);
        if d[i].abs() <= 1e-6 {
            // Parallel to this pair of planes: either inside the slab for the
            // whole ray, or never.
            if o[i] < lo || o[i] > hi {
                return None;
            }
            continue;
        }
        let t1 = (lo - o[i]) / d[i];
        let t2 = (hi - o[i]) / d[i];
        let (near, far, n_near, n_far) = if t1 < t2 {
            (t1, t2, -1.0, 1.0)
        } else {
            (t2, t1, 1.0, -1.0)
        };
        if near > t_enter {
            t_enter = near;
            normal_enter = [0.0; 3];
            normal_enter[i] = n_near;
        }
        if far < t_exit {
            t_exit = far;
            normal_exit = [0.0; 3];
            normal_exit[i] = n_far;
        }
    }

    // The slabs overlap only if the last entry precedes the first exit.
    (t_enter <= t_exit).then_some((t_enter, t_exit, normal_enter, normal_exit))
}

/// Builds the wireframe edges of the chunk's bounding box as a `LineList`
/// (every pair of consecutive vertices is one segment).
pub(super) fn bounding_box_lines(chunk: &crate::core::Chunk) -> Vec<Vertex> {
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

#[cfg(test)]
mod tests {
    use super::ray_box;
    use glam::Vec3;

    const DIMS: [i32; 3] = [16, 16, 16];

    /// The reach of a cursor ray must not depend on how far the camera has been
    /// zoomed out: the box is found at 500 units away just as it is at 5.
    #[test]
    pub(super) fn box_is_found_from_any_distance() {
        for eye_z in [20.0, 100.0, 500.0, 10_000.0] {
            let (t_enter, t_exit, n_enter, n_exit) =
                ray_box(Vec3::new(8.0, 8.0, eye_z), Vec3::new(0.0, 0.0, -1.0), DIMS)
                    .expect("head-on ray must hit the box");
            assert_eq!(t_enter, eye_z - 16.0);
            assert_eq!(t_exit, eye_z);
            // Entered through the +Z face, leaves through the -Z one.
            assert_eq!(n_enter, [0.0, 0.0, 1.0]);
            assert_eq!(n_exit, [0.0, 0.0, -1.0]);
        }
    }

    #[test]
    pub(super) fn ray_parallel_to_and_outside_a_slab_misses() {
        assert!(ray_box(Vec3::new(100.0, 8.0, 500.0), Vec3::new(0.0, 0.0, -1.0), DIMS).is_none());
    }

    /// Slabs that never overlap: the ray passes the box by, so the last entry
    /// comes after the first exit.
    #[test]
    pub(super) fn ray_missing_diagonally_is_rejected() {
        assert!(ray_box(Vec3::new(-10.0, -1.0, 8.0), Vec3::new(1.0, 0.01, 0.0), DIMS).is_none());
    }

    /// An eye inside the box gives a negative entry distance, which `raycast`
    /// clamps to zero so traversal starts at the eye's own cell.
    #[test]
    pub(super) fn eye_inside_the_box_enters_behind_itself() {
        let (t_enter, t_exit, _, n_exit) =
            ray_box(Vec3::new(8.0, 8.0, 8.0), Vec3::new(0.0, 0.0, 1.0), DIMS).unwrap();
        assert_eq!(t_enter, -8.0);
        assert_eq!(t_exit, 8.0);
        assert_eq!(n_exit, [0.0, 0.0, 1.0]);
    }

    /// Looking directly away from the canvas: both distances are negative, and
    /// `raycast` turns that into "no hit".
    #[test]
    pub(super) fn box_behind_the_camera_yields_negative_distances() {
        let (_, t_exit, _, _) =
            ray_box(Vec3::new(8.0, 8.0, 500.0), Vec3::new(0.0, 0.0, 1.0), DIMS).unwrap();
        assert!(t_exit < 0.0, "t_exit = {t_exit}");
    }
}
