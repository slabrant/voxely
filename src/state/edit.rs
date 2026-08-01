//! Window-event dispatch and every gesture it drives: freehand strokes,
//! rectangle fills, flood fill, extrude, move, and the voxel writes and
//! history bookkeeping underneath them.

use super::*;

impl State {
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
            && !self.egui_ctx.wants_keyboard_input() {
                self.cycle_tool(self.modifiers.shift_key());
                return true;
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
                                // either armed (tapped `Q`) or via Alt+Left where
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

                // Bare letters and digits are tool/color shortcuts, but they are
                // also ordinary text. Typing `8` into a canvas-extent field must
                // not also jump the palette to slot 8, so they only fire when
                // egui isn't holding the keyboard. Ctrl-chords are unambiguous
                // and stay live either way.
                let typing = self.egui_ctx.wants_keyboard_input();
                if pressed && (ctrl || !typing) {
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
                        // Direct tool hotkeys, so tools are one keypress away
                        // instead of cycling with Tab. Digits are taken by color
                        // selection, so tools use letters.
                        KeyCode::KeyB if !ctrl && !*repeat => { self.tool = Tool::Build; return true; }
                        KeyCode::KeyP if !ctrl && !*repeat => { self.tool = Tool::Paint; return true; }
                        KeyCode::KeyF if !ctrl && !*repeat => { self.tool = Tool::Bucket; return true; }
                        KeyCode::KeyE if !ctrl && !*repeat => { self.tool = Tool::Extrude; return true; }
                        KeyCode::KeyM if !ctrl && !*repeat => { self.tool = Tool::Move; return true; }
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
                        KeyCode::KeyN if ctrl && !*repeat => { self.new_project(); return true; }
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
                let consumed = self.camera_controller.process_events(event);
                // Zooming moves the camera without moving the cursor, so the
                // ray now points somewhere else; without this the highlight
                // stays stuck on the previously hovered face until the mouse
                // is jiggled. The camera has already been updated by
                // `process_events`, so the new ray is the one we want.
                self.update_overlay();
                consumed
            }
            _ => false,
        }
    }

    /// Applies one click. `erase` (Shift held) clears the hit voxel; otherwise
    /// the active tool decides: Build places against the hit face, Paint
    /// recolors the hit voxel in place (never adds *or* removes geometry).
    pub(super) fn handle_click(&mut self, erase: bool) {
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
    pub(super) fn handle_drag(&mut self, erase: bool) {
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
    pub(super) fn begin_rect(&mut self, remove: bool) {
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
    pub(super) fn update_rect(&mut self) {
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
    pub(super) fn project_to_plane(
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
    pub(super) fn commit_rect(&mut self) {
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
    pub(super) fn handle_bucket(&mut self, erase: bool) {
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
        //
        // A cell is recolored as it is *pushed*, not as it is popped, so it can
        // never be queued twice. Deferring that let every solid neighbour push
        // its own copy: a filled 256³ region peaked at ~6x the cell count on the
        // stack, around 2.4 GB.
        let recolored = Voxel { color_index: new_color };
        // Claiming a cell means recoloring it now, which also removes it from
        // `target` and so from any later neighbour test.
        let claim = |s: &mut Self, x: usize, y: usize, z: usize| {
            let old = *s.chunk.get(x, y, z)?;
            if old.color_index != target {
                return None;
            }
            s.chunk.set(x, y, z, recolored);
            s.history.record(VoxelEdit { x: x as u16, y: y as u16, z: z as u16, old, new: recolored });
            Some([x, y, z])
        };

        let mut stack = Vec::new();
        stack.extend(claim(self, sx, sy, sz));
        while let Some([x, y, z]) = stack.pop() {
            let neighbors = [
                (x + 1, y, z, x + 1 < cw),
                (x.wrapping_sub(1), y, z, x > 0),
                (x, y + 1, z, y + 1 < ch),
                (x, y.wrapping_sub(1), z, y > 0),
                (x, y, z + 1, z + 1 < cd),
                (x, y, z.wrapping_sub(1), z > 0),
            ];
            for (nx, ny, nz, in_bounds) in neighbors {
                if in_bounds {
                    stack.extend(claim(self, nx, ny, nz));
                }
            }
        }

        self.remesh();
    }

    pub(super) fn handle_extrude_click(&mut self, erase: bool) {
        let ray_dir = self.screen_to_ray(self.cursor_position);
        let ray_origin = self.camera.eye;

        if let Some((pos, normal, _)) = self.raycast(ray_origin, ray_dir) {
            // A boundary-only hit reports a cell just outside the canvas, whose
            // negative coordinate would wrap to a huge `usize`.
            if pos[0] < 0 || pos[1] < 0 || pos[2] < 0 {
                return;
            }
            if let Some(v) = self.chunk.get(pos[0] as usize, pos[1] as usize, pos[2] as usize)
                && !v.is_empty() {
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
                            
                            if !visited.contains(&next)
                                && let Some(nv) = self.chunk.get(next[0] as usize, next[1] as usize, next[2] as usize)
                                    && nv.color_index == target_color {
                                        // Check if this voxel also has an exposed face in the same direction.
                                        // A face is exposed if the neighbor in the normal direction is empty.
                                        let neighbor_pos = [
                                            next[0] + normal_i[0],
                                            next[1] + normal_i[1],
                                            next[2] + normal_i[2],
                                        ];
                                        let is_exposed = self.chunk.get(
                                            neighbor_pos[0] as usize,
                                            neighbor_pos[1] as usize,
                                            neighbor_pos[2] as usize
                                        ).map(|v| v.is_empty()).unwrap_or(true);
                                        
                                        if is_exposed {
                                            visited.insert(next);
                                            stack.push(next);
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

    pub(super) fn handle_extrude_drag(&mut self, _erase: bool) {
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
    pub(super) fn steps_along_axis(&self, start_center: glam::Vec3, normal: glam::Vec3, fallback: i32) -> i32 {
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

    pub(super) fn commit_extrude(&mut self) {
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
    pub(super) fn handle_move_click(&mut self) {
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
            // 26-connected: voxels meeting only at an edge or a corner are part
            // of the same object. Face adjacency alone splits anything built on
            // a diagonal into pieces that slide independently.
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let next = [curr[0] + dx, curr[1] + dy, curr[2] + dz];
                        if !visited.contains(&next) && self.solid_at(next) {
                            visited.insert(next);
                            stack.push(next);
                        }
                    }
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
    pub(super) fn clamp_move_steps(&self, md: &MoveDrag, steps: i32) -> i32 {
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
    pub(super) fn commit_move(&mut self) {
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

    /// Whether `coord` is in-bounds and holds a non-empty voxel.
    pub(super) fn solid_at(&self, coord: [i32; 3]) -> bool {
        if coord[0] < 0 || coord[1] < 0 || coord[2] < 0 {
            return false;
        }
        self.chunk
            .get(coord[0] as usize, coord[1] as usize, coord[2] as usize)
            .is_some_and(|v| !v.is_empty())
    }

    /// Eyedropper: read the color index of the voxel under the cursor and make
    /// it the active color. Does nothing unless the cursor is over a solid voxel.
    pub(super) fn pick_color(&mut self) {
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
    pub(super) fn write_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel, is_drag: bool) -> bool {
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
            self.drag_start_voxels.insert([x, y, z]);
        } else if self.is_erasing_gesture && voxel.is_empty() {
            // Cells cleared this erase stroke keep blocking the ray (see
            // `raycast`), so the stroke carves the surface without tunnelling.
            self.erased_cells.insert([x, y, z]);
        }

        if is_drag {
            self.history.record_continuous(VoxelEdit { x: xu as u16, y: yu as u16, z: zu as u16, old, new: voxel });
        } else {
            self.history.record(VoxelEdit { x: xu as u16, y: yu as u16, z: zu as u16, old, new: voxel });
        }
        true
    }

    /// Like [`write_voxel`](Self::write_voxel) but remeshes immediately. For
    /// one-off single-voxel edits (a click, the head of a stroke).
    pub(super) fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel, is_drag: bool) -> bool {
        let changed = self.write_voxel(x, y, z, voxel, is_drag);
        if changed {
            self.remesh();
        }
        changed
    }

}
