//! The egui interface: the tool/palette side panel, the menu bar, and the
//! transient status banner.

use super::*;
use super::picker::{color_picker_button, select_all_on_keyboard_focus};

impl State {
    pub(super) fn build_ui(&mut self, ctx: &egui::Context) {
        self.build_menu_bar(ctx);
        self.build_status_banner(ctx);
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
                        .selectable_label(self.tool == Tool::Build, "🔨 Build (B)")
                        .clicked()
                    {
                        self.tool = Tool::Build;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Paint, "🖌 Paint (P)")
                        .clicked()
                    {
                        self.tool = Tool::Paint;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Bucket, "🪣 Bucket (F)")
                        .clicked()
                    {
                        self.tool = Tool::Bucket;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Extrude, "⇗ Extrude (E)")
                        .clicked()
                    {
                        self.tool = Tool::Extrude;
                    }
                    if ui
                        .selectable_label(self.tool == Tool::Move, "✥ Move (M)")
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
                // sRGB *byte* space, the same values we store and draw the
                // swatches with — reading them as linear floats would show a
                // brighter, mismatched color.
                let idx = self.current_color_index as usize;
                let c = self.palette.colors[idx];
                let mut srgb = [c[0], c[1], c[2]];
                if color_picker_button(ui, &mut srgb, &mut self.hex_text) {
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
    /// A transient banner at the bottom showing the last open/save outcome.
    /// Green for success, red for errors; fades after a few seconds.
    pub(super) fn build_status_banner(&mut self, ctx: &egui::Context) {
        const TTL: std::time::Duration = std::time::Duration::from_secs(6);
        let Some((msg, is_error, shown_at)) = self.status.clone() else { return };
        let elapsed = shown_at.elapsed();
        if elapsed >= TTL {
            self.status = None;
            return;
        }
        // Keep animating so the banner disappears on time without needing input.
        ctx.request_repaint_after(TTL - elapsed);

        let (bg, fg) = if is_error {
            (egui::Color32::from_rgb(120, 30, 30), egui::Color32::WHITE)
        } else {
            (egui::Color32::from_rgb(30, 90, 40), egui::Color32::WHITE)
        };
        let mut dismiss = false;
        egui::TopBottomPanel::bottom("status_banner")
            .frame(egui::Frame::none().fill(bg).inner_margin(egui::Margin::symmetric(8.0, 4.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(fg, if is_error { "⚠" } else { "✔" });
                    ui.colored_label(fg, &msg);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✖").clicked() {
                            dismiss = true;
                        }
                    });
                });
            });
        if dismiss {
            self.status = None;
        }
    }

    pub(super) fn build_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.new_project();
                        ui.close_menu();
                    }
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
                    // Apply when a field is committed (Enter, Tab away, or a
                    // click elsewhere — a singleline `TextEdit` surrenders focus
                    // on Enter, so `lost_focus` covers all three), so editing a
                    // size takes effect on its own.
                    let mut commit = false;
                    for (i, name) in ["Width", "Height", "Depth"].iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{name}:"));
                            let id = egui::Id::new(("canvas_size", i));
                            // Outside of an active edit the buffer mirrors
                            // `pending_size`, so clamped and rejected values
                            // snap back to what the app actually holds.
                            if !ui.memory(|m| m.has_focus(id)) {
                                self.size_text[i] = self.pending_size[i].to_string();
                            }
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.size_text[i])
                                    .id(id)
                                    .desired_width(60.0),
                            );
                            select_all_on_keyboard_focus(ui, &resp, &self.size_text[i]);
                            if resp.lost_focus() {
                                if let Ok(val) = self.size_text[i].trim().parse::<usize>() {
                                    self.pending_size[i] = val.clamp(1, max);
                                }
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
                    ui.label("B/P/F/E/M: pick tool (Build/Paint/Fill/Extrude/Move)");
                    ui.label("Tab / Shift + Tab: cycle tools");
                    ui.label("Bucket: click fills region · Shift + Left erases it");
                });
            });
        });
    }
}
