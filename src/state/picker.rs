//! A colour picker built by hand, plus the text-field helpers it shares with
//! the rest of the UI. egui's own picker bakes in a row of R/G/B `DragValue`s
//! that cannot be swapped for a hex field, so the popup is rebuilt here.

/// Selects a text field's whole contents when focus arrives from the keyboard
/// (Tab / Shift+Tab), so the next keystroke replaces the value instead of
/// landing wherever egui happened to leave the caret. Clicking into a field is
/// deliberately excluded — there the click position *is* the intended caret.
pub(super) fn select_all_on_keyboard_focus(ui: &egui::Ui, resp: &egui::Response, text: &str) {
    if !resp.gained_focus() || resp.clicked() {
        return;
    }
    let ctx = ui.ctx();
    let mut state =
        egui::widgets::text_edit::TextEditState::load(ctx, resp.id).unwrap_or_default();
    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
        egui::text::CCursor::new(0),
        egui::text::CCursor::new(text.chars().count()),
    )));
    state.store(ctx, resp.id);
}

/// Parses `RRGGBB`, `#RRGGBB`, `RGB` or `#RGB` (either case) into sRGB bytes.
/// `None` for anything else, including a partially typed value.
pub(super) fn parse_hex_rgb(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let nib = |i: usize| u8::from_str_radix(&s[i..i + 1], 16).ok();
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    match s.len() {
        // Shorthand doubles each digit: `#1AF` is `#11AAFF`.
        3 => Some([nib(0)? * 0x11, nib(1)? * 0x11, nib(2)? * 0x11]),
        6 => Some([byte(0)?, byte(2)?, byte(4)?]),
        _ => None,
    }
}

/// The sRGB bytes a gamma-space HSV value describes.
pub(super) fn srgb_from_hsvag(hsvag: egui::ecolor::HsvaGamma) -> [u8; 3] {
    let [r, g, b, _] = egui::ecolor::Hsva::from(hsvag).to_srgba_unmultiplied();
    [r, g, b]
}

/// A colour swatch that opens a picker popup: a saturation/value square, a hue
/// slider, and a single hexadecimal field.
///
/// This replaces [`egui::Ui::color_edit_button_srgb`], whose popup bakes in a
/// row of R/G/B `DragValue`s. egui builds that row in private helpers
/// (`color_picker_hsvag_2d` / `srgba_edit_ui`), so swapping it for a hex field
/// means hand-rolling the popup rather than configuring the built-in one.
///
/// Works throughout in sRGB *byte* space, matching what the palette stores.
/// Returns `true` when `srgb` changed this frame.
pub(super) fn color_picker_button(ui: &mut egui::Ui, srgb: &mut [u8; 3], hex_text: &mut String) -> bool {
    let popup_id = ui.make_persistent_id("palette_color_popup");
    let size = ui.spacing().interact_size;
    let (rect, mut button_response) = ui.allocate_exact_size(size, egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&button_response);
        let rounding = visuals.rounding.at_most(2.0);
        ui.painter()
            .rect_filled(rect, rounding, egui::Color32::from_rgb(srgb[0], srgb[1], srgb[2]));
        ui.painter().rect_stroke(rect, rounding, visuals.fg_stroke);
    }
    if button_response.clicked() {
        ui.memory_mut(|m| m.toggle_popup(popup_id));
    }

    let mut changed = false;
    if ui.memory(|m| m.is_popup_open(popup_id)) {
        let area_response = egui::Area::new(popup_id)
            .order(egui::Order::Foreground)
            .fixed_pos(button_response.rect.max)
            .constrain(true)
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.spacing_mut().slider_width = 180.0;
                    changed = color_picker_body(ui, srgb, hex_text);
                });
            })
            .response;

        if changed {
            button_response.mark_changed();
        }
        // Same dismissal rules as egui's own picker: Escape, or a click that
        // lands outside both the popup and the swatch that opened it.
        if !button_response.clicked()
            && (ui.input(|i| i.key_pressed(egui::Key::Escape)) || area_response.clicked_elsewhere())
        {
            ui.memory_mut(|m| m.close_popup());
        }
    }
    changed
}

/// The contents of [`color_picker_button`]'s popup. Returns `true` on change.
pub(super) fn color_picker_body(ui: &mut egui::Ui, srgb: &mut [u8; 3], hex_text: &mut String) -> bool {
    use egui::ecolor::{Hsva, HsvaGamma};

    // HSV is carried across frames rather than re-derived from `srgb` every
    // time: the round trip loses hue at zero saturation or value, which would
    // make the hue slider snap to red the moment you drag the square into
    // black or grey. Re-derive only when the stored value stops describing the
    // incoming colour (a different palette slot, or an edit from elsewhere).
    let hsv_id = ui.make_persistent_id("picker_hsv");
    let mut hsvag: HsvaGamma = ui.memory(|m| m.data.get_temp(hsv_id)).unwrap_or_default();
    if srgb_from_hsvag(hsvag) != *srgb {
        hsvag = HsvaGamma::from(Hsva::from_srgb(*srgb));
    }
    hsvag.a = 1.0;

    let before = hsvag;
    let hue = hsvag.h;
    sat_value_square(ui, &mut hsvag.s, &mut hsvag.v, hue);
    hue_slider(ui, &mut hsvag.h);

    let mut changed = false;
    if hsvag != before {
        *srgb = srgb_from_hsvag(hsvag);
        changed = true;
    }

    // Outside of an active edit the buffer mirrors the colour, so a partial or
    // invalid entry reverts to the canonical `#RRGGBB` the moment focus leaves.
    let hex_id = ui.make_persistent_id("picker_hex");
    if !ui.memory(|m| m.has_focus(hex_id)) {
        *hex_text = format!("#{:02X}{:02X}{:02X}", srgb[0], srgb[1], srgb[2]);
    }
    ui.horizontal(|ui| {
        ui.label("Hex");
        let resp = ui.add(
            egui::TextEdit::singleline(hex_text)
                .id(hex_id)
                .desired_width(80.0)
                .char_limit(7),
        );
        select_all_on_keyboard_focus(ui, &resp, hex_text);
        // Applied as you type, so the square and slider track the entry. The
        // buffer is the user's until they leave, so a half-typed value is never
        // overwritten mid-edit.
        if resp.changed()
            && let Some(rgb) = parse_hex_rgb(hex_text)
                && rgb != *srgb {
                    *srgb = rgb;
                    hsvag = HsvaGamma::from(Hsva::from_srgb(rgb));
                    changed = true;
                }
    });

    ui.memory_mut(|m| m.data.insert_temp(hsv_id, hsvag));
    changed
}

/// A saturation (x) by value (y) square for a fixed hue. Mirrors egui's private
/// `color_slider_2d`.
pub(super) fn sat_value_square(ui: &mut egui::Ui, s: &mut f32, v: &mut f32, hue: f32) {
    use egui::{lerp, pos2, remap_clamp};
    const N: u32 = 6;

    let size = egui::Vec2::splat(ui.spacing().slider_width);
    let (rect, response) = ui.allocate_at_least(size, egui::Sense::click_and_drag());
    if let Some(mpos) = response.interact_pointer_pos() {
        *s = remap_clamp(mpos.x, rect.left()..=rect.right(), 0.0..=1.0);
        *v = remap_clamp(mpos.y, rect.bottom()..=rect.top(), 0.0..=1.0);
    }
    if !ui.is_rect_visible(rect) {
        return;
    }

    let color_at = |s: f32, v: f32| {
        egui::Color32::from(egui::ecolor::Hsva::from(egui::ecolor::HsvaGamma {
            h: hue,
            s,
            v,
            a: 1.0,
        }))
    };

    let mut mesh = egui::Mesh::default();
    for xi in 0..=N {
        for yi in 0..=N {
            let (xt, yt) = (xi as f32 / N as f32, yi as f32 / N as f32);
            mesh.colored_vertex(
                pos2(
                    lerp(rect.left()..=rect.right(), xt),
                    lerp(rect.bottom()..=rect.top(), yt),
                ),
                color_at(xt, yt),
            );
            if xi < N && yi < N {
                let tl = yi * (N + 1) + xi;
                mesh.add_triangle(tl, tl + 1, tl + N + 1);
                mesh.add_triangle(tl + 1, tl + N + 1, tl + N + 2);
            }
        }
    }
    ui.painter().add(egui::Shape::mesh(mesh));
    ui.painter()
        .rect_stroke(rect, 0.0, ui.style().interact(&response).bg_stroke);

    let picked = color_at(*s, *v);
    ui.painter().add(egui::epaint::CircleShape {
        center: pos2(
            lerp(rect.left()..=rect.right(), *s),
            lerp(rect.bottom()..=rect.top(), *v),
        ),
        radius: rect.width() / 12.0,
        fill: picked,
        stroke: egui::Stroke::new(2.0, contrast_color(picked)),
    });
}

/// A horizontal hue strip. Mirrors egui's private `color_slider_1d`.
pub(super) fn hue_slider(ui: &mut egui::Ui, hue: &mut f32) {
    use egui::{lerp, pos2, remap_clamp};
    const N: u32 = 6;

    let size = egui::vec2(ui.spacing().slider_width, ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_at_least(size, egui::Sense::click_and_drag());
    if let Some(mpos) = response.interact_pointer_pos() {
        *hue = remap_clamp(mpos.x, rect.left()..=rect.right(), 0.0..=1.0);
    }
    if !ui.is_rect_visible(rect) {
        return;
    }

    // Full saturation and value so the strip reads as a pure hue ramp.
    let color_at =
        |t: f32| egui::Color32::from(egui::ecolor::Hsva::new(t, 1.0, 1.0, 1.0));

    let mut mesh = egui::Mesh::default();
    for i in 0..=N {
        let t = i as f32 / N as f32;
        let x = lerp(rect.left()..=rect.right(), t);
        mesh.colored_vertex(pos2(x, rect.top()), color_at(t));
        mesh.colored_vertex(pos2(x, rect.bottom()), color_at(t));
        if i < N {
            mesh.add_triangle(2 * i, 2 * i + 1, 2 * i + 2);
            mesh.add_triangle(2 * i + 1, 2 * i + 2, 2 * i + 3);
        }
    }
    ui.painter().add(egui::Shape::mesh(mesh));
    ui.painter()
        .rect_stroke(rect, 0.0, ui.style().interact(&response).bg_stroke);

    let x = lerp(rect.left()..=rect.right(), *hue);
    let r = rect.height() / 4.0;
    let picked = color_at(*hue);
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            pos2(x, rect.center().y),
            pos2(x + r, rect.bottom()),
            pos2(x - r, rect.bottom()),
        ],
        picked,
        egui::Stroke::new(2.0, contrast_color(picked)),
    ));
}

/// Black or white, whichever stays legible on top of `color`.
pub(super) fn contrast_color(color: egui::Color32) -> egui::Color32 {
    if egui::ecolor::Rgba::from(color).intensity() < 0.5 {
        egui::Color32::WHITE
    } else {
        egui::Color32::BLACK
    }
}
