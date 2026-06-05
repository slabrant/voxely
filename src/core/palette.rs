use serde::{Deserialize, Serialize};

/// Number of color slots. Index 0 is reserved for "empty" (no voxel).
pub const PALETTE_SIZE: usize = 256;

/// A fixed-size color palette. Voxels store a `color_index` into this table.
#[derive(Clone, Serialize, Deserialize)]
pub struct Palette {
    /// sRGB colors, one per index. Stored as `[r, g, b, a]` bytes.
    pub colors: Vec<[u8; 4]>,
}

impl Default for Palette {
    fn default() -> Self {
        let mut colors = vec![[0, 0, 0, 0]; PALETTE_SIZE];
        // Familiar starter colors on number keys 1..=9.
        colors[1] = [204, 26, 26, 255]; // red
        colors[2] = [26, 204, 26, 255]; // green
        colors[3] = [51, 102, 230, 255]; // blue
        colors[4] = [220, 200, 40, 255]; // yellow
        colors[5] = [230, 126, 34, 255]; // orange
        colors[6] = [142, 68, 200, 255]; // purple
        colors[7] = [26, 200, 200, 255]; // cyan
        colors[8] = [240, 240, 240, 255]; // white
        colors[9] = [70, 70, 70, 255]; // dark gray
        // Fill the rest with a smooth ramp so imports/large palettes aren't blank.
        for (i, slot) in colors.iter_mut().enumerate().skip(10) {
            let t = i as f32 / PALETTE_SIZE as f32;
            *slot = [
                (t * 255.0) as u8,
                (((t * std::f32::consts::TAU).sin() * 0.5 + 0.5) * 255.0) as u8,
                ((1.0 - t) * 255.0) as u8,
                255,
            ];
        }
        Self { colors }
    }
}

impl Palette {
    /// Returns the color at `index` converted to linear RGB for shading on an
    /// sRGB surface. Out-of-range indices fall back to white.
    pub fn linear_rgb(&self, index: u8) -> [f32; 3] {
        let c = self
            .colors
            .get(index as usize)
            .copied()
            .unwrap_or([255, 255, 255, 255]);
        [
            srgb_to_linear(c[0]),
            srgb_to_linear(c[1]),
            srgb_to_linear(c[2]),
        ]
    }
}

fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}
