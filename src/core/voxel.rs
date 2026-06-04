#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Voxel {
    pub color_index: u8,
}

impl Voxel {
    pub fn is_empty(&self) -> bool {
        self.color_index == 0
    }
}
