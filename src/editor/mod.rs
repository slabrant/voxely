use crate::core::Voxel;

/// A single reversible change to one voxel.
#[derive(Clone, Copy)]
pub struct VoxelEdit {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub old: Voxel,
    pub new: Voxel,
}

/// Undo/redo stack for voxel edits.
///
/// Recording a new edit clears the redo stack, matching the usual editor
/// behavior where doing something new discards the "future" you undid past.
#[derive(Default)]
pub struct EditHistory {
    undo: Vec<VoxelEdit>,
    redo: Vec<VoxelEdit>,
}

impl EditHistory {
    pub fn record(&mut self, edit: VoxelEdit) {
        self.undo.push(edit);
        self.redo.clear();
    }

    /// Records an edit that is part of a continuous stroke. 
    /// If the last edit was at the same coordinates, it updates it instead of pushing a new one.
    /// This prevents a single drag stroke from filling the undo buffer with many small changes
    /// to the same voxel.
    pub fn record_continuous(&mut self, edit: VoxelEdit) {
        if let Some(last) = self.undo.last_mut() {
            if last.x == edit.x && last.y == edit.y && last.z == edit.z {
                last.new = edit.new;
                return;
            }
        }
        self.record(edit);
    }

    /// Pops the most recent edit. The caller should re-apply `old` at the
    /// edit's position to revert it.
    pub fn undo(&mut self) -> Option<VoxelEdit> {
        let edit = self.undo.pop()?;
        self.redo.push(edit);
        Some(edit)
    }

    /// Pops the most recently undone edit. The caller should re-apply `new`.
    pub fn redo(&mut self) -> Option<VoxelEdit> {
        let edit = self.redo.pop()?;
        self.undo.push(edit);
        Some(edit)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old: u8, new: u8) -> VoxelEdit {
        VoxelEdit {
            x: 1,
            y: 2,
            z: 3,
            old: Voxel { color_index: old },
            new: Voxel { color_index: new },
        }
    }

    #[test]
    fn undo_then_redo_round_trips() {
        let mut h = EditHistory::default();
        h.record(edit(0, 5));
        let u = h.undo().expect("an edit to undo");
        assert_eq!(u.old.color_index, 0, "undo should restore the old value");
        let r = h.redo().expect("an edit to redo");
        assert_eq!(r.new.color_index, 5, "redo should reapply the new value");
        assert!(h.undo().is_some(), "edit is back on the undo stack after redo");
    }

    #[test]
    fn recording_clears_redo() {
        let mut h = EditHistory::default();
        h.record(edit(0, 1));
        h.undo();
        h.record(edit(0, 2));
        assert!(h.redo().is_none(), "a new edit must discard the redo stack");
    }
}
