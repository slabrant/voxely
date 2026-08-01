use crate::core::Voxel;
use std::rc::Rc;

/// How many gestures the undo stack keeps. A rectangle fill can be tens of
/// thousands of edits, so an unbounded stack grows without limit across a long
/// session; the oldest group is dropped once this is exceeded.
const MAX_UNDO_DEPTH: usize = 256;

/// One gesture's worth of edits. Shared rather than copied, so shuttling a
/// group between the undo and redo stacks is a refcount bump instead of a
/// multi-megabyte memcpy.
pub type EditGroup = Rc<Vec<VoxelEdit>>;

/// A single reversible change to one voxel.
#[derive(Clone, Copy)]
pub struct VoxelEdit {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub old: Voxel,
    pub new: Voxel,
}

/// Undo/redo stacks of voxel edits, grouped per user gesture.
///
/// Each entry on a stack is a *group*: all the voxels touched by one action
/// (a single click, a drag stroke, a rectangle fill, a paint-bucket flood)
/// undo and redo as a unit. A gesture is bracketed by [`begin_group`] /
/// [`end_group`]; edits recorded in between accumulate into the open group.
///
/// Recording a new group clears the redo stack, matching the usual editor
/// behavior where doing something new discards the "future" you undid past.
/// The stack is capped at [`MAX_UNDO_DEPTH`] gestures.
///
/// [`begin_group`]: EditHistory::begin_group
/// [`end_group`]: EditHistory::end_group
#[derive(Default)]
pub struct EditHistory {
    undo: Vec<EditGroup>,
    redo: Vec<EditGroup>,
    current: Option<Vec<VoxelEdit>>,
}

impl EditHistory {
    /// Opens a new group. Edits recorded until [`end_group`](Self::end_group)
    /// accumulate into it and undo/redo together.
    pub fn begin_group(&mut self) {
        self.current = Some(Vec::new());
    }

    /// Closes the open group, committing it to the undo stack if it touched
    /// anything. An empty group (a gesture that changed nothing) is dropped.
    pub fn end_group(&mut self) {
        if let Some(group) = self.current.take()
            && !group.is_empty() {
                self.push_undo(Rc::new(group));
            }
    }

    /// Commits a group and drops the oldest once the stack is over depth.
    fn push_undo(&mut self, group: EditGroup) {
        self.undo.push(group);
        if self.undo.len() > MAX_UNDO_DEPTH {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Records an edit into the open group. If no group is open, the edit is
    /// committed as a group of its own.
    pub fn record(&mut self, edit: VoxelEdit) {
        match &mut self.current {
            Some(group) => group.push(edit),
            None => self.push_undo(Rc::new(vec![edit])),
        }
    }

    /// Records an edit that is part of a continuous stroke. If the most recent
    /// edit in the open group is at the same coordinates, it updates that edit
    /// in place instead of appending a new one, so a drag that lingers on one
    /// voxel doesn't bloat the group.
    pub fn record_continuous(&mut self, edit: VoxelEdit) {
        if let Some(group) = &mut self.current
            && let Some(last) = group.last_mut()
                && last.x == edit.x && last.y == edit.y && last.z == edit.z {
                    last.new = edit.new;
                    return;
                }
        self.record(edit);
    }

    /// Pops the most recent group. The caller should re-apply each edit's `old`
    /// value (in reverse order) to revert it.
    pub fn undo(&mut self) -> Option<EditGroup> {
        let group = self.undo.pop()?;
        self.redo.push(Rc::clone(&group));
        Some(group)
    }

    /// Pops the most recently undone group. The caller should re-apply each
    /// edit's `new` value (in forward order).
    pub fn redo(&mut self) -> Option<EditGroup> {
        let group = self.redo.pop()?;
        self.undo.push(Rc::clone(&group));
        Some(group)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.current = None;
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
        let u = h.undo().expect("a group to undo");
        assert_eq!(u[0].old.color_index, 0, "undo should restore the old value");
        let r = h.redo().expect("a group to redo");
        assert_eq!(r[0].new.color_index, 5, "redo should reapply the new value");
        assert!(h.undo().is_some(), "group is back on the undo stack after redo");
    }

    #[test]
    fn recording_clears_redo() {
        let mut h = EditHistory::default();
        h.record(edit(0, 1));
        h.undo();
        h.record(edit(0, 2));
        assert!(h.redo().is_none(), "a new edit must discard the redo stack");
    }

    #[test]
    fn a_group_undoes_as_a_unit() {
        let mut h = EditHistory::default();
        h.begin_group();
        h.record(edit(0, 1));
        h.record(edit(0, 2));
        h.record(edit(0, 3));
        h.end_group();
        let g = h.undo().expect("one group covering all three edits");
        assert_eq!(g.len(), 3, "the whole gesture undoes at once");
        assert!(h.undo().is_none(), "there was only a single group");
    }

    /// The undo stack is capped, so a long session can't grow it without bound.
    /// The oldest gestures fall off; the most recent MAX_UNDO_DEPTH survive.
    #[test]
    fn undo_stack_is_capped_at_max_depth() {
        let mut h = EditHistory::default();
        for i in 0..MAX_UNDO_DEPTH + 50 {
            h.record(edit(0, (i % 200) as u8 + 1));
        }
        let mut popped = 0;
        while h.undo().is_some() {
            popped += 1;
        }
        assert_eq!(popped, MAX_UNDO_DEPTH, "stack should hold exactly the cap");
    }

    /// Moving a group between the stacks shares it rather than copying it.
    #[test]
    fn undo_shares_the_group_instead_of_copying_it() {
        let mut h = EditHistory::default();
        h.begin_group();
        h.record(edit(0, 1));
        h.end_group();
        let g = h.undo().expect("a group to undo");
        // The group is now on the redo stack too, so it has more than one owner.
        assert_eq!(Rc::strong_count(&g), 2, "undo must not deep-copy the group");
    }

    #[test]
    fn empty_group_is_dropped() {
        let mut h = EditHistory::default();
        h.begin_group();
        h.end_group();
        assert!(h.undo().is_none(), "a gesture that changed nothing leaves no history");
    }
}
