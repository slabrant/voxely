//! Opening, saving and starting over. Everything on disk is a Wavefront `.obj`
//! plus its companion `.mtl`; see [`crate::io`].

use super::*;

impl State {
    /// Save to the current path without prompting, falling back to `Save As` the
    /// first time.
    pub(super) fn save_project(&mut self) {
        match self.current_path.clone() {
            Some(path) => self.write_path(&path),
            None => self.save_project_as(),
        }
    }

    /// Save the model via a "save as" dialog as a Wavefront `.obj` mesh (plus a
    /// companion `.mtl`), so export lives here rather than as a separate command.
    /// The path is remembered for subsequent plain `Save`s.
    pub(super) fn save_project_as(&mut self) {
        let suggested = self
            .current_path
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or(OBJ_PATH);
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save as")
            .add_filter("Wavefront OBJ", &["obj"])
            .set_file_name(suggested);
        if let Some(dir) = &self.last_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else {
            return; // user cancelled
        };
        self.write_path(&path);
    }

    /// Write the model to `path` (format chosen by extension) and adopt it as the
    /// current path.
    pub(super) fn write_path(&mut self, path: &std::path::Path) {
        match crate::io::save(path, &self.chunk, &self.palette) {
            Ok(()) => {
                println!("Saved {}", path.display());
                self.current_path = Some(path.to_path_buf());
                self.remember_dir(path);
                self.set_status(format!("Saved {}", path.display()), false);
            }
            Err(e) => {
                eprintln!("Save failed: {e}");
                self.set_status(format!("Save failed: {e}"), true);
            }
        }
    }

    /// Import a Voxely-exported `.obj` via a file picker.
    pub(super) fn open_file(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Open model")
            .add_filter("Wavefront OBJ", &["obj"]);
        if let Some(dir) = &self.last_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_file() else {
            return; // user cancelled
        };
        self.load_path(&path);
    }

    /// Load a model from `path` into the editor, replacing the current scene.
    /// Shared by the file picker, the OS "Open With" command-line argument, and
    /// drag-and-drop. The openable format (`.obj`) is also a save
    /// target, so the path is adopted for subsequent plain `Save`s.
    pub fn load_path(&mut self, path: &std::path::Path) {
        match crate::io::open(path) {
            Ok(project) => {
                self.chunk = project.chunk;
                self.palette = project.palette;
                self.history.clear();
                self.sync_to_chunk();
                // Frame the model we just opened. Without this the camera keeps
                // whatever position it had, and opening anything bigger than the
                // default canvas leaves the eye *inside* the model -- every ray
                // then starts in a solid voxel, so clicking appears to do nothing
                // at all.
                self.frame_camera_to_chunk();
                println!("Opened {}", path.display());
                self.current_path = Some(path.to_path_buf());
                self.remember_dir(path);
                self.set_status(format!("Opened {}", path.display()), false);
            }
            Err(e) => {
                eprintln!("Open failed: {e}");
                self.set_status(format!("Open failed: {e}"), true);
            }
        }
    }

    /// Discard the current scene and start a fresh, empty model. Forgets the
    /// current path so the next `Save` prompts for a new one.
    pub(super) fn new_project(&mut self) {
        self.chunk = crate::core::Chunk::new();
        self.palette = Palette::default();
        self.history.clear();
        self.sync_to_chunk();
        self.frame_camera_to_chunk();
        self.current_path = None;
    }

}
