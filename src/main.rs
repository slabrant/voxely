// Don't spawn a console window on Windows in release builds. Debug builds keep
// it so logs / `println!` stay visible while developing.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    pollster::block_on(voxely::run());
}
