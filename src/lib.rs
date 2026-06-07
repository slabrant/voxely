use std::sync::Arc;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::WindowBuilder,
};

mod state;
mod camera;
pub mod core;
pub mod editor;
pub mod io;
pub mod render;

pub const ACTION_REPEAT_DELAY: std::time::Duration = std::time::Duration::from_millis(150);

    pub async fn run() {
    env_logger::init();
    let start_time = std::time::Instant::now();

    println!("--- Voxely Quick Start ---");
    println!("Orbit: Right-Click + Drag  |  Pan: Middle-Click + Drag  |  Zoom: Scroll");
    println!("Build/Paint: Left-Click    |  Erase: Shift + Right-Click");
    println!("Eyedropper: Shift + Left-Click  |  Cycle Tool: B");
    println!("Change Color: 1-9");
    println!("Undo: Ctrl+Z  |  Redo: Ctrl+Y");
    println!("Save: Ctrl+S  |  Save As: Ctrl+Shift+S  |  Open: Ctrl+O");
    println!("Export .obj: Ctrl+E");
    println!("--------------------------");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Voxely")
            .build(&event_loop)
            .unwrap(),
    );

    let mut state = state::State::new(Arc::clone(&window)).await;
    let window_id = window.id();

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id: id,
            } if id == window_id => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update(start_time.elapsed());
                            match state.render() {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                // The next frame should resolve all other errors (Outdated, Timeout)
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
