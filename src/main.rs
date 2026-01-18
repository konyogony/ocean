use crate::state::State;
use anyhow::Result;
use std::sync::Arc;
use winit::keyboard::KeyCode;
use winit::window::CursorGrabMode;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::Window,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DESC: &str = env!("CARGO_PKG_DESCRIPTION");

pub mod camera;
pub mod settings;
pub mod skybox;
pub mod state;
pub mod texture;
pub mod vertex;

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Result<Self> {
        Ok(Self { state: None })
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title(format!("Ocean Simulation, build {VERSION}"))
            .with_inner_size(LogicalSize::new(2560, 1440));
        // Doesnt matter since on hyprland it
        // opens full screen anyway
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        window.set_cursor_visible(false);
        if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
            log::warn!("Could not lock cursor")
        }

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: State) {
        self.state = Some(event)
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        state.handle_window_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {e}");
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape), // Only for escape
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            state.handle_device_event(&event);
        }
    }
}

pub fn run() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn main() -> Result<()> {
    run()
}
