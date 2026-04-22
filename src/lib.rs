extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate image;
extern crate libc;
#[macro_use]
extern crate log;
extern crate mint;

pub mod config;
mod fxaa;
mod mesh;

use cgmath::EuclideanSpace;
use config::{AAMethod, Config};
use glium::backend::Facade;
use glium::{CapabilitiesSource, Surface};
use image::{ImageEncoder, ImageFormat};
use libc::c_char;
use mesh::Mesh;
use std::error::Error;
use std::ffi::CStr;
use std::{cell::RefCell, io, slice, thread};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;

#[cfg(target_os = "linux")]
use std::env;

const CAM_FOV_DEG: f32 = 30.0;
const CAM_POSITION: cgmath::Point3<f32> = cgmath::Point3 {
    x: 2.0,
    y: -4.0,
    z: 2.0,
};

fn print_matrix(m: [[f32; 4]; 4]) {
    for row in &m {
        debug!("{:.3}\t{:.3}\t{:.3}\t{:.3}", row[0], row[1], row[2], row[3]);
    }
    debug!("");
}

fn print_context_info(display: &impl Facade) {
    let ctx = display.get_context();
    info!("GL Version: {:?}", ctx.get_opengl_version());
    info!("GL Version: {}", ctx.get_opengl_version_string());
    info!("GLSL Version: {:?}", ctx.get_supported_glsl_version());
    info!("Vendor: {}", ctx.get_opengl_vendor_string());
    info!("Renderer {}", ctx.get_opengl_renderer_string());
    info!("Free GPU Mem: {:?}", ctx.get_free_video_memory());
    info!("Depth Bits: {:?}\n", ctx.get_capabilities().depth_bits);
}

thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

fn create_event_loop_once() -> EventLoop<()> {
    #[cfg(target_os = "linux")]
    {
        use winit::platform::x11::EventLoopBuilderExtX11;
        if let Ok(el) = EventLoop::builder().with_any_thread(true).build() {
            return el;
        }
    }
    #[cfg(target_os = "linux")]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        if let Ok(el) = EventLoop::builder().with_any_thread(true).build() {
            return el;
        }
    }
    EventLoop::builder()
        .build()
        .expect("Failed to create event loop")
}

fn with_event_loop<F, R>(f: F) -> R
where
    F: FnOnce(&EventLoop<()>) -> R,
{
    EVENT_LOOP.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(create_event_loop_once());
        }
        f(opt.as_ref().unwrap())
    })
}

fn render_pipeline<F>(
    display: &F,
    config: &Config,
    mesh: &Mesh,
    framebuffer: &mut glium::framebuffer::SimpleFrameBuffer,
    texture: &glium::Texture2d,
) -> image::DynamicImage
where
    F: Facade,
{
    let params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
        ..Default::default()
    };

    let vertex_shader_src = include_str!("shaders/model.vert");
    let pixel_shader_src = include_str!("shaders/model.frag");

    let program = glium::Program::from_source(display, vertex_shader_src, pixel_shader_src, None);
    let program = match program {
        Ok(p) => p,
        Err(glium::CompilationError(err, _)) => {
            error!("{}", err);
            panic!("Compiling shaders");
        }
        Err(err) => panic!("{}", err),
    };

    let vertex_buf = glium::VertexBuffer::new(display, &mesh.vertices).unwrap();
    let normal_buf = glium::VertexBuffer::new(display, &mesh.normals).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let transform_matrix = mesh.scale_and_center();
    let view_matrix = cgmath::Matrix4::look_at_rh(
        CAM_POSITION,
        cgmath::Point3::origin(),
        cgmath::Vector3::unit_z(),
    );

    debug!("View:");
    print_matrix(view_matrix.into());

    let perspective_matrix = cgmath::perspective(
        cgmath::Deg(CAM_FOV_DEG),
        config.width as f32 / config.height as f32,
        0.1,
        1024.0,
    );

    debug!("Perspective:");
    print_matrix(perspective_matrix.into());

    let light_dir = [-1.1, 0.4, 1.0f32];

    let uniforms = uniform! {
        modelview: Into::<[[f32; 4]; 4]>::into(view_matrix * transform_matrix),
        perspective: Into::<[[f32; 4]; 4]>::into(perspective_matrix),
        u_light: light_dir,
        ambient_color: config.material.ambient,
        diffuse_color: config.material.diffuse,
        specular_color: config.material.specular,
    };

    let fxaa = fxaa::FxaaSystem::new(display);
    let fxaa_enable = matches!(config.aamethod, AAMethod::FXAA);

    fxaa::draw(&fxaa, framebuffer, fxaa_enable, |target| {
        target.clear_color_and_depth(config.background, 1.0);
        target
            .draw(
                (&vertex_buf, &normal_buf),
                indices,
                &program,
                &uniforms,
                &params,
            )
            .unwrap();
    });

    let pixels: glium::texture::RawImage2d<u8> = texture.read();
    let img = image::ImageBuffer::from_raw(config.width, config.height, pixels.data.into_owned())
        .unwrap();
    image::DynamicImage::ImageRgba8(img).flipv()
}

pub fn render_to_window(config: Config) -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::load(&config.model_filename, config.recalc_normals)?;

    let event_loop = EVENT_LOOP.with(|cell| {
        cell.borrow_mut()
            .take()
            .unwrap_or_else(create_event_loop_once)
    });

    let window_dim = PhysicalSize::new(config.width, config.height);

    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
        .set_window_builder(
            winit::window::WindowAttributes::default()
                .with_title("stl-thumb")
                .with_inner_size(window_dim)
                .with_min_inner_size(window_dim)
                .with_max_inner_size(window_dim)
                .with_visible(config.visible),
        )
        .build(&event_loop);

    print_context_info(&display);

    let texture = glium::Texture2d::empty(&display, config.width, config.height).unwrap();
    let depthtexture =
        glium::texture::DepthTexture2d::empty(&display, config.width, config.height).unwrap();

    {
        let mut framebuffer = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(
            &display,
            &texture,
            &depthtexture,
        )
        .unwrap();
        render_pipeline(&display, &config, &mesh, &mut framebuffer, &texture);
    }

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => elwt.exit(),
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            let framebuffer = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(
                &display,
                &texture,
                &depthtexture,
            )
            .unwrap();
            let target = display.draw();
            target.blit_from_simple_framebuffer(
                &framebuffer,
                &glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: config.width,
                    height: config.height,
                },
                &glium::BlitTarget {
                    left: 0,
                    bottom: 0,
                    width: config.width as i32,
                    height: config.height as i32,
                },
                glium::uniforms::MagnifySamplerFilter::Nearest,
            );
            target.finish().unwrap();
        }
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => (),
    })?;

    Ok(())
}

pub fn render_to_image(config: &Config) -> Result<image::DynamicImage, Box<dyn Error>> {
    let mesh = Mesh::load(&config.model_filename, config.recalc_normals)?;

    with_event_loop(
        |event_loop| -> Result<image::DynamicImage, Box<dyn Error>> {
            let window_dim = PhysicalSize::new(config.width, config.height);

            let (_window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
                .set_window_builder(
                    winit::window::WindowAttributes::default()
                        .with_title("stl-thumb")
                        .with_inner_size(window_dim)
                        .with_visible(false),
                )
                .build(event_loop);

            print_context_info(&display);

            let texture = glium::Texture2d::empty(&display, config.width, config.height).unwrap();
            let depthtexture =
                glium::texture::DepthTexture2d::empty(&display, config.width, config.height)
                    .unwrap();
            let mut framebuffer = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(
                &display,
                &texture,
                &depthtexture,
            )
            .unwrap();

            Ok(render_pipeline(
                &display,
                config,
                &mesh,
                &mut framebuffer,
                &texture,
            ))
        },
    )
}

pub fn render_to_file(config: &Config) -> Result<(), Box<dyn Error>> {
    let img = render_to_image(config)?;

    let mut output: Box<dyn io::Write> = match config.img_filename.as_str() {
        "-" => Box::new(io::stdout()),
        _ => Box::new(std::fs::File::create(&config.img_filename).unwrap()),
    };

    let mut buff: Vec<u8> = Vec::new();
    let mut cursor = io::Cursor::new(&mut buff);

    match config.format {
        ImageFormat::Png => {
            let encoder = image::codecs::png::PngEncoder::new_with_quality(
                &mut cursor,
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::Adaptive,
            );
            encoder.write_image(
                img.as_bytes(),
                config.width,
                config.height,
                img.color().into(),
            )?;
        }
        _ => img.write_to(&mut cursor, config.format.to_owned())?,
    }

    output.write_all(&buff)?;
    output.flush()?;

    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn render_to_buffer(
    buf_ptr: *mut u8,
    width: u32,
    height: u32,
    model_filename_c: *const c_char,
) -> bool {
    unsafe {
        #[cfg(target_os = "linux")]
        env::set_var("MESA_GL_VERSION_OVERRIDE", "2.1")
    };

    if buf_ptr.is_null() {
        error!("Image buffer pointer is null");
        return false;
    };

    let buf_size = (width * height * 4) as usize;
    let buf = unsafe { slice::from_raw_parts_mut(buf_ptr, buf_size) };

    let model_filename_cstr = unsafe {
        if model_filename_c.is_null() {
            error!("model file path pointer is null");
            return false;
        }
        CStr::from_ptr(model_filename_c)
    };

    let model_filename_str = match model_filename_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            error!("Invalid model file path {:?}", model_filename_cstr);
            return false;
        }
    };

    let config = Config {
        model_filename: model_filename_str.to_string(),
        width,
        height,
        ..Default::default()
    };

    let render_thread = thread::spawn(move || render_to_image(&config).unwrap());
    let img = match render_thread.join() {
        Ok(s) => s,
        Err(e) => {
            error!("Application error: {:?}", e);
            return false;
        }
    };

    match img.as_rgba8() {
        Some(s) => buf.copy_from_slice(s),
        None => {
            error!("Unable to get image");
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::ErrorKind;

    #[test]
    fn cube_stl() {
        let img_filename = "cube-stl.png".to_string();
        let config = Config {
            model_filename: "test_data/cube.stl".to_string(),
            img_filename: img_filename.clone(),
            format: image::ImageFormat::Png,
            ..Default::default()
        };
        match fs::remove_file(&img_filename) {
            Ok(_) => (),
            Err(ref error) if error.kind() == ErrorKind::NotFound => (),
            Err(_) => panic!("Couldn't clean files before testing"),
        }
        render_to_file(&config).expect("Error in render function");
        let size = fs::metadata(img_filename).expect("No file created").len();
        assert_ne!(0, size);
    }

    #[test]
    fn cube_obj() {
        let img_filename = "cube-obj.png".to_string();
        let config = Config {
            model_filename: "test_data/cube.obj".to_string(),
            img_filename: img_filename.clone(),
            format: image::ImageFormat::Png,
            ..Default::default()
        };
        match fs::remove_file(&img_filename) {
            Ok(_) => (),
            Err(ref error) if error.kind() == ErrorKind::NotFound => (),
            Err(_) => panic!("Couldn't clean files before testing"),
        }
        render_to_file(&config).expect("Error in render function");
        let size = fs::metadata(img_filename).expect("No file created").len();
        assert_ne!(0, size);
    }

    #[test]
    fn cube_3mf() {
        let img_filename = "cube-3mf.png".to_string();
        let config = Config {
            model_filename: "test_data/cube.3mf".to_string(),
            img_filename: img_filename.clone(),
            format: image::ImageFormat::Png,
            ..Default::default()
        };
        match fs::remove_file(&img_filename) {
            Ok(_) => (),
            Err(ref error) if error.kind() == ErrorKind::NotFound => (),
            Err(_) => panic!("Couldn't clean files before testing"),
        }
        render_to_file(&config).expect("Error in render function");
        let size = fs::metadata(img_filename).expect("No file created").len();
        assert_ne!(0, size);
    }
}
