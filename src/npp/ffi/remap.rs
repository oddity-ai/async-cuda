use cpp::cpp;

use crate::npp::ffi::context::Context;
use crate::npp::ffi::result;

type Result<T> = std::result::Result<T, crate::npp::error::Error>;

/// Synchroneous implementation of [`crate::remap()`].
///
/// Refer to [`crate::remap()`] for documentation.
pub fn remap(
    input: &crate::ffi::memory::DeviceBuffer2D<u8>,
    output: &mut crate::ffi::memory::DeviceBuffer2D<u8>,
    map_x: &crate::ffi::memory::DeviceBuffer2D<f32>,
    map_y: &crate::ffi::memory::DeviceBuffer2D<f32>,
    context: &Context,
) -> Result<()> {
    assert_eq!(input.num_channels, 3, "input image must be in RGB format");
    assert_eq!(output.num_channels, 3, "output image must be in RGB format");
    assert_eq!(map_x.num_channels, 1, "map must have one channel");
    assert_eq!(map_y.num_channels, 1, "map must have one channel");
    assert_eq!(
        output.width, map_x.width,
        "map x must have same width as output image"
    );
    assert_eq!(
        output.height, map_x.height,
        "map x must have same height as output image"
    );
    assert_eq!(
        output.width, map_y.width,
        "map y must have same width as output image"
    );
    assert_eq!(
        output.height, map_y.height,
        "map y must have same height as output image"
    );

    let (src_width, src_height, src_pitch) = (input.width as i32, input.height as i32, input.pitch);
    let (dst_width, dst_height, dst_pitch) =
        (output.width as i32, output.height as i32, output.pitch);

    let map_x_pitch = map_x.pitch;
    let map_y_pitch = map_y.pitch;

    let src_ptr = input.as_internal().as_ptr();
    let dst_ptr = output.as_mut_internal().as_mut_ptr();
    let map_x_ptr = map_x.as_internal().as_ptr();
    let map_y_ptr = map_y.as_internal().as_ptr();
    let context_ptr = context.as_ptr();
    let ret = cpp!(unsafe [
        src_ptr as "const std::uint8_t*",
        src_width as "std::int32_t",
        src_height as "std::int32_t",
        src_pitch as "std::size_t",
        map_x_ptr as "const float*",
        map_x_pitch as "std::size_t",
        map_y_ptr as "const float*",
        map_y_pitch as "std::size_t",
        dst_ptr as "std::uint8_t*",
        dst_width as "std::int32_t",
        dst_height as "std::int32_t",
        dst_pitch as "std::size_t",
        context_ptr as "void*"
    ] -> i32 as "std::int32_t" {
        NppiSize src_size = { src_width, src_height };
        NppiSize dst_size = { dst_width, dst_height };
        NppiRect src_rect = { 0, 0, src_width, src_height };
        return nppiRemap_8u_C3R_Ctx(
            (const Npp8u*) src_ptr,
            src_size,
            src_pitch,
            src_rect,
            (const Npp32f*) map_x_ptr,
            map_x_pitch,
            (const Npp32f*) map_y_ptr,
            map_y_pitch,
            (Npp8u*) dst_ptr,
            dst_pitch,
            dst_size,
            // We use bilinear interpolation, which is the fastest resize method that does not
            // produce messed up quality.
            NPPI_INTER_LINEAR,
            *((NppStreamContext*) context_ptr)
        );
    });
    result!(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::npp::ffi::context::Context;
    use crate::npp::tests::image::*;
    use crate::npp::tests::sync::memory::*;

    #[test]
    fn test_remap() {
        const MAP_X: &[f32; 16] = &[
            0.0, 1.0, 2.0, 3.0, // No mapping at all
            1.0, 1.0, 2.0, 2.0, // Ignore the red border
            1.0, 1.0, 2.0, 2.0, // Ignore the red border
            1.0, 1.0, 2.0, 2.0, // Ignore the red border
        ];
        const MAP_Y: &[f32; 16] = &[
            0.0, 0.0, 0.0, 0.0, // No mapping at all
            1.0, 1.0, 1.0, 1.0, // Take from green band
            1.0, 1.0, 1.0, 1.0, // Take from green band
            2.0, 2.0, 2.0, 2.0, // Take from blue band
        ];
        const OUTPUT: Image4x4 = [
            [R, R, R, R], // Red band
            [G, G, G, G], // Green band
            [G, G, G, G], // Green band
            [B, B, B, B], // Blue band
        ];
        const OUTPUT_FLAT: [u8; 4 * 4 * 3] = flatten!(OUTPUT, 4 * 4 * 3);

        let context = Context::from_null_stream();

        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &context);
        let map_x = to_device_2d!(MAP_X, 4, 4, 1, &context);
        let map_y = to_device_2d!(MAP_Y, 4, 4, 1, &context);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(4, 4, 3);
        assert!(remap(&image, &mut output, &map_x, &map_y, &context).is_ok());

        let output = to_host_2d!(output, &context);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_input_num_channels_incorrect() {
        let context = Context::from_null_stream();
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2);
        let map_x = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 1);
        let map_y = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 1);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        remap(&input, &mut output, &map_x, &map_y, &context).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_output_num_channels_incorrect() {
        let context = Context::from_null_stream();
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let map_x = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 1);
        let map_y = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 1);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2);
        remap(&input, &mut output, &map_x, &map_y, &context).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_map_num_channels_incorrect() {
        let context = Context::from_null_stream();
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let map_x = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 2);
        let map_y = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 100, 3);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        remap(&input, &mut output, &map_x, &map_y, &context).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_map_width_incorrect() {
        let context = Context::from_null_stream();
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let map_x = crate::ffi::memory::DeviceBuffer2D::<f32>::new(120, 100, 1);
        let map_y = crate::ffi::memory::DeviceBuffer2D::<f32>::new(120, 100, 1);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        remap(&input, &mut output, &map_x, &map_y, &context).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_map_height_incorrect() {
        let context = Context::from_null_stream();
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let map_x = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 120, 1);
        let map_y = crate::ffi::memory::DeviceBuffer2D::<f32>::new(100, 120, 1);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        remap(&input, &mut output, &map_x, &map_y, &context).unwrap();
    }
}
