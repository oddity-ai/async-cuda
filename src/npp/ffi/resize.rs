use cpp::cpp;

use crate::npp::ffi::context::Context;
use crate::npp::ffi::result;
use crate::npp::region::Region;

type Result<T> = std::result::Result<T, crate::npp::error::Error>;

/// Synchroneous implementation of [`crate::resize()`].
///
/// Refer to [`crate::resize()`] for documentation.
pub fn resize(
    input: &crate::ffi::memory::DeviceBuffer2D<u8>,
    input_region: Region,
    output: &mut crate::ffi::memory::DeviceBuffer2D<u8>,
    output_region: Region,
    context: &Context,
) -> Result<()> {
    assert_eq!(input.num_channels, 3, "input image must be in RGB format");
    assert_eq!(output.num_channels, 3, "output image must be in RGB format");

    let (src_pitch, src_width, src_height) = (input.pitch, input.width as i32, input.height as i32);
    let (src_rect_x, src_rect_y, src_rect_width, src_rect_height) =
        input_region.resolve_to_xywh(src_width as usize, src_height as usize);
    let (src_rect_x, src_rect_y, src_rect_width, src_rect_height) = (
        src_rect_x as i32,
        src_rect_y as i32,
        src_rect_width as i32,
        src_rect_height as i32,
    );

    let (dst_pitch, dst_width, dst_height) =
        (output.pitch, output.width as i32, output.height as i32);
    let (dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height) =
        output_region.resolve_to_xywh(dst_width as usize, dst_height as usize);
    let (dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height) = (
        dst_rect_x as i32,
        dst_rect_y as i32,
        dst_rect_width as i32,
        dst_rect_height as i32,
    );

    let src_ptr = input.as_internal().as_ptr();
    let dst_ptr = output.as_mut_internal().as_mut_ptr();
    let context_ptr = context.as_ptr();
    let ret = cpp!(unsafe [
        src_ptr as "const void*",
        src_pitch as "std::size_t",
        src_width as "std::int32_t",
        src_height as "std::int32_t",
        src_rect_x as "std::int32_t",
        src_rect_y as "std::int32_t",
        src_rect_width as "std::int32_t",
        src_rect_height as "std::int32_t",
        dst_ptr as "void*",
        dst_pitch as "std::size_t",
        dst_width as "std::int32_t",
        dst_height as "std::int32_t",
        dst_rect_x as "std::int32_t",
        dst_rect_y as "std::int32_t",
        dst_rect_width as "std::int32_t",
        dst_rect_height as "std::int32_t",
        context_ptr as "void*"
    ] -> i32 as "std::int32_t" {
        NppiSize src_size = { src_width, src_height };
        NppiSize dst_size = { dst_width, dst_height };
        NppiRect src_rect = { src_rect_x, src_rect_y, src_rect_width, src_rect_height };
        NppiRect dst_rect = { dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height };
        return nppiResize_8u_C3R_Ctx(
            (const Npp8u*) src_ptr,
            src_pitch,
            src_size,
            src_rect,
            (Npp8u*) dst_ptr,
            dst_pitch,
            dst_size,
            dst_rect,
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
    fn test_resize() {
        // This is the expected result when resizing the RGB flag to 2 by 2 with bilinear
        // interpolation.
        const OUTPUT: Image2x2 = [[R, R], [R, B]];
        const OUTPUT_FLAT: [u8; 2 * 2 * 3] = flatten!(OUTPUT, 2 * 2 * 3);

        let context = Context::from_null_stream();

        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &context);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(2, 2, 3);
        resize(&image, Region::Full, &mut output, Region::Full, &context).unwrap();

        let output = to_host_2d!(output, &context);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[test]
    fn test_resize_with_input_region() {
        // This is the raw expected result when resizing the center part of the RGB flag from two by
        // to two four by four.
        #[rustfmt::skip]
        #[allow(clippy::zero_prefixed_literal)]
        const OUTPUT: [u8; 4 * 4 * 3] = [
            000, 255, 000, 000, 255, 000, 000, 255, 000, 064, 191, 000,
            000, 191, 064, 000, 191, 064, 000, 191, 064, 064, 143, 048,
            000, 064, 191, 000, 064, 191, 000, 064, 191, 064, 048, 143,
            064, 000, 191, 064, 000, 191, 064, 000, 191, 112, 000, 143,
        ];

        let context = Context::from_null_stream();
        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &context);
        let center = Region::Rectangle {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(4, 4, 3);
        resize(&image, center, &mut output, Region::Full, &context).unwrap();

        let output = to_host_2d!(output, &context);
        assert_eq!(&output, &OUTPUT);
    }

    #[test]
    fn test_resize_with_output_region() {
        #[rustfmt::skip]
        const INPUT: [u8; 2 * 2 * 3] = [
            0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
            0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
        ];
        #[rustfmt::skip]
        const EXPECTED_OUTPUT: [u8; 2 * 2 * 3] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
        ];

        let context = Context::from_null_stream();
        let bottom_half = Region::Rectangle {
            x: 0,
            y: 1,
            width: 2,
            height: 1,
        };

        let image = to_device_2d!(&INPUT, 2, 2, 3, &context);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(2, 2, 3);
        output.fill_with_byte(0x00, context.stream.inner()).unwrap();
        resize(&image, Region::Full, &mut output, bottom_half, &context).unwrap();

        let output = to_host_2d!(output, &context);
        assert_eq!(&output, &EXPECTED_OUTPUT);
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_input_num_channels_incorrect() {
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 3);
        resize(
            &input,
            Region::Full,
            &mut output,
            Region::Full,
            &Context::from_null_stream(),
        )
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_output_num_channels_incorrect() {
        let input = crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 2);
        resize(
            &input,
            Region::Full,
            &mut output,
            Region::Full,
            &Context::from_null_stream(),
        )
        .unwrap();
    }
}
