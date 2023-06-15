use crate::constant_border::ConstantBorder;
use crate::ffi::context::Context;
use crate::ffi::result;
use crate::ffi::utils::cpp;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchroneous implementation of [`crate::copy_constant_border()`].
///
/// Refer to [`crate::copy_constant_border()`] for documentation.
pub fn copy_constant_border(
    input: &async_cuda_core::ffi::memory::DeviceBuffer2D<u8>,
    output: &mut async_cuda_core::ffi::memory::DeviceBuffer2D<u8>,
    border: &ConstantBorder,
    context: &Context,
) -> Result<()> {
    assert_eq!(input.num_channels, 3, "input image must be in RGB format");
    assert_eq!(output.num_channels, 3, "output image must be in RGB format");

    let (src_pitch, src_width, src_height) = (input.pitch, input.width as i32, input.height as i32);
    let (dst_pitch, dst_width, dst_height) =
        (output.pitch, output.width as i32, output.height as i32);

    let (border_left, border_top) = (border.left as i32, border.top as i32);
    let border_color_ptr = border.color.as_ptr();

    let src_ptr = input.as_internal().as_ptr();
    let dst_ptr = output.as_mut_internal().as_mut_ptr();
    let context_ptr = context.as_ptr();
    let ret = cpp!(unsafe [
        src_ptr as "const void*",
        src_pitch as "std::size_t",
        src_width as "std::int32_t",
        src_height as "std::int32_t",
        dst_ptr as "void*",
        dst_pitch as "std::size_t",
        dst_width as "std::int32_t",
        dst_height as "std::int32_t",
        border_left as "std::int32_t",
        border_top as "std::int32_t",
        border_color_ptr as "const std::uint8_t*",
        context_ptr as "void*"
    ] -> i32 as "std::int32_t" {
        NppiSize src_size = { src_width, src_height };
        NppiSize dst_size = { dst_width, dst_height };
        return nppiCopyConstBorder_8u_C3R_Ctx(
            (const Npp8u*) src_ptr,
            src_pitch,
            src_size,
            (Npp8u*) dst_ptr,
            dst_pitch,
            dst_size,
            border_top,
            border_left,
            border_color_ptr,
            *((NppStreamContext*) context_ptr)
        );
    });
    result!(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::image::*;
    use crate::tests::sync::memory::*;

    use crate::ffi::context::Context;

    #[test]
    fn test_copy_constant_border() {
        // Input image is 1x2 and just contains one red and one green pixel.
        const INPUT: [[Pixel; 2]; 1] = [[R, G]];
        const INPUT_FLAT: [u8; 6] = flatten!(INPUT, 6);

        // Expected output of copy constant border with left border of 1 and top border of 2, if
        // the border color is blue.
        const OUTPUT: [[Pixel; 4]; 5] = [
            [B, B, B, B],
            [B, B, B, B],
            [B, R, G, B],
            [B, B, B, B],
            [B, B, B, B],
        ];
        const OUTPUT_FLAT: [u8; 4 * 5 * 3] = flatten!(OUTPUT, 4 * 5 * 3);

        let context = Context::from_null_stream();

        let image = to_device_2d!(&INPUT_FLAT, 2, 1, 3, &context);
        let mut output = async_cuda_core::ffi::memory::DeviceBuffer2D::<u8>::new(4, 5, 3);
        copy_constant_border(&image, &mut output, &ConstantBorder::new(1, 2, B), &context).unwrap();

        let output = to_host_2d!(output, &context);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_input_num_channels_incorrect() {
        let input = async_cuda_core::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2);
        let mut output = async_cuda_core::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 3);
        copy_constant_border(
            &input,
            &mut output,
            &ConstantBorder::black(10, 20),
            &Context::from_null_stream(),
        )
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_output_num_channels_incorrect() {
        let input = async_cuda_core::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 3);
        let mut output = async_cuda_core::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 2);
        copy_constant_border(
            &input,
            &mut output,
            &ConstantBorder::black(10, 20),
            &Context::from_null_stream(),
        )
        .unwrap();
    }
}
