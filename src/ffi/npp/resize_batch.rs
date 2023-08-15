use cpp::cpp;

use crate::ffi::npp::context::Context;
use crate::ffi::npp::result;
use crate::npp::region::Region;

type Result<T> = std::result::Result<T, crate::npp::error::Error>;

/// Synchroneous implementation of [`crate::resize_batch()`].
///
/// Refer to [`crate::resize_batch()`] for documentation.
pub fn resize_batch(
    inputs_and_outputs: &mut [(
        &crate::ffi::memory::DeviceBuffer2D<u8>,
        &mut crate::ffi::memory::DeviceBuffer2D<u8>,
    )],
    input_region: Region,
    output_region: Region,
    context: &Context,
) -> Result<()> {
    assert!(
        !inputs_and_outputs.is_empty(),
        "batch must have at least one item"
    );

    let (first_input, first_output) = &inputs_and_outputs[0];
    let first_input_width = first_input.width;
    let first_input_height = first_input.height;
    let first_output_width = first_output.width;
    let first_output_height = first_output.height;
    for (input, output) in inputs_and_outputs.iter() {
        assert_eq!(
            input.width, first_input_width,
            "all inputs in batch must have the same width",
        );
        assert_eq!(
            input.height, first_input_height,
            "all inputs in batch must have the same height",
        );
        assert_eq!(
            output.width, first_output_width,
            "all outputs in batch must have the same width",
        );
        assert_eq!(
            output.height, first_output_height,
            "all outputs in batch must have the same height",
        );
        assert_eq!(
            input.num_channels, 3,
            "all inputs and outputs must be in RGB format"
        );
        assert_eq!(
            output.num_channels, 3,
            "all inputs and outputs must be in RGB format"
        );
    }

    let batch_size = inputs_and_outputs.len();

    let (src_width, src_height) = (first_input_width as i32, first_input_height as i32);
    let (src_rect_x, src_rect_y, src_rect_width, src_rect_height) =
        input_region.resolve_to_xywh(src_width as usize, src_height as usize);
    let (src_rect_x, src_rect_y, src_rect_width, src_rect_height) = (
        src_rect_x as i32,
        src_rect_y as i32,
        src_rect_width as i32,
        src_rect_height as i32,
    );

    let (dst_width, dst_height) = (first_output_width as i32, first_output_height as i32);
    let (dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height) =
        output_region.resolve_to_xywh(dst_width as usize, dst_height as usize);
    let (dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height) = (
        dst_rect_x as i32,
        dst_rect_y as i32,
        dst_rect_width as i32,
        dst_rect_height as i32,
    );

    let srcs = inputs_and_outputs
        .iter()
        // SAFETY: This is safe because we keep the original input and output device buffers around
        // for the duration of this call.
        .map(|(input, _)| input.as_internal().as_ptr())
        .collect::<Vec<_>>();
    let src_pitches = inputs_and_outputs
        .iter()
        .map(|(input, _)| input.pitch)
        .collect::<Vec<_>>();
    let dsts = inputs_and_outputs
        .iter_mut()
        // SAFETY: This is safe because we keep the original input and output device buffers around
        // for the duration of this call.
        .map(|(_, output)| output.as_mut_internal().as_mut_ptr())
        .collect::<Vec<_>>();
    let dst_pitches = inputs_and_outputs
        .iter()
        .map(|(_, output)| output.pitch)
        .collect::<Vec<_>>();

    let src_array = srcs.as_ptr();
    let src_pitches_array = src_pitches.as_ptr();
    let dst_array = dsts.as_ptr();
    let dst_pitches_array = dst_pitches.as_ptr();

    let context_ptr = context.as_ptr();
    let ret = cpp!(unsafe [
        src_array as "const void* const*",
        src_pitches_array as "const std::size_t*",
        src_width as "std::int32_t",
        src_height as "std::int32_t",
        src_rect_x as "std::int32_t",
        src_rect_y as "std::int32_t",
        src_rect_width as "std::int32_t",
        src_rect_height as "std::int32_t",
        dst_array as "void* const*",
        dst_pitches_array as "const std::size_t*",
        dst_width as "std::int32_t",
        dst_height as "std::int32_t",
        dst_rect_x as "std::int32_t",
        dst_rect_y as "std::int32_t",
        dst_rect_width as "std::int32_t",
        dst_rect_height as "std::int32_t",
        batch_size as "std::size_t",
        context_ptr as "void*"
    ] -> i32 as "std::int32_t" {
        NppStatus ret {};
        cudaError_t ret_cuda {};

        NppiSize src_size = { src_width, src_height };
        NppiSize dst_size = { dst_width, dst_height };
        NppiRect src_rect = { src_rect_x, src_rect_y, src_rect_width, src_rect_height };
        NppiRect dst_rect = { dst_rect_x, dst_rect_y, dst_rect_width, dst_rect_height };

        NppiResizeBatchCXR* batch_host = new NppiResizeBatchCXR[batch_size];
        for (std::size_t i = 0; i < batch_size; i++) {
            batch_host[i].pSrc = src_array[i];
            batch_host[i].nSrcStep = src_pitches_array[i];
            batch_host[i].pDst = dst_array[i];
            batch_host[i].nDstStep = dst_pitches_array[i];
        }

        NppiResizeBatchCXR* batch = nullptr;
        ret_cuda = cudaMallocAsync(
            &batch,
            batch_size * sizeof(NppiResizeBatchCXR),
            ((NppStreamContext*) context_ptr)->hStream
        );
        if (ret_cuda != cudaSuccess)
            goto cleanup;
        ret_cuda = cudaMemcpyAsync(
            batch,
            batch_host,
            batch_size * sizeof(NppiResizeBatchCXR),
            cudaMemcpyHostToDevice,
            ((NppStreamContext*) context_ptr)->hStream
        );
        if (ret_cuda != cudaSuccess)
            goto cleanup;

        ret = nppiResizeBatch_8u_C3R_Ctx(
            src_size,
            src_rect,
            dst_size,
            dst_rect,
            // We use bilinear interpolation, which is the fastest resize method that does not
            // produce messed up quality.
            NPPI_INTER_LINEAR,
            batch,
            batch_size,
            *((NppStreamContext*) context_ptr)
        );

    cleanup:
        if (batch != nullptr)
            cudaFreeAsync(
                batch,
                ((NppStreamContext*) context_ptr)->hStream
            );
        if (batch_host != nullptr)
            delete[] batch_host;

        return ret;
    });
    result!(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ffi::npp::context::Context;
    use crate::npp::tests::sync::memory::*;

    #[test]
    fn test_resize_batch() {
        #[rustfmt::skip]
        const INPUT: [u8; 12] = [
            10, 10, 10, 20, 20, 20,
            30, 30, 30, 40, 40, 40,
        ];
        #[rustfmt::skip]
        const EXPECTED_OUTPUT: [u8; 27] = [
            10, 10, 10, 14, 14, 14, 20, 20, 20,
            18, 18, 18, 23, 23, 23, 28, 28, 28,
            30, 30, 30, 34, 34, 34, 40, 40, 40,
        ];

        let context = Context::from_null_stream();

        let mut inputs_and_outputs = (0..10)
            .map(|_| {
                let image = to_device_2d!(&INPUT, 2, 2, 3, &context);
                let output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(3, 3, 3);
                (image, output)
            })
            .collect::<Vec<_>>();
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(
            &mut inputs_and_outputs_ref,
            Region::Full,
            Region::Full,
            &context,
        )
        .unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &context);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[test]
    fn test_resize_batch_with_input_region() {
        #[rustfmt::skip]
        const INPUT: [u8; 27] = [
            99, 99, 99, 10, 10, 10, 20, 20, 20,
            99, 99, 99, 30, 30, 30, 40, 40, 40,
            99, 99, 99, 99, 99, 99, 99, 99, 99,
        ];
        #[rustfmt::skip]
        const EXPECTED_OUTPUT: [u8; 27] = [
            32, 32, 32, 14, 14, 14, 20, 20, 20,
            39, 39, 39, 23, 23, 23, 28, 28, 28,
            52, 52, 52, 40, 40, 40, 45, 45, 45,
        ];

        let context = Context::from_null_stream();
        let center = Region::Rectangle {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        };
        let mut inputs_and_outputs = (0..10)
            .map(|_| {
                let image = to_device_2d!(&INPUT, 3, 3, 3, &context);
                let output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(3, 3, 3);
                (image, output)
            })
            .collect::<Vec<_>>();
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(&mut inputs_and_outputs_ref, center, Region::Full, &context).unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &context);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[test]
    fn test_resize_batch_with_output_region() {
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

        let mut inputs_and_outputs = (0..10)
            .map(|_| {
                let image = to_device_2d!(&INPUT, 2, 2, 3, &context);
                let mut output = crate::ffi::memory::DeviceBuffer2D::<u8>::new(2, 2, 3);
                output.fill_with_byte(0x00, context.stream.inner()).unwrap();
                (image, output)
            })
            .collect::<Vec<_>>();
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(
            &mut inputs_and_outputs_ref,
            Region::Full,
            bottom_half,
            &context,
        )
        .unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &context);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_input_num_channels_incorrect() {
        let mut inputs_and_outputs = vec![
            (
                crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2),
                crate::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 2),
            ),
            (
                crate::ffi::memory::DeviceBuffer2D::<u8>::new(100, 100, 2),
                crate::ffi::memory::DeviceBuffer2D::<u8>::new(200, 200, 2),
            ),
        ];
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(
            &mut inputs_and_outputs_ref,
            Region::Full,
            Region::Full,
            &Context::from_null_stream(),
        )
        .unwrap();
    }
}
