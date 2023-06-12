use async_cuda_core::runtime::Future;
use async_cuda_core::DeviceBuffer2D;

use crate::ffi;
use crate::region::Region;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Resize a batch of images using bilinear interpolation. This function expects a batch with
/// on-device input and output buffers.
///
/// This function assumes the following about the input and output images:
/// * Images are in RGB format.
/// * Images are in standard memory order, i.e. HWC.
///
/// This is the batched version of [`crate::resize`].
///
/// # Stability
///
/// This function is only available when the `unstable` feature is enabled. Testing shows that the
/// batched version can be imprecise when the input image dimensions are small.
///
/// Currently identified suspicious behavior:
/// * It does not necessarily produce the same output over a batch of images that would have been
///   produced if the non-batched version of resize were used on each image individually.
/// * When invoking batched resize to resize to the same dimensions as the input, it might not
///   reproduce the input image exactly.
///
/// # Stream ordered semantics
///
/// This function uses stream ordered semantics. It can only be guaranteed to complete sequentially
/// relative to operations scheduled on the same stream or the default stream.
///
/// # Arguments
///
/// * `batch` - The on-device input and output images as batch.
/// * `input_region` - Specify region of interest in input image. This can be used to combine crop
///   and resize in a single operation.
/// * `output_region` - Specify region of interest in output image.
/// * `stream` - Stream to use.
pub async fn resize_batch(
    inputs_and_outputs: &mut [(&DeviceBuffer2D<u8>, &mut DeviceBuffer2D<u8>)],
    input_region: Region,
    output_region: Region,
    stream: &Stream,
) -> Result<()> {
    assert!(
        !inputs_and_outputs.is_empty(),
        "batch must have at least one item"
    );

    let (first_input, first_output) = &inputs_and_outputs[0];
    let first_input_width = first_input.width();
    let first_input_height = first_input.height();
    let first_output_width = first_output.width();
    let first_output_height = first_output.height();
    for (input, output) in inputs_and_outputs.iter() {
        assert_eq!(
            input.width(),
            first_input_width,
            "all inputs in batch must have the same width",
        );
        assert_eq!(
            input.height(),
            first_input_height,
            "all inputs in batch must have the same height",
        );
        assert_eq!(
            output.width(),
            first_output_width,
            "all outputs in batch must have the same width",
        );
        assert_eq!(
            output.height(),
            first_output_height,
            "all outputs in batch must have the same height",
        );
        assert_eq!(
            input.num_channels(),
            3,
            "all inputs and outputs must be in RGB format"
        );
        assert_eq!(
            output.num_channels(),
            3,
            "all inputs and outputs must be in RGB format"
        );
    }

    let context = stream.to_context();
    Future::new(move || {
        let mut inputs_and_outputs_inner = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (input.inner(), output.inner_mut()))
            .collect::<Vec<_>>();
        ffi::resize_batch::resize_batch(
            inputs_and_outputs_inner.as_mut_slice(),
            input_region,
            output_region,
            &context,
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stream::Stream;
    use crate::tests::memory::*;

    use async_cuda_core::DeviceBuffer2D;

    use futures::future;

    #[tokio::test]
    async fn test_resize_batch() {
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

        let stream = Stream::new().await.unwrap();

        let mut inputs_and_outputs = future::join_all((0..10).map(|_| async {
            let image = to_device_2d!(&INPUT, 2, 2, 3, &stream);
            let output = DeviceBuffer2D::<u8>::new(3, 3, 3).await;
            (image, output)
        }))
        .await;
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(
            &mut inputs_and_outputs_ref,
            Region::Full,
            Region::Full,
            &stream,
        )
        .await
        .unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &stream);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[tokio::test]
    async fn test_resize_batch_with_input_region() {
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

        let stream = Stream::new().await.unwrap();

        let center = Region::Rectangle {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        };

        let mut inputs_and_outputs = future::join_all((0..10).map(|_| async {
            let image = to_device_2d!(&INPUT, 3, 3, 3, &stream);
            let output = DeviceBuffer2D::<u8>::new(3, 3, 3).await;
            (image, output)
        }))
        .await;
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(&mut inputs_and_outputs_ref, center, Region::Full, &stream)
            .await
            .unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &stream);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[tokio::test]
    async fn test_resize_batch_with_output_region() {
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

        let stream = Stream::new().await.unwrap();
        let bottom_half = Region::Rectangle {
            x: 0,
            y: 1,
            width: 2,
            height: 1,
        };

        let mut inputs_and_outputs = future::join_all((0..10).map(|_| async {
            let image = to_device_2d!(&INPUT, 2, 2, 3, &stream);
            let mut output = DeviceBuffer2D::<u8>::new(2, 2, 3).await;
            output.fill_with_byte(0x00, &stream).await.unwrap();
            (image, output)
        }))
        .await;
        let mut inputs_and_outputs_ref = inputs_and_outputs
            .iter_mut()
            .map(|(input, output)| (&*input, output))
            .collect::<Vec<_>>();
        resize_batch(
            &mut inputs_and_outputs_ref,
            Region::Full,
            bottom_half,
            &stream,
        )
        .await
        .unwrap();

        for (_, output) in inputs_and_outputs {
            let output = to_host_2d!(output, &stream);
            assert_eq!(&output, &EXPECTED_OUTPUT);
        }
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_input_num_channels_incorrect() {
        let mut inputs_and_outputs = vec![
            (
                DeviceBuffer2D::<u8>::new(100, 100, 2).await,
                DeviceBuffer2D::<u8>::new(200, 200, 2).await,
            ),
            (
                DeviceBuffer2D::<u8>::new(100, 100, 2).await,
                DeviceBuffer2D::<u8>::new(200, 200, 2).await,
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
            &Stream::null(),
        )
        .await
        .unwrap();
    }
}
