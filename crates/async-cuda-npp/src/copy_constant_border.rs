use async_cuda_core::runtime::Future;
use async_cuda_core::DeviceBuffer2D;

use crate::constant_border::ConstantBorder;
use crate::ffi;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Copy an image with a constant border. This function expects a reference to a device image for
/// input, and a mutable reference to a device image to place the output in.
///
/// This function assumes the following about the input and output images:
/// * Images are in RGB format.
/// * Images are in standard memory order, i.e. HWC.
///
/// # Stream ordered semantics
///
/// This function uses stream ordered semantics. It can only be guaranteed to complete sequentially
/// relative to operations scheduled on the same stream or the default stream.
///
/// # Arguments
///
/// * `input` - The on-device input image.
/// * `output` - The on-device output image.
/// * `constant_border` - The constant border parameters to apply.
/// * `stream` - Stream to use.
pub async fn copy_constant_border(
    input: &DeviceBuffer2D<u8>,
    output: &mut DeviceBuffer2D<u8>,
    constant_border: &ConstantBorder,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(input.num_channels(), 3, "input image must be in RGB format");
    assert_eq!(
        output.num_channels(),
        3,
        "output image must be in RGB format"
    );

    let context = stream.to_context();
    Future::new(move || {
        ffi::copy_constant_border::copy_constant_border(
            input.inner(),
            output.inner_mut(),
            constant_border,
            &context,
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stream::Stream;
    use crate::tests::image::*;
    use crate::tests::memory::*;

    use async_cuda_core::DeviceBuffer2D;

    #[tokio::test]
    async fn test_copy_constant_border() {
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

        let stream = Stream::new().await.unwrap();

        let image = to_device_2d!(&INPUT_FLAT, 2, 1, 3, &stream);
        let mut output = DeviceBuffer2D::<u8>::new(4, 5, 3).await;
        copy_constant_border(&image, &mut output, &ConstantBorder::new(1, 2, B), &stream)
            .await
            .unwrap();

        let output = to_host_2d!(output, &stream);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_input_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 2).await;
        let mut output = DeviceBuffer2D::<u8>::new(200, 200, 3).await;
        copy_constant_border(
            &input,
            &mut output,
            &ConstantBorder::black(10, 20),
            &Stream::null().await,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_output_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        let mut output = DeviceBuffer2D::<u8>::new(200, 200, 2).await;
        copy_constant_border(
            &input,
            &mut output,
            &ConstantBorder::black(10, 20),
            &Stream::null().await,
        )
        .await
        .unwrap();
    }
}
