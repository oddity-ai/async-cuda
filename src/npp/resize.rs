use crate::memory::DeviceBuffer2D;
use crate::npp::region::Region;
use crate::npp::stream::Stream;
use crate::runtime::Future;

type Result<T> = std::result::Result<T, crate::npp::error::Error>;

/// Resize an image using bilinear interpolation. This function expects a reference to a device
/// image for input, and a mutable reference to a device image to place the output in.
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
/// * `input_region` - Specify region of interest in input image. This can be used to combine crop
///   and resize in a single operation.
/// * `output_region` - Specify region of interest in input image.
/// * `output` - The on-device output image.
/// * `stream` - Stream to use.
pub async fn resize(
    input: &DeviceBuffer2D<u8>,
    input_region: Region,
    output: &mut DeviceBuffer2D<u8>,
    output_region: Region,
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
        crate::ffi::npp::resize::resize(
            input.inner(),
            input_region,
            output.inner_mut(),
            output_region,
            &context,
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::memory::DeviceBuffer2D;
    use crate::npp::stream::Stream;
    use crate::npp::tests::image::*;
    use crate::npp::tests::memory::*;

    #[tokio::test]
    async fn test_resize() {
        // This is the expected result when resizing the RGB flag to 2 by 2 with bilinear
        // interpolation.
        const OUTPUT: Image2x2 = [[R, R], [R, B]];
        const OUTPUT_FLAT: [u8; 2 * 2 * 3] = flatten!(OUTPUT, 2 * 2 * 3);

        let stream = Stream::new().await.unwrap();

        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &stream);
        let mut output = DeviceBuffer2D::<u8>::new(2, 2, 3).await;
        resize(&image, Region::Full, &mut output, Region::Full, &stream)
            .await
            .unwrap();

        let output = to_host_2d!(output, &stream);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[tokio::test]
    async fn test_resize_with_input_region() {
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

        let stream = Stream::new().await.unwrap();

        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &stream);
        let center = Region::Rectangle {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let mut output = DeviceBuffer2D::<u8>::new(4, 4, 3).await;
        resize(&image, center, &mut output, Region::Full, &stream)
            .await
            .unwrap();

        let output = to_host_2d!(output, &stream);
        assert_eq!(&output, &OUTPUT);
    }

    #[tokio::test]
    async fn test_resize_with_output_region() {
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

        let image = to_device_2d!(&INPUT, 2, 2, 3, &stream);
        let mut output = DeviceBuffer2D::<u8>::new(2, 2, 3).await;
        output.fill_with_byte(0x00, &stream).await.unwrap();
        resize(&image, Region::Full, &mut output, bottom_half, &stream)
            .await
            .unwrap();

        let output = to_host_2d!(output, &stream);
        assert_eq!(&output, &EXPECTED_OUTPUT);
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_input_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 2).await;
        let mut output = DeviceBuffer2D::<u8>::new(200, 200, 3).await;
        resize(
            &input,
            Region::Full,
            &mut output,
            Region::Full,
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
        resize(
            &input,
            Region::Full,
            &mut output,
            Region::Full,
            &Stream::null().await,
        )
        .await
        .unwrap();
    }
}
