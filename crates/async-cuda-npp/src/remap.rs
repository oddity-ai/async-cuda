use async_cuda_core::runtime::Future;
use async_cuda_core::DeviceBuffer2D;

use crate::ffi;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Remaps an image using bilinear interpolation. This function expects a reference to a device
/// buffer as inputs, and a mutable reference to a device buffer to store the output of the
/// operation in.
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
/// * `map_x` - On-device X pixel map.
/// * `map_y` - On-device Y pixel map.
/// * `stream` - Stream to use.
pub async fn remap(
    input: &DeviceBuffer2D<u8>,
    output: &mut DeviceBuffer2D<u8>,
    map_x: &DeviceBuffer2D<f32>,
    map_y: &DeviceBuffer2D<f32>,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(input.num_channels(), 3, "input image must be in RGB format");
    assert_eq!(
        output.num_channels(),
        3,
        "output image must be in RGB format"
    );
    assert_eq!(map_x.num_channels(), 1, "map must have one channel");
    assert_eq!(map_y.num_channels(), 1, "map must have one channel");
    assert_eq!(
        output.width(),
        map_x.width(),
        "map x must have same width as output image"
    );
    assert_eq!(
        output.height(),
        map_x.height(),
        "map x must have same height as output image"
    );
    assert_eq!(
        output.width(),
        map_y.width(),
        "map y must have same width as output image"
    );
    assert_eq!(
        output.height(),
        map_y.height(),
        "map y must have same height as output image"
    );

    let context = stream.to_context();
    Future::new(move || {
        ffi::remap::remap(
            input.inner(),
            output.inner_mut(),
            map_x.inner(),
            map_y.inner(),
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
    async fn test_remap() {
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

        let stream = Stream::new().await.unwrap();

        let image = to_device_2d!(&RGB_FLAG, 4, 4, 3, &stream);
        let map_x = to_device_2d!(MAP_X, 4, 4, 1, &stream);
        let map_y = to_device_2d!(MAP_Y, 4, 4, 1, &stream);
        let mut output = DeviceBuffer2D::<u8>::new(4, 4, 3).await;
        assert!(remap(&image, &mut output, &map_x, &map_y, &stream)
            .await
            .is_ok());

        let output = to_host_2d!(output, &stream);
        assert_eq!(&output, &OUTPUT_FLAT);
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_input_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 2).await;
        let map_x = DeviceBuffer2D::<f32>::new(100, 100, 1).await;
        let map_y = DeviceBuffer2D::<f32>::new(100, 100, 1).await;
        let mut output = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        remap(&input, &mut output, &map_x, &map_y, &Stream::null())
            .await
            .unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_output_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        let map_x = DeviceBuffer2D::<f32>::new(100, 100, 1).await;
        let map_y = DeviceBuffer2D::<f32>::new(100, 100, 1).await;
        let mut output = DeviceBuffer2D::<u8>::new(100, 100, 2).await;
        remap(&input, &mut output, &map_x, &map_y, &Stream::null())
            .await
            .unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_map_num_channels_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        let map_x = DeviceBuffer2D::<f32>::new(100, 100, 2).await;
        let map_y = DeviceBuffer2D::<f32>::new(100, 100, 3).await;
        let mut output = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        remap(&input, &mut output, &map_x, &map_y, &Stream::null())
            .await
            .unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_map_width_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        let map_x = DeviceBuffer2D::<f32>::new(120, 100, 1).await;
        let map_y = DeviceBuffer2D::<f32>::new(120, 100, 1).await;
        let mut output = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        remap(&input, &mut output, &map_x, &map_y, &Stream::null())
            .await
            .unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_map_height_incorrect() {
        let input = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        let map_x = DeviceBuffer2D::<f32>::new(100, 120, 1).await;
        let map_y = DeviceBuffer2D::<f32>::new(100, 120, 1).await;
        let mut output = DeviceBuffer2D::<u8>::new(100, 100, 3).await;
        remap(&input, &mut output, &map_x, &map_y, &Stream::null())
            .await
            .unwrap();
    }
}
