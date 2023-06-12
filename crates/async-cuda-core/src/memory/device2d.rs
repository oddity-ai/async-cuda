use crate::ffi;
use crate::memory::HostBuffer;
use crate::runtime::Future;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// A buffer on the device.
///
/// # Example
///
/// Copying data from a [`HostBuffer`] to a [`DeviceBuffer2D`]:
///
/// ```
/// # use async_cuda_core::{DeviceBuffer2D, HostBuffer, Stream};
/// # tokio_test::block_on(async {
/// let stream = Stream::new().await.unwrap();
/// let all_ones = vec![1_u8; 300];
/// let host_buffer = HostBuffer::<u8>::from_slice(&all_ones).await;
/// let mut device_buffer = DeviceBuffer2D::<u8>::new(10, 10, 3).await;
/// device_buffer.copy_from(&host_buffer, &stream).await.unwrap();
/// # })
/// ```
pub struct DeviceBuffer2D<T: Copy + 'static> {
    inner: ffi::memory::DeviceBuffer2D<T>,
}

impl<T: Copy + 'static> DeviceBuffer2D<T> {
    /// Allocates 2D memory on the device.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)
    ///
    /// # Arguments
    ///
    /// * `width` - Width of 2-dimensional buffer.
    /// * `height` - Height of 2-dimensional buffer.
    /// * `num_channels` - Number of channels per item.
    pub async fn new(width: usize, height: usize, num_channels: usize) -> Self {
        let inner =
            Future::new(move || ffi::memory::DeviceBuffer2D::<T>::new(width, height, num_channels))
                .await;
        Self { inner }
    }

    /// Allocate memory on the device, and copy 3D array from host into it.
    ///
    /// This function creates a temporary [`HostBuffer`], copies the slice into it, then finally
    /// copies the data from the host buffer to the [`DeviceBuffer`].
    ///
    /// The given stream is automatically synchronized, since the temporary host buffer might
    /// otherwise be dropped before the copy can complete.
    ///
    /// # Arguments
    ///
    /// * `array` - 3-dimensional array to copy into the buffer. The first and second dimensions are
    ///   equivalent to the height and width of the 2D buffer (respectively), and the third
    ///   dimension is the number of channels.
    /// * `stream` - Stream to use.
    #[cfg(feature = "ndarray")]
    pub async fn from_array(array: &ndarray::ArrayView3<'_, T>, stream: &Stream) -> Result<Self> {
        let host_buffer = HostBuffer::from_array(array).await;
        let (height, width, num_channels) = array.dim();
        let mut this = Self::new(width, height, num_channels).await;
        this.copy_from(&host_buffer, stream).await?;
        Ok(this)
    }

    /// Copies memory from the provided pinned host buffer to this 2D buffer.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// The host buffer must be of the same size. For the 2D buffer, the total number of elements is
    /// `width` times `height` times `num_channels`.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1)
    ///
    /// # Pinned transfer
    ///
    /// The other buffer (of type [`HostBuffer`]) is always a pinned buffer. This function is
    /// guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `other` - Buffer to copy from.
    /// * `stream` - Stream to use.
    #[inline]
    pub async fn copy_from(&mut self, other: &HostBuffer<T>, stream: &Stream) -> Result<()> {
        // SAFETY: Stream is synchronized after this.
        unsafe {
            self.copy_from_async(other, stream).await?;
        }
        stream.synchronize().await?;
        Ok(())
    }

    /// Copies memory from the provided pinned host buffer to this 2D buffer.
    ///
    /// The host buffer must be of the same size. For the 2D buffer, the total number of elements is
    /// `width` times `height` times `num_channels`.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1)
    ///
    /// # Pinned transfer
    ///
    /// The other buffer (of type [`HostBuffer`]) is always a pinned buffer. This function is
    /// guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the operation might not have completed when the function
    /// returns, and thus the state of the buffer is undefined.
    ///
    /// # Arguments
    ///
    /// * `other` - Buffer to copy from.
    /// * `stream` - Stream to use.
    pub async unsafe fn copy_from_async(
        &mut self,
        other: &HostBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(self.num_elements(), other.num_elements());
        Future::new(move || self.inner.copy_from_async(other.inner(), stream.inner())).await
    }

    /// Copies memory from this 2D buffer to the provided pinned host buffer.
    ///
    /// The host buffer must be of the same size. For the 2D buffer, the total number of elements is
    /// `width` times `height` times `num_channels`.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1)
    ///
    /// # Pinned transfer
    ///
    /// The other buffer (of type [`HostBuffer`]) is always a pinned buffer. This function is
    /// guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `other` - Buffer to copy to.
    /// * `stream` - Stream to use.
    #[inline]
    pub async fn copy_to(&self, other: &mut HostBuffer<T>, stream: &Stream) -> Result<()> {
        // SAFETY: Stream is synchronized after this.
        unsafe {
            self.copy_to_async(other, stream).await?;
        }
        stream.synchronize().await?;
        Ok(())
    }

    /// Copies memory from this 2D buffer to the provided pinned host buffer.
    ///
    /// The host buffer must be of the same size. For the 2D buffer, the total number of elements is
    /// `width` times `height` times `num_channels`.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1)
    ///
    /// # Pinned transfer
    ///
    /// The other buffer (of type [`HostBuffer`]) is always a pinned buffer. This function is
    /// guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the operation might not have completed when the function
    /// returns, and thus the state of the buffer is undefined.
    ///
    /// # Arguments
    ///
    /// * `other` - Buffer to copy to.
    /// * `stream` - Stream to use.
    pub async unsafe fn copy_to_async(
        &self,
        other: &mut HostBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(self.num_elements(), other.num_elements());
        Future::new(move || self.inner.copy_to_async(other.inner_mut(), stream.inner())).await
    }

    /// Fill the entire buffer with the given byte.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8fdcc53996ff49c570f4b5ead0256ef0)
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `value` - Byte value to fill buffer with.
    pub async fn fill_with_byte(&mut self, value: u8, stream: &Stream) -> Result<()> {
        Future::new(move || self.inner.fill_with_byte(value, stream.inner())).await
    }

    /// Get 2D buffer width.
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    /// Get 2D buffer height.
    #[inline(always)]
    pub fn height(&self) -> usize {
        self.inner.height
    }

    /// Get 2D buffer number of channels.
    #[inline(always)]
    pub fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    /// Get the total number of elements in buffer.
    ///
    /// This is equal to: `width` times `height` times `num_channels`.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    /// Access the inner synchronous implementation of [`DeviceBuffer2D`].
    #[inline(always)]
    pub fn inner(&self) -> &ffi::memory::DeviceBuffer2D<T> {
        &self.inner
    }

    /// Access the inner synchronous implementation of [`DeviceBuffer2D`].
    #[inline(always)]
    pub fn inner_mut(&mut self) -> &mut ffi::memory::DeviceBuffer2D<T> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new() {
        let buffer = DeviceBuffer2D::<u32>::new(120, 80, 3).await;
        assert_eq!(buffer.width(), 120);
        assert_eq!(buffer.height(), 80);
        assert_eq!(buffer.num_channels(), 3);
        assert_eq!(buffer.num_elements(), 120 * 80 * 3);
        assert!(buffer.inner().pitch >= 360);
    }

    #[tokio::test]
    async fn test_copy() {
        let stream = Stream::new().await.unwrap();
        let all_ones = vec![1_u32; 150];
        let host_buffer_all_ones = HostBuffer::from_slice(all_ones.as_slice()).await;

        let mut device_buffer = DeviceBuffer2D::<u32>::new(10, 5, 3).await;
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer_all_ones, &stream)
                .await
                .unwrap();
        }

        let mut host_buffer = HostBuffer::<u32>::new(150).await;
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .await
                .unwrap();
        }

        let mut another_device_buffer = DeviceBuffer2D::<u32>::new(10, 5, 3).await;
        unsafe {
            another_device_buffer
                .copy_from_async(&host_buffer, &stream)
                .await
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(150).await;
        unsafe {
            another_device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .await
                .unwrap();
        }

        stream.synchronize().await.unwrap();

        assert_eq!(return_host_buffer.num_elements(), 150);
        let return_data = return_host_buffer.to_vec();
        assert_eq!(return_data.len(), 150);
        assert!(return_data.into_iter().all(|v| v == 1_u32));
    }

    #[tokio::test]
    async fn test_copy_2d() {
        let stream = Stream::new().await.unwrap();
        let image: [u8; 12] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4];
        let host_buffer = HostBuffer::from_slice(&image).await;
        let mut device_buffer = DeviceBuffer2D::<u8>::new(2, 2, 3).await;
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer, &stream)
                .await
                .unwrap();
        }
        let mut return_host_buffer = HostBuffer::<u8>::new(12).await;
        unsafe {
            device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .await
                .unwrap();
        }
        stream.synchronize().await.unwrap();
        assert_eq!(
            &return_host_buffer.to_vec(),
            &[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        );
    }

    #[tokio::test]
    async fn test_fill_with_byte() {
        let stream = Stream::new().await.unwrap();
        let mut device_buffer = DeviceBuffer2D::<u8>::new(2, 2, 3).await;
        let mut host_buffer = HostBuffer::<u8>::new(2 * 2 * 3).await;
        device_buffer.fill_with_byte(0xab, &stream).await.unwrap();
        device_buffer
            .copy_to(&mut host_buffer, &stream)
            .await
            .unwrap();
        assert_eq!(
            host_buffer.to_vec(),
            &[0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab]
        );
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().await.unwrap();
        let device_buffer = DeviceBuffer2D::<u32>::new(5, 5, 3).await;
        let mut host_buffer = HostBuffer::<u32>::new(80).await;
        let _ = unsafe { device_buffer.copy_to_async(&mut host_buffer, &stream).await };
    }
}
