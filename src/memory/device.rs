use crate::ffi;
use crate::memory::HostBuffer;
use crate::runtime::Future;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// A buffer on the device.
///
/// # Example
///
/// Copying data from a [`HostBuffer`] to a [`DeviceBuffer`]:
///
/// ```
/// # use async_cuda::{DeviceBuffer, HostBuffer, Stream};
/// # tokio_test::block_on(async {
/// let stream = Stream::new().await.unwrap();
/// let all_ones = vec![1_u8; 100];
/// let host_buffer = HostBuffer::<u8>::from_slice(&all_ones).await;
/// let mut device_buffer = DeviceBuffer::<u8>::new(100, &stream).await;
/// device_buffer.copy_from(&host_buffer, &stream).await.unwrap();
/// # })
/// ```
pub struct DeviceBuffer<T: Copy + 'static> {
    inner: ffi::memory::DeviceBuffer<T>,
}

impl<T: Copy + 'static> DeviceBuffer<T> {
    /// Allocates memory on the device.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS_1gbbf70065888d61853c047513baa14081)
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `num_elements` - Number of elements to allocate.
    /// * `stream` - Stream to use.
    pub async fn new(num_elements: usize, stream: &Stream) -> Self {
        let inner =
            Future::new(move || ffi::memory::DeviceBuffer::<T>::new(num_elements, stream.inner()))
                .await;
        Self { inner }
    }

    /// Allocate memory on the device, and copy data from host into it.
    ///
    /// This function creates a temporary [`HostBuffer`], copies the slice into it, then finally
    /// copies the data from the host buffer to the [`DeviceBuffer`].
    ///
    /// The given stream is automatically synchronized, since the temporary host buffer might
    /// otherwise be dropped before the copy can complete.
    ///
    /// # Arguments
    ///
    /// * `slice` - Data to copy into the buffer.
    /// * `stream` - Stream to use.
    pub async fn from_slice(slice: &[T], stream: &Stream) -> Result<Self> {
        let host_buffer = HostBuffer::from_slice(slice).await;
        let mut this = Self::new(slice.len(), stream).await;
        this.copy_from(&host_buffer, stream).await?;
        Ok(this)
    }

    /// Allocate memory on the device, and copy array from host into it.
    ///
    /// This function creates a temporary [`HostBuffer`], copies the slice into it, then finally
    /// copies the data from the host buffer to the [`DeviceBuffer`].
    ///
    /// The given stream is automatically synchronized, since the temporary host buffer might
    /// otherwise be dropped before the copy can complete.
    ///
    /// # Arguments
    ///
    /// * `slice` - Data to copy into the buffer.
    /// * `stream` - Stream to use.
    #[cfg(feature = "ndarray")]
    pub async fn from_array<D: ndarray::Dimension>(
        array: &ndarray::ArrayView<'_, T, D>,
        stream: &Stream,
    ) -> Result<Self> {
        let host_buffer = HostBuffer::from_array(array).await;
        let mut this = Self::new(array.len(), stream).await;
        this.copy_from(&host_buffer, stream).await?;
        Ok(this)
    }

    /// Copies memory from the provided pinned host buffer to this buffer.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
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

    /// Copies memory from the provided pinned host buffer to this buffer.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
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

    /// Copies memory from this buffer to the provided pinned host buffer.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
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

    /// Copies memory from this buffer to the provided pinned host buffer.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
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
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7c9761e21d9f0999fd136c51e7b9b2a0)
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

    /// Get number of elements in buffer.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.inner.num_elements
    }

    /// Access the inner synchronous implementation of [`DeviceBuffer`].
    #[inline(always)]
    pub fn inner(&self) -> &ffi::memory::DeviceBuffer<T> {
        &self.inner
    }

    /// Access the inner synchronous implementation of [`DeviceBuffer`].
    #[inline(always)]
    pub fn inner_mut(&mut self) -> &mut ffi::memory::DeviceBuffer<T> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new() {
        let buffer = DeviceBuffer::<u32>::new(100, &Stream::null()).await;
        assert_eq!(buffer.num_elements(), 100);
    }

    #[tokio::test]
    async fn test_copy() {
        let stream = Stream::new().await.unwrap();
        let all_ones = vec![1_u32; 100];
        let host_buffer_all_ones = HostBuffer::from_slice(all_ones.as_slice()).await;

        let mut device_buffer = DeviceBuffer::<u32>::new(100, &stream).await;
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer_all_ones, &stream)
                .await
                .unwrap();
        }

        let mut host_buffer = HostBuffer::<u32>::new(100).await;
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .await
                .unwrap();
        }

        let mut another_device_buffer = DeviceBuffer::<u32>::new(100, &stream).await;
        unsafe {
            another_device_buffer
                .copy_from_async(&host_buffer, &stream)
                .await
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(100).await;
        unsafe {
            another_device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .await
                .unwrap();
        }

        stream.synchronize().await.unwrap();

        assert_eq!(return_host_buffer.num_elements(), 100);
        let return_data = return_host_buffer.to_vec();
        assert_eq!(return_data.len(), 100);
        assert!(return_data.into_iter().all(|v| v == 1_u32));
    }

    #[tokio::test]
    async fn test_fill_with_byte() {
        let stream = Stream::new().await.unwrap();
        let mut device_buffer = DeviceBuffer::<u8>::new(4, &stream).await;
        let mut host_buffer = HostBuffer::<u8>::new(4).await;
        device_buffer.fill_with_byte(0xab, &stream).await.unwrap();
        device_buffer
            .copy_to(&mut host_buffer, &stream)
            .await
            .unwrap();
        assert_eq!(host_buffer.to_vec(), &[0xab, 0xab, 0xab, 0xab]);
    }

    #[tokio::test]
    #[should_panic]
    async fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().await.unwrap();
        let device_buffer = DeviceBuffer::<u32>::new(101, &stream).await;
        let mut host_buffer = HostBuffer::<u32>::new(100).await;
        let _ = unsafe { device_buffer.copy_to_async(&mut host_buffer, &stream).await };
    }
}
