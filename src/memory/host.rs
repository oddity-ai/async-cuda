use crate::ffi;
use crate::memory::DeviceBuffer;
use crate::runtime::Future;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// A host buffer.
///
/// # Performance
///
/// Host buffers are managed by CUDA and can be used for pinned memory transfer. Pinned memory
/// transfer speeds are usually higher compared to paged memory transfers. Pinned memory buffers are
/// especially important for this crate because the runtime thread must do the least amount of CPU
/// work possible. Paged transfers do require the host to move data into a CUDA managed buffer first
/// (an extra memory copy) whilst pinned transfers do not.
pub struct HostBuffer<T: Copy + 'static> {
    inner: ffi::memory::HostBuffer<T>,
}

impl<T: Copy + 'static> HostBuffer<T> {
    /// Allocates memory on the host. This creates a pinned buffer. Any transfers to and from this
    /// buffer automatically become pinned transfers, and will be much faster.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c)
    ///
    /// # Arguments
    ///
    /// * `num_elements` - Number of elements to allocate.
    pub async fn new(num_elements: usize) -> Self {
        let inner = Future::new(move || ffi::memory::HostBuffer::<T>::new(num_elements)).await;
        Self { inner }
    }

    /// Allocates memory on the host and copies the provided data into it.
    ///
    /// This creates a pinned buffer. Any transfers to and from this buffer automatically become
    /// pinned transfers, and will be much faster.
    ///
    /// This is a convenience function that allows the caller to quickly put data into a host
    /// buffer. It is roughly similar to `buffer.copy_from_slice(slice)`.
    ///
    /// # Arguments
    ///
    /// * `slice` - Data to copy into the new host buffer.
    pub async fn from_slice(slice: &[T]) -> Self {
        let mut this = Self::new(slice.len()).await;
        this.copy_from_slice(slice);
        this
    }

    /// Allocates memory on the host and copies the provided array into it.
    ///
    /// This creates a pinned buffer. Any transfers to and from this buffer automatically become
    /// pinned transfers, and will be much faster.
    ///
    /// This is a convenience function that allows the caller to quickly put data into a host
    /// buffer. It is roughly similar to `buffer.copy_from_array(slice)`.
    ///
    /// # Arguments
    ///
    /// * `array` - Array to copy into the new host buffer.
    #[cfg(feature = "ndarray")]
    pub async fn from_array<D: ndarray::Dimension>(array: &ndarray::ArrayView<'_, T, D>) -> Self {
        let mut this = Self::new(array.len()).await;
        this.copy_from_array(array);
        this
    }

    /// Copies memory from the provided device buffer to this buffer.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
    ///
    /// # Pinned transfer
    ///
    /// This function is guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `other` - Device buffer to copy from.
    /// * `stream` - Stream to use.
    #[inline(always)]
    pub async fn copy_from(&mut self, other: &DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        other.copy_to(self, stream).await
    }

    /// Copies memory from the provided device buffer to this buffer.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
    ///
    /// # Pinned transfer
    ///
    /// This function is guaranteed to produce a pinned transfer on the runtime thread.
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
    /// * `other` - Device buffer to copy from.
    /// * `stream` - Stream to use.
    #[inline(always)]
    pub async unsafe fn copy_from_async(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        other.copy_to_async(self, stream).await
    }

    /// Copies memory from this buffer to the provided device buffer.
    ///
    /// This function synchronizes the stream implicitly.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
    ///
    /// # Pinned transfer
    ///
    /// This function is guaranteed to produce a pinned transfer on the runtime thread.
    ///
    /// # Stream ordered semantics
    ///
    /// This function uses stream ordered semantics. It can only be guaranteed to complete
    /// sequentially relative to operations scheduled on the same stream or the default stream.
    ///
    /// # Arguments
    ///
    /// * `other` - Device buffer to copy to.
    /// * `stream` - Stream to use.
    #[inline(always)]
    pub async fn copy_to(&self, other: &mut DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        other.copy_from(self, stream).await
    }

    /// Copies memory from this buffer to the provided device buffer.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79)
    ///
    /// # Pinned transfer
    ///
    /// This function is guaranteed to produce a pinned transfer on the runtime thread.
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
    /// * `other` - Device buffer to copy to.
    /// * `stream` - Stream to use.
    #[inline(always)]
    pub async unsafe fn copy_to_async(
        &self,
        other: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        other.copy_from_async(self, stream).await
    }

    /// Copy data into the host buffer from a slice.
    ///
    /// # Synchronization safety
    ///
    /// This call is only synchronization-safe if all streams that have previously been used for
    /// copy operations either from or to this host buffer have been synchronized, and no operations
    /// have been scheduled since.
    ///
    /// # Arguments
    ///
    /// * `slice` - Data to copy into the new host buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use async_cuda::HostBuffer;
    /// # tokio_test::block_on(async {
    /// let mut host_buffer = HostBuffer::<u8>::new(100).await;
    /// let some_data = vec![10; 100];
    /// host_buffer.copy_from_slice(&some_data);
    /// # })
    /// ```
    #[inline(always)]
    pub fn copy_from_slice(&mut self, slice: &[T]) {
        self.inner.copy_from_slice(slice);
    }

    /// Copy array into the host buffer from a slice.
    ///
    /// # Synchronization safety
    ///
    /// This call is only synchronization-safe if all streams that have previously been used for
    /// copy operations either from or to this host buffer have been synchronized, and no operations
    /// have been scheduled since.
    ///
    /// # Arguments
    ///
    /// * `array` - Array to copy into the new host buffer.
    #[cfg(feature = "ndarray")]
    #[inline(always)]
    pub fn copy_from_array<D: ndarray::Dimension>(&mut self, array: &ndarray::ArrayView<T, D>) {
        self.inner.copy_from_array(array)
    }

    /// Copy the data to a [`Vec`] and return it.
    #[inline(always)]
    pub fn to_vec(&self) -> Vec<T> {
        self.inner.to_vec()
    }

    /// Copy the data to an [`ndarray::Array`] and return it.
    ///
    /// Function panics if provided shape does not match size of array.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape for array.
    #[cfg(feature = "ndarray")]
    #[inline(always)]
    pub fn to_array_with_shape<D: ndarray::Dimension>(
        &self,
        shape: impl Into<ndarray::StrideShape<D>>,
    ) -> ndarray::Array<T, D> {
        self.inner.to_array_with_shape::<D>(shape)
    }

    /// Get number of elements in buffer.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.inner.num_elements
    }

    /// Access the inner synchronous implementation of [`HostBuffer`].
    #[inline(always)]
    pub fn inner(&self) -> &ffi::memory::HostBuffer<T> {
        &self.inner
    }

    /// Access the inner synchronous implementation of [`HostBuffer`].
    #[inline(always)]
    pub fn inner_mut(&mut self) -> &mut ffi::memory::HostBuffer<T> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new() {
        let buffer = HostBuffer::<u32>::new(100).await;
        assert_eq!(buffer.num_elements(), 100);
        assert_eq!(buffer.to_vec().len(), 100);
    }

    #[tokio::test]
    async fn test_from_slice() {
        let all_ones = vec![1_u32; 200];
        let buffer = HostBuffer::from_slice(all_ones.as_slice()).await;
        assert_eq!(buffer.num_elements(), 200);
        let data = buffer.to_vec();
        assert_eq!(data.len(), 200);
        assert!(data.into_iter().all(|v| v == 1_u32));
    }

    #[tokio::test]
    async fn test_copy() {
        let stream = Stream::new().await.unwrap();
        let all_ones = vec![1_u32; 100];
        let host_buffer = HostBuffer::from_slice(all_ones.as_slice()).await;

        let mut device_buffer = DeviceBuffer::<u32>::new(100, &stream).await;
        unsafe {
            host_buffer
                .copy_to_async(&mut device_buffer, &stream)
                .await
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(100).await;
        unsafe {
            return_host_buffer
                .copy_from_async(&device_buffer, &stream)
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
    #[should_panic]
    async fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().await.unwrap();
        let host_buffer = HostBuffer::<u32>::new(100).await;
        let mut device_buffer = DeviceBuffer::<u32>::new(101, &Stream::null()).await;
        let _ = unsafe { host_buffer.copy_to_async(&mut device_buffer, &stream).await };
    }
}
