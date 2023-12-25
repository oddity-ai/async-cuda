use crate::ffi;
use crate::runtime::{Future, SynchronizeFuture};
use crate::device::DeviceId;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// CUDA stream.
pub struct Stream {
    inner: ffi::stream::Stream,
}

impl Stream {
    /// Create a [`Stream`] object that represent the default stream, also known as the null stream.
    ///
    /// Refer to the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
    /// for more information regarding the default ("null") stream:
    ///
    /// # Prefer owned streams
    ///
    /// It is recommended to use owned streams as much as possible, for two reasons:
    ///
    /// * Using streams to separate semanticly unrelated streams of operations allows the GPU to
    ///   overlap operations and improved parallelism.
    /// * Using the default stream can incur implicit synchronization, even on other streams, which
    ///   causes their performance to degrade.
    ///
    /// Note that it is not enforced that there is only one [`Stream`] object that represents the
    /// default stream. This is safe because all operations are serialized anyway.
    pub fn null() -> Self {
        Self {
            inner: ffi::stream::Stream::null(),
        }
    }

    /// Create an asynchronous stream.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da)
    pub async fn new() -> Result<Self> {
        let inner = Future::new(ffi::stream::Stream::new).await?;
        Ok(Self { inner })
    }

    /// Create an asynchronous stream with a given device.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da)
    pub async fn with_device(device: DeviceId) -> Result<Self> {
        let inner = Future::new(|| {
            ffi::device::Device::set_or_panic(device);
            ffi::stream::Stream::new()
        }).await?;
        Ok(Self { inner })
    }

    /// Synchronize stream. This future will only return once all currently enqueued work on the
    /// stream is done.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498)
    ///
    /// # Behavior
    ///
    /// In constrast to most of the API, this future does not become ready eagerly. Instead, a
    /// callback is pushed onto the given stream that will be invoked to make the future ready once
    /// all work on the stream that was previously queued asynchroneously is completed.
    ///
    /// Internally, the future uses `cudaStreamAddCallback` to schedule the callback on the stream.
    pub async fn synchronize(&self) -> Result<()> {
        SynchronizeFuture::new(self).await
    }

    /// Access the inner synchronous implementation of [`Stream`].
    #[inline(always)]
    pub fn inner(&self) -> &ffi::stream::Stream {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new() {
        assert!(Stream::new().await.is_ok());
    }

    #[tokio::test]
    async fn test_synchronize() {
        let stream = Stream::new().await.unwrap();
        assert!(stream.synchronize().await.is_ok());
    }

    #[tokio::test]
    async fn test_synchronize_null_stream() {
        let stream = Stream::null();
        assert!(stream.synchronize().await.is_ok());
    }
}
