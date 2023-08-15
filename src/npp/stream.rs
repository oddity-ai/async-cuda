use std::sync::Arc;

use crate::ffi::npp::context::Context;
use crate::runtime::Future;

/// Represents an NPP stream.
///
/// An NPP stream is a thin wrapper around a normal CUDA stream ([`crate::Stream`]). It manages some
/// additional context information required in NPP to statelessly execute on a user-provided stream.
///
/// This struct implements `Deref` such that it can be used as a normal [`crate::Stream`] as well.
///
/// # Usage
///
/// If the caller wants to use a stream context for mixed NPP and non-NPP operations, they should
/// create an NPP stream and pass it as CUDA stream when desired. This should work out-of-the-box
/// since [`Stream`] dereferences to [`crate::Stream`].
pub struct Stream {
    context: Arc<Context>,
}

impl Stream {
    /// Create an NPP [`Stream`] that represent the default stream, also known as the null stream.
    ///
    /// This type is a wrapper around the actual CUDA stream type: [`crate::Stream`].
    #[inline]
    pub async fn null() -> Self {
        let context = Future::new(Context::from_null_stream).await;
        Self {
            context: Arc::new(context),
        }
    }

    /// Create a new [`Stream`] for use with NPP.
    ///
    /// This type is a wrapper around the actual CUDA stream type: [`crate::Stream`].
    #[inline]
    pub async fn new() -> std::result::Result<Self, crate::Error> {
        let stream = crate::Stream::new().await?;
        let context = Future::new(move || Context::from_stream(stream)).await;
        Ok(Self {
            context: Arc::new(context),
        })
    }

    /// Acquire shared access to the underlying NPP context object.
    ///
    /// This NPP object can be safetly sent to the runtime thread so it can be used as a context.
    ///
    /// # Safety
    ///
    /// The [`Context`] object may only be *used* from the runtime thread.
    pub(crate) fn to_context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

impl std::ops::Deref for Stream {
    type Target = crate::Stream;

    fn deref(&self) -> &Self::Target {
        &self.context.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new() {
        let stream = Stream::new().await.unwrap();
        assert!(!stream.to_context().as_ptr().is_null());
        // SAFETY: This works because we know that the first field of the underlying
        // `NppStreamContext` struct used internally is `hStream`, which should refer to the wrapped
        // stream or it was not initalized correctly.
        assert_eq!(
            unsafe { *(stream.to_context().as_ptr() as *const *const std::ffi::c_void) },
            stream.inner().as_internal().as_ptr(),
        );
    }

    #[tokio::test]
    async fn test_null() {
        let stream = Stream::null().await;
        assert!(!stream.to_context().as_ptr().is_null());
        // SAFETY: This works because we know that the first field of the underlying
        // `NppStreamContext` struct used internally is `hStream`, which should refer to the wrapped
        // stream, which is the null stream in this case.
        assert!(
            unsafe { *(stream.to_context().as_ptr() as *const *const std::ffi::c_void) }.is_null()
        );
    }
}
