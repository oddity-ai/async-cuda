use cpp::cpp;

use crate::npp::ffi::result;
use crate::stream::Stream;

/// NPP stream context structure.
///
/// [NPP documentation](https://docs.nvidia.com/cuda/npp/struct_npp_stream_context.html)
pub struct Context {
    raw: *mut std::ffi::c_void,
    pub stream: Stream,
}

/// Implements [`Send`] for [`Context`].
///
/// # Safety
///
/// This is safe because the way we use the underlying `NppStreamContext` object is thread-safe.
unsafe impl Send for Context {}

/// Implements [`Sync`] for [`Context`].
///
/// # Safety
///
/// This is safe because the way we use the underlying `NppStreamContext` object is thread-safe.
unsafe impl Sync for Context {}

impl Context {
    /// Create context on null stream.
    ///
    /// This creates a context that can be passed to NPP functions. Any functions using this context
    /// will be executed on the null stream.
    pub fn from_null_stream() -> Self {
        let mut raw = std::ptr::null_mut();
        let raw_ptr = std::ptr::addr_of_mut!(raw);
        // SAFETY:
        // * Must call this function on runtime since `nppGetStreamContext` needs the correct thread
        //   locals to determine current device and other context settings.
        // * We can store a reference to the stream in `NppStreamContext` as long as we make sure
        //   `NppStreamContext` cannot outlive the stream, which we can guarantee because we take
        //   ownership of the stream.
        let ret = cpp!(unsafe [
            raw_ptr as "void**"
        ] -> i32 as "std::int32_t" {
            NppStreamContext* stream_context = new NppStreamContext();
            NppStatus ret = nppGetStreamContext(stream_context);
            if (ret == NPP_SUCCESS) {
                stream_context->hStream = nullptr;
                *raw_ptr = (void*) stream_context;
            }
            return ret;
        });
        match result!(ret) {
            Ok(()) => Self {
                raw,
                stream: Stream::null(),
            },
            Err(err) => {
                panic!("failed to get current NPP stream context: {err}")
            }
        }
    }

    /// Create context.
    ///
    /// This creates an NPP context object. It can be passed to NPP functions, and they will execute
    /// on the associated stream.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream to associate with context.
    pub fn from_stream(stream: Stream) -> Self {
        let (ret, raw) = {
            let mut raw = std::ptr::null_mut();
            let raw_ptr = std::ptr::addr_of_mut!(raw);
            let stream_ptr = stream.inner().as_internal().as_ptr();
            // SAFETY:
            // * Must call this function on runtime since `nppGetStreamContext` needs the correct
            //   thread locals to determine current device and other context settings.
            // * We can store a reference to the stream in `NppStreamContext` as long as we make
            //   sure `NppStreamContext` cannot outlive the stream, which we can guarantee because
            //   we take ownership of the stream.
            let ret = cpp!(unsafe [
                raw_ptr as "void**",
                stream_ptr as "void*"
            ] -> i32 as "std::int32_t" {
                NppStreamContext* stream_context = new NppStreamContext();
                NppStatus ret = nppGetStreamContext(stream_context);
                if (ret == NPP_SUCCESS) {
                    stream_context->hStream = (cudaStream_t) stream_ptr;
                    *raw_ptr = (void*) stream_context;
                }
                return ret;
            });
            (ret, raw)
        };
        match result!(ret) {
            Ok(()) => Self { raw, stream },
            Err(err) => {
                panic!("failed to get current NPP stream context: {err}")
            }
        }
    }

    /// Get internal readonly pointer.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_void {
        self.raw
    }

    /// Delete the context.
    ///
    /// # Safety
    ///
    /// The context may not be used after this function is called, except for being dropped.
    pub unsafe fn delete(&mut self) {
        if self.raw.is_null() {
            return;
        }

        let raw = self.raw;
        self.raw = std::ptr::null_mut();

        cpp!(unsafe [raw as "void*"] {
            delete ((NppStreamContext*) raw);
        });
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe since the buffer cannot be used after this.
        unsafe {
            self.delete();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_from_stream() {
        let stream = Stream::new().await.unwrap();
        let context = Context::from_stream(stream);
        assert!(!context.as_ptr().is_null());
        assert!(!context.stream.inner().as_internal().as_ptr().is_null());
    }

    #[test]
    fn test_from_null_stream() {
        let context = Context::from_null_stream();
        assert!(!context.as_ptr().is_null());
        assert!(context.stream.inner().as_internal().as_ptr().is_null());
    }
}
