use cpp::cpp;

use crate::ffi::ptr::DevicePtr;
use crate::ffi::result;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::Stream`].
///
/// Refer to [`crate::Stream`] for documentation.
pub struct Stream {
    internal: DevicePtr,
}

/// Implements [`Send`] for [`Stream`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl Send for Stream {}

/// Implements [`Sync`] for [`Stream`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl Sync for Stream {}

impl Stream {
    pub fn null() -> Self {
        Self {
            // SAFETY: This is safe because a null pointer for stream indicates the default
            // stream in CUDA and all functions accept this.
            internal: unsafe { DevicePtr::null() },
        }
    }

    pub fn new() -> Result<Self> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ptr_ptr = std::ptr::addr_of_mut!(ptr);

        let ret = cpp!(unsafe [
            ptr_ptr as "void**"
        ] -> i32 as "std::int32_t" {
            return cudaStreamCreate((cudaStream_t*) ptr_ptr);
        });
        result!(
            ret,
            Stream {
                internal: ptr.into()
            }
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        let ptr = self.internal.as_ptr();
        let ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaStreamSynchronize((cudaStream_t) ptr);
        });
        result!(ret)
    }

    pub fn add_callback(&self, f: impl FnOnce() + Send) -> Result<()> {
        let ptr = self.internal.as_ptr();

        let f_boxed = Box::new(f) as Box<dyn FnOnce()>;
        let f_boxed2 = Box::new(f_boxed);
        let f_boxed2_ptr = Box::into_raw(f_boxed2);
        let user_data = f_boxed2_ptr as *mut std::ffi::c_void;

        let ret = cpp!(unsafe [
            ptr as "void*",
            user_data as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaStreamAddCallback(
                (cudaStream_t) ptr,
                cuda_ffi_Callback,
                user_data,
                0
            );
        });

        result!(ret)
    }

    /// Get readonly reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_internal(&self) -> &DevicePtr {
        &self.internal
    }

    /// Get mutable reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_mut_internal(&mut self) -> &mut DevicePtr {
        &mut self.internal
    }

    /// Destroy stream.
    ///
    /// # Safety
    ///
    /// The object may not be used after this function is called, except for being dropped.
    pub unsafe fn destroy(&mut self) {
        if self.internal.is_null() {
            return;
        }

        // SAFETY: This will cause `self` to hold a null pointer. It is safe here because we don't
        // use the object after this.
        let mut internal = unsafe { self.internal.take() };
        let ptr = internal.as_mut_ptr();

        // SAFETY: We must synchronize the stream before destroying it to make sure we are not
        // dropping a stream that still has operations pending.
        let _ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaStreamSynchronize((cudaStream_t) ptr);
        });

        let _ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaStreamDestroy((cudaStream_t) ptr);
        });
    }
}

impl Drop for Stream {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe since the object cannot be used after this.
        unsafe {
            self.destroy();
        }
    }
}

cpp! {{
    /// Holds the C++ code that makes up the native part required to get our CUDA callback to work
    /// over the FFI.
    ///
    /// # Arguments
    ///
    /// * `stream` - The CUDA stream on which the callback was scheduled.
    /// * `status` - The CUDA status value (this could represent an error from an earlier async CUDA
    ///   call).
    /// * `user_data` - The user data pointer provided when adding the callback.
    ///
    /// # Example
    ///
    /// It can be used like so:
    ///
    /// ```cpp
    /// return cudaStreamAddCallback(
    ///     stream,
    ///     cuda_ffi_Callback,
    ///     user_data,
    ///     0
    /// );
    /// ```
    static void cuda_ffi_Callback(
      __attribute__((unused)) cudaStream_t stream,
      cudaError_t status,
      void* user_data
    ) {
        rust!(cuda_ffi_Callback_internal [
            status : i32 as "std::int32_t",
            user_data : *mut std::ffi::c_void as "void*"
        ] {
            // SAFETY: We boxed the closure ourselves and did `Box::into_raw`, which allows us to
            // reinstate the box here and use it accordingly. It will be dropped here after use.
            unsafe {
                let user_data = std::mem::transmute(user_data);
                let function = Box::<Box<dyn FnOnce()>>::from_raw(user_data);
                function()
            }
        });
    }
}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        assert!(Stream::new().is_ok());
    }

    #[test]
    fn test_synchronize() {
        let stream = Stream::new().unwrap();
        assert!(stream.synchronize().is_ok());
    }

    #[test]
    fn test_synchronize_null_stream() {
        let stream = Stream::null();
        assert!(stream.synchronize().is_ok());
    }
}
