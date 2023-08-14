use cpp::cpp;

use crate::ffi::memory::host::HostBuffer;
use crate::ffi::ptr::DevicePtr;
use crate::ffi::result;
use crate::ffi::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::DeviceBuffer`].
///
/// Refer to [`crate::DeviceBuffer`] for documentation.
pub struct DeviceBuffer<T: Copy> {
    pub num_elements: usize,
    internal: DevicePtr,
    _phantom: std::marker::PhantomData<T>,
}

/// Implements [`Send`] for [`DeviceBuffer`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Send for DeviceBuffer<T> {}

/// Implements [`Sync`] for [`DeviceBuffer`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn new(num_elements: usize, stream: &Stream) -> Self {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ptr_ptr = std::ptr::addr_of_mut!(ptr);
        let size = num_elements * std::mem::size_of::<T>();
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr_ptr as "void**",
            size as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMallocAsync(ptr_ptr, size, (cudaStream_t) stream_ptr);
        });
        match result!(ret, ptr.into()) {
            Ok(internal) => Self {
                internal,
                num_elements,
                _phantom: Default::default(),
            },
            Err(err) => {
                panic!("failed to allocate device memory: {err}");
            }
        }
    }

    pub fn from_slice(slice: &[T], stream: &Stream) -> Result<Self> {
        let host_buffer = HostBuffer::from_slice(slice);
        let mut this = Self::new(slice.len(), stream);
        // SAFETY: Safe because the stream is synchronized after this.
        unsafe {
            this.copy_from_async(&host_buffer, stream)?;
        }
        stream.synchronize()?;
        Ok(this)
    }

    #[cfg(feature = "ndarray")]
    pub fn from_array<D: ndarray::Dimension>(
        array: &ndarray::ArrayView<T, D>,
        stream: &Stream,
    ) -> Result<Self> {
        let host_buffer = HostBuffer::from_array(array);
        let mut this = Self::new(array.len(), stream);
        // SAFETY: Safe because the stream is synchronized after this.
        unsafe {
            this.copy_from_async(&host_buffer, stream)?;
        }
        stream.synchronize()?;
        Ok(this)
    }

    /// Copy from host buffer.
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns.
    pub unsafe fn copy_from_async(&mut self, other: &HostBuffer<T>, stream: &Stream) -> Result<()> {
        assert_eq!(self.num_elements, other.num_elements);
        let ptr_to = self.as_mut_internal().as_mut_ptr();
        let ptr_from = other.as_internal().as_ptr();
        let stream_ptr = stream.as_internal().as_ptr();
        let size = self.num_elements * std::mem::size_of::<T>();
        let ret = cpp!(unsafe [
            ptr_from as "void*",
            ptr_to as "void*",
            size as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpyAsync(
                ptr_to,
                ptr_from,
                size,
                cudaMemcpyHostToDevice,
                (cudaStream_t) stream_ptr
            );
        });
        result!(ret)
    }

    /// Copy to host buffer.
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns.
    pub unsafe fn copy_to_async(&self, other: &mut HostBuffer<T>, stream: &Stream) -> Result<()> {
        assert_eq!(self.num_elements, other.num_elements);
        let ptr_from = self.as_internal().as_ptr();
        let ptr_to = other.as_mut_internal().as_mut_ptr();
        let size = self.num_elements * std::mem::size_of::<T>();
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr_from as "void*",
            ptr_to as "void*",
            size as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpyAsync(
                ptr_to,
                ptr_from,
                size,
                cudaMemcpyDeviceToHost,
                (cudaStream_t) stream_ptr
            );
        });
        result!(ret)
    }

    /// Fill buffer with byte value.
    pub fn fill_with_byte(&mut self, value: u8, stream: &Stream) -> Result<()> {
        let ptr = self.as_internal().as_ptr();
        let value = value as std::ffi::c_int;
        let size = self.num_elements * std::mem::size_of::<T>();
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr as "void*",
            value as "int",
            size as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemsetAsync(
                ptr,
                value,
                size,
                (cudaStream_t) stream_ptr
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

    /// Release the buffer memory.
    ///
    /// # Safety
    ///
    /// The buffer may not be used after this function is called, except for being dropped.
    pub unsafe fn free(&mut self) {
        if self.internal.is_null() {
            return;
        }

        // SAFETY: Safe because we won't use pointer after this.
        let mut internal = unsafe { self.internal.take() };
        let ptr = internal.as_mut_ptr();
        let _ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaFree(ptr);
        });
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe since the buffer cannot be used after this.
        unsafe {
            self.free();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buffer = DeviceBuffer::<u32>::new(100, &Stream::null());
        assert_eq!(buffer.num_elements, 100);
    }

    #[test]
    fn test_copy() {
        let stream = Stream::new().unwrap();
        let all_ones = vec![1_u32; 100];
        let host_buffer_all_ones = HostBuffer::from_slice(all_ones.as_slice());

        let mut device_buffer = DeviceBuffer::<u32>::new(100, &stream);
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer_all_ones, &stream)
                .unwrap();
        }

        let mut host_buffer = HostBuffer::<u32>::new(100);
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .unwrap();
        }

        let mut another_device_buffer = DeviceBuffer::<u32>::new(100, &stream);
        unsafe {
            another_device_buffer
                .copy_from_async(&host_buffer, &stream)
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(100);
        unsafe {
            another_device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .unwrap();
        }

        stream.synchronize().unwrap();

        assert_eq!(return_host_buffer.num_elements, 100);
        let return_data = return_host_buffer.to_vec();
        assert_eq!(return_data.len(), 100);
        assert!(return_data.into_iter().all(|v| v == 1_u32));
    }

    #[test]
    fn test_fill_with_byte() {
        let stream = Stream::new().unwrap();
        let mut device_buffer = DeviceBuffer::<u8>::new(4, &stream);
        let mut host_buffer = HostBuffer::<u8>::new(4);
        device_buffer.fill_with_byte(0xab, &stream).unwrap();
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(host_buffer.to_vec(), &[0xab, 0xab, 0xab, 0xab]);
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().unwrap();
        let device_buffer = DeviceBuffer::<u32>::new(101, &stream);
        let mut host_buffer = HostBuffer::<u32>::new(100);
        let _ = unsafe { device_buffer.copy_to_async(&mut host_buffer, &stream) };
    }
}
