use cpp::cpp;

use crate::ffi::memory::host::HostBuffer;
use crate::ffi::ptr::DevicePtr;
use crate::ffi::result;
use crate::ffi::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::DeviceBuffer2D`].
///
/// Refer to [`crate::DeviceBuffer2D`] for documentation.
pub struct DeviceBuffer2D<T: Copy> {
    pub width: usize,
    pub height: usize,
    pub num_channels: usize,
    pub pitch: usize,
    internal: DevicePtr,
    _phantom: std::marker::PhantomData<T>,
}

/// Implements [`Send`] for [`DeviceBuffer2D`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Send for DeviceBuffer2D<T> {}

/// Implements [`Sync`] for [`DeviceBuffer2D`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Sync for DeviceBuffer2D<T> {}

impl<T: Copy> DeviceBuffer2D<T> {
    pub fn new(width: usize, height: usize, num_channels: usize) -> Self {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ptr_ptr = std::ptr::addr_of_mut!(ptr);
        let mut pitch = 0_usize;
        let pitch_ptr = std::ptr::addr_of_mut!(pitch);
        let line_size = width * num_channels * std::mem::size_of::<T>();
        let ret = cpp!(unsafe [
            ptr_ptr as "void**",
            pitch_ptr as "std::size_t*",
            line_size as "std::size_t",
            height as "std::size_t"
        ] -> i32 as "std::int32_t" {
            return cudaMallocPitch(
                ptr_ptr,
                pitch_ptr,
                line_size,
                height
            );
        });
        match result!(ret, ptr.into()) {
            Ok(internal) => Self {
                width,
                height,
                num_channels,
                pitch,
                internal,
                _phantom: Default::default(),
            },
            Err(err) => {
                panic!("failed to allocate device memory: {err}");
            }
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn from_array(array: &ndarray::ArrayView3<T>, stream: &Stream) -> Result<Self> {
        let host_buffer = HostBuffer::from_array(array);
        let (height, width, num_channels) = array.dim();
        let mut this = Self::new(width, height, num_channels);
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
        assert_eq!(self.num_elements(), other.num_elements);
        let ptr_from = other.as_internal().as_ptr();
        let ptr_to = self.as_mut_internal().as_mut_ptr();
        let line_size = self.width * self.num_channels * std::mem::size_of::<T>();
        let height = self.height;
        let pitch = self.pitch;
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr_from as "void*",
            ptr_to as "void*",
            pitch as "std::size_t",
            line_size as "std::size_t",
            height as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpy2DAsync(
                ptr_to,
                pitch,
                ptr_from,
                line_size,
                line_size,
                height,
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
        assert_eq!(self.num_elements(), other.num_elements);
        let ptr_from = self.as_internal().as_ptr();
        let ptr_to = other.as_mut_internal().as_mut_ptr();
        let line_size = self.width * self.num_channels * std::mem::size_of::<T>();
        let height = self.height;
        let pitch = self.pitch;
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr_from as "void*",
            ptr_to as "void*",
            pitch as "std::size_t",
            line_size as "std::size_t",
            height as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpy2DAsync(
                ptr_to,
                line_size,
                ptr_from,
                pitch,
                line_size,
                height,
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
        let line_size = self.width * self.num_channels * std::mem::size_of::<T>();
        let height = self.height;
        let pitch = self.pitch;
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            ptr as "void*",
            value as "int",
            pitch as "std::size_t",
            line_size as "std::size_t",
            height as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemset2DAsync(
                ptr,
                pitch,
                value,
                line_size,
                height,
                (cudaStream_t) stream_ptr
            );
        });
        result!(ret)
    }

    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.width * self.height * self.num_channels
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

impl<T: Copy> Drop for DeviceBuffer2D<T> {
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
        let buffer = DeviceBuffer2D::<u32>::new(120, 80, 3);
        assert_eq!(buffer.width, 120);
        assert_eq!(buffer.height, 80);
        assert_eq!(buffer.num_channels, 3);
        assert_eq!(buffer.num_elements(), 120 * 80 * 3);
        assert!(buffer.pitch >= 360);
    }

    #[test]
    fn test_copy() {
        let stream = Stream::new().unwrap();
        let all_ones = vec![1_u32; 150];
        let host_buffer_all_ones = HostBuffer::from_slice(all_ones.as_slice());

        let mut device_buffer = DeviceBuffer2D::<u32>::new(10, 5, 3);
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer_all_ones, &stream)
                .unwrap();
        }

        let mut host_buffer = HostBuffer::<u32>::new(150);
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .unwrap();
        }

        let mut another_device_buffer = DeviceBuffer2D::<u32>::new(10, 5, 3);
        unsafe {
            another_device_buffer
                .copy_from_async(&host_buffer, &stream)
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(150);
        unsafe {
            another_device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .unwrap();
        }

        stream.synchronize().unwrap();

        assert_eq!(return_host_buffer.num_elements, 150);
        let return_data = return_host_buffer.to_vec();
        assert_eq!(return_data.len(), 150);
        assert!(return_data.into_iter().all(|v| v == 1_u32));
    }

    #[test]
    fn test_copy_2d() {
        let stream = Stream::new().unwrap();
        let image: [u8; 12] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4];
        let host_buffer = HostBuffer::from_slice(&image);
        let mut device_buffer = DeviceBuffer2D::<u8>::new(2, 2, 3);
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer, &stream)
                .unwrap();
        }
        let mut return_host_buffer = HostBuffer::<u8>::new(12);
        unsafe {
            device_buffer
                .copy_to_async(&mut return_host_buffer, &stream)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(
            &return_host_buffer.to_vec(),
            &[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        );
    }

    #[test]
    fn test_fill_with_byte() {
        let stream = Stream::new().unwrap();
        let mut device_buffer = DeviceBuffer2D::<u8>::new(2, 2, 3);
        let mut host_buffer = HostBuffer::<u8>::new(2 * 2 * 3);
        device_buffer.fill_with_byte(0xab, &stream).unwrap();
        unsafe {
            device_buffer
                .copy_to_async(&mut host_buffer, &stream)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(
            host_buffer.to_vec(),
            &[0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab]
        );
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().unwrap();
        let device_buffer = DeviceBuffer2D::<u32>::new(5, 5, 3);
        let mut host_buffer = HostBuffer::<u32>::new(80);
        let _ = unsafe { device_buffer.copy_to_async(&mut host_buffer, &stream) };
    }
}
