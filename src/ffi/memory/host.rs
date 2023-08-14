use cpp::cpp;

use crate::ffi::memory::device::DeviceBuffer;
use crate::ffi::ptr::DevicePtr;
use crate::ffi::result;
use crate::ffi::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::HostBuffer`].
///
/// Refer to [`crate::HostBuffer`] for documentation.
pub struct HostBuffer<T: Copy> {
    pub num_elements: usize,
    internal: DevicePtr,
    _phantom: std::marker::PhantomData<T>,
}

/// Implements [`Send`] for [`HostBuffer`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Send for HostBuffer<T> {}

/// Implements [`Sync`] for [`HostBuffer`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Sync for HostBuffer<T> {}

impl<T: Copy> HostBuffer<T> {
    pub fn new(num_elements: usize) -> Self {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ptr_ptr = std::ptr::addr_of_mut!(ptr);
        let size = num_elements * std::mem::size_of::<T>();
        let ret = cpp!(unsafe [
            ptr_ptr as "void**",
            size as "std::size_t"
        ] -> i32 as "std::int32_t" {
            return cudaMallocHost(ptr_ptr, size);
        });
        match result!(ret, ptr.into()) {
            Ok(internal) => Self {
                internal,
                num_elements,
                _phantom: Default::default(),
            },
            Err(err) => {
                panic!("failed to allocate host memory: {err}");
            }
        }
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let mut this = Self::new(slice.len());
        this.copy_from_slice(slice);
        this
    }

    #[cfg(feature = "ndarray")]
    pub fn from_array<D: ndarray::Dimension>(array: &ndarray::ArrayView<T, D>) -> Self {
        let mut this = Self::new(array.len());
        this.copy_from_array(array);
        this
    }

    /// Copy from device buffer.
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns.
    #[inline]
    pub unsafe fn copy_from_async(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> Result<()> {
        other.copy_to_async(self, stream)
    }

    /// Copy to device buffer.
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns.
    #[inline]
    pub unsafe fn copy_to_async(&self, other: &mut DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        other.copy_from_async(self, stream)
    }

    pub fn copy_from_slice(&mut self, slice: &[T]) {
        // SAFETY: This is safe because we only instantiate the slice temporarily whilst having
        // exclusive mutable access to it to copy the data into it.
        let target = unsafe {
            std::slice::from_raw_parts_mut(self.internal.as_mut_ptr() as *mut T, self.num_elements)
        };
        target.copy_from_slice(slice);
    }

    #[cfg(feature = "ndarray")]
    pub fn copy_from_array<D: ndarray::Dimension>(&mut self, array: &ndarray::ArrayView<T, D>) {
        assert!(
            array.is_standard_layout(),
            "array must be in standard layout"
        );
        // SAFETY: This is safe because we only instantiate the slice temporarily whilst having
        // exclusive mutable access to it to copy the data into it.
        let target = unsafe {
            std::slice::from_raw_parts_mut(self.internal.as_mut_ptr() as *mut T, self.num_elements)
        };
        target.copy_from_slice(array.as_slice().unwrap());
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<T> {
        // SAFETY: This is safe because we only instantiate the slice temporarily to copy the data
        // to a safe Rust [`Vec`].
        let source = unsafe {
            std::slice::from_raw_parts(self.internal.as_ptr() as *const T, self.num_elements)
        };
        source.to_vec()
    }

    #[cfg(feature = "ndarray")]
    pub fn to_array_with_shape<D: ndarray::Dimension>(
        &self,
        shape: impl Into<ndarray::StrideShape<D>>,
    ) -> ndarray::Array<T, D> {
        let shape = shape.into();
        assert_eq!(
            self.num_elements,
            shape.size(),
            "provided shape does not match number of elements in buffer"
        );
        ndarray::Array::from_shape_vec(shape, self.to_vec()).unwrap()
    }

    /// Get readonly reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_internal(&self) -> &DevicePtr {
        &self.internal
    }

    /// Get readonly reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_mut_internal(&mut self) -> &mut DevicePtr {
        &mut self.internal
    }
}

impl<T: Copy> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        if self.internal.is_null() {
            return;
        }

        // SAFETY: Safe because we won't use the pointer after this.
        let mut internal = unsafe { self.internal.take() };
        let ptr = internal.as_mut_ptr();
        let _ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaFreeHost(ptr);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buffer = HostBuffer::<u32>::new(100);
        assert_eq!(buffer.num_elements, 100);
        assert_eq!(buffer.to_vec().len(), 100);
    }

    #[test]
    fn test_from_slice() {
        let all_ones = vec![1_u32; 200];
        let buffer = HostBuffer::from_slice(all_ones.as_slice());
        assert_eq!(buffer.num_elements, 200);
        let data = buffer.to_vec();
        assert_eq!(data.len(), 200);
        assert!(data.into_iter().all(|v| v == 1_u32));
    }

    #[test]
    fn test_copy() {
        let stream = Stream::new().unwrap();
        let all_ones = vec![1_u32; 100];
        let host_buffer = HostBuffer::from_slice(all_ones.as_slice());

        let mut device_buffer = DeviceBuffer::<u32>::new(100, &stream);
        unsafe {
            host_buffer
                .copy_to_async(&mut device_buffer, &stream)
                .unwrap();
        }

        let mut return_host_buffer = HostBuffer::<u32>::new(100);
        unsafe {
            return_host_buffer
                .copy_from_async(&device_buffer, &stream)
                .unwrap();
        }

        stream.synchronize().unwrap();

        assert_eq!(return_host_buffer.num_elements, 100);
        let return_data = return_host_buffer.to_vec();
        assert_eq!(return_data.len(), 100);
        assert!(return_data.into_iter().all(|v| v == 1_u32));
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().unwrap();
        let host_buffer = HostBuffer::<u32>::new(100);
        let mut device_buffer = DeviceBuffer::<u32>::new(101, &Stream::null());
        let _ = unsafe { host_buffer.copy_to_async(&mut device_buffer, &stream) };
    }
}
