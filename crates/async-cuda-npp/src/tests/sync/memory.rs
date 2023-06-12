/// Convenience macro for testing to take a memory slice and put it on the device and return the
/// [`async_cuda_core::DeviceBuffer2D`] that refers to it.
macro_rules! to_device_2d {
    ($slice:expr, $width:expr, $height:expr, $num_channels:expr, $context:expr) => {{
        let host_buffer = async_cuda_core::ffi::memory::HostBuffer::from_slice($slice);
        let mut device_buffer =
            async_cuda_core::ffi::memory::DeviceBuffer2D::new($width, $height, $num_channels);
        // SAFETY: Stream is synchronized right after this.
        unsafe {
            device_buffer
                .copy_from_async(&host_buffer, &$context.stream.inner())
                .unwrap();
        }
        $context.stream.inner().synchronize().unwrap();
        device_buffer
    }};
}

/// Convenience macro for testing to take a [`async_cuda_core::DeviceBuffer2D`] and copy it back to
/// the host, then return a [`Vec`] of that memory.
macro_rules! to_host_2d {
    ($device_buffer:expr, $context:expr) => {{
        let mut host_buffer =
            async_cuda_core::ffi::memory::HostBuffer::new($device_buffer.num_elements());
        // SAFETY: Stream is synchronized right after this.
        unsafe {
            $device_buffer
                .copy_to_async(&mut host_buffer, &$context.stream.inner())
                .unwrap();
        }
        $context.stream.inner().synchronize().unwrap();
        host_buffer.to_vec()
    }};
}

pub(crate) use to_device_2d;
pub(crate) use to_host_2d;
