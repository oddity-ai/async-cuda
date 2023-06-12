/// Convenience macro for testing to take a memory slice and put it on the device and return the
/// [`async_cuda_core::DeviceBuffer2D`] that refers to it.
macro_rules! to_device_2d {
    ($slice:expr, $width:expr, $height:expr, $num_channels:expr, $stream:expr) => {{
        let host_buffer = async_cuda_core::HostBuffer::from_slice($slice).await;
        let mut device_buffer =
            async_cuda_core::DeviceBuffer2D::new($width, $height, $num_channels).await;
        device_buffer
            .copy_from(&host_buffer, $stream)
            .await
            .unwrap();
        device_buffer
    }};
}

/// Convenience macro for testing to take a [`async_cuda_core::DeviceBuffer2D`] and copy it back to
/// the host, then return a [`Vec`] of that memory.
macro_rules! to_host_2d {
    ($device_buffer:expr, $stream:expr) => {{
        let mut host_buffer = async_cuda_core::HostBuffer::new($device_buffer.num_elements()).await;
        $device_buffer
            .copy_to(&mut host_buffer, $stream)
            .await
            .unwrap();
        host_buffer.to_vec()
    }};
}

pub(crate) use to_device_2d;
pub(crate) use to_host_2d;
