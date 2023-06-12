use crate::ffi;
use crate::runtime::Future;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Returns the number of compute-capable devices.
///
/// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f)
///
/// # Return value
///
/// Number of CUDA devices or error in case of failure.
pub async fn num_devices() -> Result<usize> {
    Future::new(ffi::device::num_devices).await
}

/// CUDA device ID.
pub type DeviceId = usize;

/// CUDA device.
pub struct Device;

impl Device {
    /// Returns which device is currently being used by [`DeviceId`].
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78)
    pub async fn get() -> Result<DeviceId> {
        Future::new(ffi::device::Device::get).await
    }

    /// Set device to be used for GPU executions.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb)
    ///
    /// # Arguments
    ///
    /// * `id` - Device ID to use.
    pub async fn set(id: DeviceId) -> Result<()> {
        Future::new(move || ffi::device::Device::set(id)).await
    }

    /// Synchronize the current CUDA device.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d)
    ///
    /// # Warning
    ///
    /// Note that this operation will block all device operations, even from other processes while
    /// running. Use this operation sparingly.
    pub async fn synchronize() -> Result<()> {
        Future::new(ffi::device::Device::synchronize).await
    }

    /// Gets free and total device memory.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g376b97f5ab20321ca46f7cfa9511b978)
    ///
    /// # Return value
    ///
    /// Total amount of memory and free memory in bytes.
    pub async fn memory_info() -> Result<MemoryInfo> {
        Future::new(ffi::device::Device::memory_info).await
    }
}

/// CUDA device memory information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryInfo {
    /// Amount of free device memory in bytes.
    pub free: usize,
    /// Total amount of device memroy in bytes.
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_num_devices() {
        assert!(matches!(num_devices().await, Ok(num) if num > 0));
    }

    #[tokio::test]
    async fn test_get_device() {
        assert!(matches!(Device::get().await, Ok(0)));
    }

    #[tokio::test]
    async fn test_set_device() {
        assert!(Device::set(0).await.is_ok());
        assert!(matches!(Device::get().await, Ok(0)));
    }

    #[tokio::test]
    async fn test_synchronize() {
        assert!(Device::synchronize().await.is_ok());
    }

    #[tokio::test]
    async fn test_memory_info() {
        let memory_info = Device::memory_info().await.unwrap();
        assert!(memory_info.free > 0);
        assert!(memory_info.total > 0);
    }
}
