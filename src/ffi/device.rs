use cpp::cpp;

use crate::device::DeviceId;
use crate::device::MemoryInfo;
use crate::ffi::result;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::num_devices`].
///
/// Refer to [`crate::num_devices`] for documentation.
pub fn num_devices() -> Result<usize> {
    let mut num = 0_i32;
    let num_ptr = std::ptr::addr_of_mut!(num);
    let ret = cpp!(unsafe [
        num_ptr as "std::int32_t*"
    ] -> i32 as "std::int32_t" {
        return cudaGetDeviceCount(num_ptr);
    });

    result!(ret, num as usize)
}

/// Synchronous implementation of [`crate::Device`].
///
/// Refer to [`crate::Device`] for documentation.
pub struct Device;

impl Device {
    pub fn get() -> Result<DeviceId> {
        let mut id: i32 = 0;
        let id_ptr = std::ptr::addr_of_mut!(id);
        let ret = cpp!(unsafe [
            id_ptr as "int"
        ] -> i32 as "int" {
            return cudaGetDevice(id_ptr);
        });
        result!(ret, id)
    }

    pub fn set(id: DeviceId) -> Result<()> {
        let ret = cpp!(unsafe [
            id as "int"
        ] -> i32 as "int" {
            return cudaSetDevice(id);
        });
        result!(ret)
    }

    #[inline]
    pub fn bind(id: DeviceId) -> Result<DeviceGuard> {
        DeviceGuard::activate(id)
    }

    #[inline]
    pub fn bind_or_panic(id: DeviceId) -> DeviceGuard {
        DeviceGuard::activate(id)
            .unwrap_or_else(|err| panic!("failed to bind to device {}: {}", id, err))
    }

    pub fn synchronize() -> Result<()> {
        let ret = cpp!(unsafe [] -> i32 as "std::int32_t" {
            return cudaDeviceSynchronize();
        });
        result!(ret)
    }

    pub fn memory_info() -> Result<MemoryInfo> {
        let mut free: usize = 0;
        let free_ptr = std::ptr::addr_of_mut!(free);
        let mut total: usize = 0;
        let total_ptr = std::ptr::addr_of_mut!(total);

        let ret = cpp!(unsafe [
            free_ptr as "std::size_t*",
            total_ptr as "std::size_t*"
        ] -> i32 as "std::int32_t" {
            return cudaMemGetInfo(free_ptr, total_ptr);
        });
        result!(ret, MemoryInfo { free, total })
    }
}

/// Guard to keep specified active for the duration of the surrounding scope.
pub struct DeviceGuard {
    pub active: DeviceId,
    pub previous: DeviceId,
}

impl DeviceGuard {
    /// Create [`DeviceGuard`] and activate it.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to activate.
    fn activate(device: DeviceId) -> Result<DeviceGuard> {
        let previous = Device::get()?;
        Device::set(device)?;
        Ok(Self {
            active: device,
            previous,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        Device::set(self.previous).unwrap_or_else(|err| {
            panic!("failed to set device ordinal: {} ({})", self.previous, err)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_devices() {
        assert!(matches!(num_devices(), Ok(num) if num > 0));
    }

    #[test]
    fn test_get_device() {
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_set_device() {
        assert!(Device::set(0).is_ok());
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_bind_device() {
        assert!(Device::bind(0).is_ok());
    }

    #[test]
    fn test_synchronize() {
        assert!(Device::synchronize().is_ok());
    }

    #[test]
    fn test_memory_info() {
        let memory_info = Device::memory_info().unwrap();
        assert!(memory_info.free > 0);
        assert!(memory_info.total > 0);
    }
}
