use crate::device::DeviceId;
use crate::device::MemoryInfo;
use crate::ffi::result;
use crate::ffi::utils::cpp;

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
            id_ptr as "std::int32_t*"
        ] -> i32 as "std::int32_t" {
            return cudaGetDevice(id_ptr);
        });
        result!(ret, id as DeviceId)
    }

    pub fn set(id: DeviceId) -> Result<()> {
        let id = id as i32;
        let ret = cpp!(unsafe [
            id as "std::int32_t"
        ] -> i32 as "std::int32_t" {
            return cudaSetDevice(id);
        });
        result!(ret)
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
