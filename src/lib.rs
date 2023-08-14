#![recursion_limit = "256"]

pub mod device;
pub mod error;
pub mod ffi;
pub mod memory;
pub mod runtime;
pub mod stream;

#[cfg(feature = "npp")]
pub mod npp;

pub use device::{num_devices, Device, DeviceId, MemoryInfo};
pub use memory::{DeviceBuffer, DeviceBuffer2D, HostBuffer};
pub use stream::Stream;

pub use error::Error;
