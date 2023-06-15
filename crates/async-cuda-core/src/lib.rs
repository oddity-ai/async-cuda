#![recursion_limit = "256"]
#![cfg_attr(no_native_deps, allow(unused))]

pub mod device;
pub mod error;
pub mod ffi;
pub mod memory;
pub mod runtime;
pub mod stream;

pub use device::{num_devices, Device, DeviceId, MemoryInfo};
pub use memory::{DeviceBuffer, DeviceBuffer2D, HostBuffer};
pub use stream::Stream;

pub use error::Error;
