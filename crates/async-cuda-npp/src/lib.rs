#![recursion_limit = "256"]
#![cfg_attr(no_native_deps, allow(unused))]

pub mod constant_border;
pub mod copy_constant_border;
pub mod error;
pub mod ffi;
pub mod region;
pub mod remap;
pub mod resize;
pub mod stream;

#[cfg(feature = "unstable")]
pub mod resize_batch;

pub use constant_border::ConstantBorder;
pub use copy_constant_border::copy_constant_border;
pub use error::Error;
pub use region::Region;
pub use remap::remap;
pub use resize::resize;
pub use stream::Stream;

#[cfg(feature = "unstable")]
pub use resize_batch::resize_batch;

#[cfg(test)]
mod tests;
