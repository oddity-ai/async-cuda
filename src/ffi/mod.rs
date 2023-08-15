mod includes;

pub mod device;
pub mod error;
pub mod memory;
pub mod ptr;
pub mod stream;

#[cfg(feature = "npp")]
pub mod npp;

/// Convenience macro for turning a CUDA error code into a `std::result::Result`.
///
/// # Usage
///
/// There are two possible uses of the macro:
///
/// (1) Shorthand to return `Ok(something)` or a CUDA error:
///
/// ```ignore
/// result!(code, return_value);
/// ```
///
/// (2) Shorthand to return `Ok(())` or a CUDA error:
///
/// ```ignore
/// result!(code)
/// ```
macro_rules! result {
    ($code:expr, $ok:expr) => {
        if $code == 0 {
            Ok($ok)
        } else {
            Err($crate::error::Error::Cuda($code))
        }
    };
    ($code:expr) => {
        result!($code, ())
    };
}

use result;
