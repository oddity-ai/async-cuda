mod includes;

pub mod context;
pub mod copy_constant_border;
pub mod remap;
pub mod resize;

#[cfg(feature = "unstable")]
pub mod resize_batch;

/// Convenience macro for turning an NPP error code into a `std::result::Result`.
///
/// # Usage
///
/// There are two possible uses of the macro:
///
/// (1) Shorthand to return `Ok(something)` or an NPP error:
///
/// ```ignore
/// result!(code, return_value);
/// ```
///
/// (2) Shorthand to return `Ok(())` or an NPP error:
///
/// ```ignore
/// result!(code)
/// ```
macro_rules! result {
    ($code:expr, $ok:expr) => {
        if $code == 0 {
            Ok($ok)
        } else {
            Err($crate::npp::error::Error::Npp($code))
        }
    };
    ($code:expr) => {
        result!($code, ())
    };
}

use result;
