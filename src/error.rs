use crate::ffi::error::error_description;

/// An error that occurred during a CUDA operation.
#[derive(Debug, Clone)]
pub enum Error {
    /// Error code as reported by the CUDA backend.
    ///
    /// [CUDA documentation](https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga1105a17b5e76381583c46ecd6a60fe21)
    Cuda(i32),
    /// The runtime backend unexpectedly broke down. This is usually irrecoverable because the
    /// entire crate assumes that all backend execution will happen on the runtime thread.
    Runtime,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Cuda(code) => {
                let error_code = *code;
                let error_description = error_description(error_code);
                write!(
                    f,
                    "CUDA error ({}): {}",
                    error_code,
                    error_description.as_str(),
                )
            }
            Error::Runtime => write!(f, "CUDA runtime broken"),
        }
    }
}

impl std::error::Error for Error {}
