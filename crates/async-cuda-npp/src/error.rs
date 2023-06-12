/// An error that occurred in NPP.
#[derive(Debug, Clone)]
pub enum Error {
    /// Error code as reported by NPP.
    ///
    /// [NPP documentation](https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga1105a17b5e76381583c46ecd6a60fe21)
    Npp(i32),
    /// Error in CUDA backend.
    ///
    /// Refer to [`async_cuda_core::Error`].
    Cuda(async_cuda_core::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Cuda(err) => write!(f, "{err}"),
            Error::Npp(error_code) => write!(f, "error code produced by NPP: {error_code}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<async_cuda_core::Error> for Error {
    #[inline]
    fn from(err: async_cuda_core::Error) -> Self {
        Error::Cuda(err)
    }
}
