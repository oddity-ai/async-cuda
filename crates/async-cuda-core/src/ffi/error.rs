use cpp::cpp;

/// Returns the description string for an error code.
///
/// Note that this function is not executed on the runtime thread, since it is purely a utility
/// function and should have no side-effects with regards to CUDA devices.
///
/// [CUDA documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g4bc9e35a618dfd0877c29c8ee45148f1)
///
/// # Arguments
///
/// * `error_code` - CUDA error code.
pub fn error_description(error_code: i32) -> String {
    let error_description = cpp!(unsafe [
      error_code as "std::int32_t"
    ] -> *const std::ffi::c_char as "const char*" {
      return cudaGetErrorString(static_cast<cudaError_t>(error_code));
    });
    // SAFETY: The pointer returned by `cudaGetErrorString` actually has a static lifetime so this
    // is safe for sure. We even copy inside the unsafe block so we just need it to remain for a
    // little bit.
    unsafe {
        std::ffi::CStr::from_ptr(error_description)
            .to_string_lossy()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_description() {
        assert_eq!(error_description(1), "invalid argument");
    }
}
