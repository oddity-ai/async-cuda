#[cfg(npp)]
use async_cuda::ffi::device::Device;
#[cfg(npp)]
use async_cuda::stream::Stream;

#[cfg(npp)]
use async_cuda::npp::ffi::context::Context;

/// This integration test helps determine which ffi functions affect the GPU state, or local thread
/// state.
///
/// This information is important to determine which function need to be executed on the runtime
/// thread, and which functions can be executed directly by the caller (and don't need to be async).
///
/// We only test functions where it is not immediately apparent whether or not the function has
/// side-effects. All wrappers for NPP operations aren't tested since it is evident that they affect
/// the GPU state.
///
/// # Find GPU side-effects
///
/// Run this integration test under the Nsight profile with the following command:
///
/// ```bash
/// nsys profile --output /tmp/side_effects_trace --force-overwrite true cargo test --release --test functions_side_effects_test
/// ```
///
/// Use the `nsys-ui` utility to inspect the report produced in `/tmp/side_effects_trace.qdstrm` and
/// determine for each function call if one or more CUDA API functions were invoked, and if the GPU
/// was affected in any way. Function calls are separated by device synchronization markers in the
/// trace.
///
/// # Find thread-local side-effects
///
/// These need to inferred from documentation or usage (or an educated guess).
///
/// # Results
///
/// | Function                      | Side-effect: GPU | Side-effect: thread-local |
/// | ----------------------------- | ---------------- | ------------------------- |
/// | `Context::from_null_stream`   | ❌               | ✅                        |
/// | `Context::from_stream`        | ❌               | ✅                        |
#[cfg(npp)]
#[tokio::test]
async fn test_side_effects() {
    // First block contains stuff we are not interested in measuring...
    let stream = Stream::new().await.unwrap();

    // A sequence of CUDA calls that is easy to find in the trace.
    Device::synchronize().unwrap();
    let _mem_info_1 = Device::memory_info().unwrap();
    let _mem_info_2 = Device::memory_info().unwrap();
    let _mem_info_3 = Device::memory_info().unwrap();
    let _mem_info_4 = Device::memory_info().unwrap();
    Device::synchronize().unwrap();

    let _context_null = Context::from_null_stream();
    Device::synchronize().unwrap();

    let _context_new = Context::from_stream(stream);
    Device::synchronize().unwrap();
}
