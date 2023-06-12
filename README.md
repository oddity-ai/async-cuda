<h1 align="center">
  Asynchronous CUDA, NPP and TensorRT
</h1>

## ℹ️ Introduction

The `async-cuda` family of libraries is an experimental set of libraries for interacting with the
GPU asynchronously. Since the GPU is just another I/O device (from the point of view of your
program), the async model actually fits surprisingly well.

The way it is implemented in `async-cuda` is that all operations are scheduled on a single runtime
thread that drives the GPU. The interface of this library enforces that synchronization happens when
it is necessary (and synchronization itself is also asynchronous).

The `async-cuda` project consists of:
* [`async-cuda-core`](crates/async-cuda-core): CUDA core primitives such as streams and buffers.
* [`async-cuda-npp`](crates/async-cuda-npp): Common NPP operations such as resizing and cropping.
* [`async-tensorrt`](crates/async-tensorrt): Minimal wrapper for TensorRT.

## 🛠 S️️tatus

This project is still a work-in-progress, and will contain bugs. Some parts of the API have not
been flushed out yet. Use with caution.

## ⚠️ Safety warning

The `async-cuda` crates are **intentionally unsafe**. Due to the limitations of how async Rust
currently works, usage of the async interface of this crate can cause undefined behavior in some
rare cases. It is up to the user of this crate to prevent this from happening by following these
rules:

* No futures produced by functions in this crate may be leaked (either by `std::mem::forget` or
  otherwise).
* Use a well-behaved runtime (one that will not forget your future) like Tokio or async-std.

Internally, the `Future` type in this crate schedules a CUDA call on a separate runtime thread. To
make the API as ergonomic as possible, the lifetime bounds of the closure (that is sent to the
runtime) are tied to the future object. To enforce this bound, the future will block and wait if it
is dropped. This mechanism relies on the future being driven to completion, and not forgotten. This
is not necessarily guaranteed. Unsafety may arise if either the runtime gives up on or forgets the
future, or the caller manually polls the future, then forgets it.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.