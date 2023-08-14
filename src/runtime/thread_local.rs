use std::sync::mpsc::Sender;

use once_cell::sync::Lazy;

use crate::error::Error;
use crate::runtime::execution::RUNTIME;
use crate::runtime::work::Work;

thread_local! {
    /// Thread-local runtime delegate.
    ///
    /// This object serves as the per-thread reference to the [`RUNTIME`] that can be used to
    /// enqueue work on the runtime thread.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// assert!(
    ///     RUNTIME_THREAD_LOCAL.with(|runtime|
    ///         runtime.enqueue(Work::new(|| ()))
    ///     ).is_ok()
    /// )
    /// ```
    pub(super) static RUNTIME_THREAD_LOCAL: Lazy<RuntimeThreadLocal> = Lazy::new(|| {
        RUNTIME.lock().unwrap().thread_local()
    });
}

/// Per-thread delegate for global runtime.
pub struct RuntimeThreadLocal(Sender<Work>);

impl RuntimeThreadLocal {
    /// Initialize [`RuntimeThreadLocal`] from [`Sender`] that allows the delegate to send work to
    /// the actual [`crate::runtime::execution::Runtime`].
    ///
    /// # Arguments
    ///
    /// * `sender` - Sender through which work can be sent to runtime.
    pub(super) fn from_sender(sender: Sender<Work>) -> Self {
        RuntimeThreadLocal(sender)
    }

    /// Enqueue work on runtime.
    ///
    /// # Arguments
    ///
    /// * `function` - Unit of work in function closure to enqueue.
    pub(super) fn enqueue(&self, function: Work) -> Result<(), Error> {
        self.0.send(function).map_err(|_| Error::Runtime)
    }
}

/// Enqueue work on the runtime without caring about the return value. This is useful in situations
/// where work must be performed but the result does not matter. For example, when destorying CUDA
/// object as part of dropping an object.
///
/// # Arguments
///
/// * `f` - Function closure to execute on runtime.
///
/// # Example
///
/// ```ignore
/// enqueue_decoupled(move || {
///     // ...
/// });
/// ```
#[inline]
pub fn enqueue_decoupled(f: impl FnOnce() + Send + 'static) {
    let f = Box::new(f);
    RUNTIME_THREAD_LOCAL
        .with(|runtime| runtime.enqueue(Work::new(f)))
        .expect("runtime broken")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_works() {
        let (tx, rx) = std::sync::mpsc::channel();
        assert!(RUNTIME_THREAD_LOCAL
            .with(|runtime| {
                runtime.enqueue(Work::new(move || {
                    assert!(tx.send(true).is_ok());
                }))
            })
            .is_ok());
        assert!(matches!(
            rx.recv_timeout(std::time::Duration::from_millis(100)),
            Ok(true),
        ));
    }

    #[test]
    fn test_enqueue_decoupled_works() {
        let (tx, rx) = std::sync::mpsc::channel();
        enqueue_decoupled(move || {
            assert!(tx.send(true).is_ok());
        });
        assert!(matches!(
            rx.recv_timeout(std::time::Duration::from_millis(100)),
            Ok(true),
        ));
    }
}
