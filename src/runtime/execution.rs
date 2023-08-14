use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

use crate::runtime::thread_local::RuntimeThreadLocal;
use crate::runtime::work::Work;

/// Refers to the global runtime. The runtime is responsible for running all CUDA operations in a
/// dedicated thread.
///
/// Note that this object should not be used by callers because each thread gets its own delegate
/// object to communicate with the runtime.
///
/// # Usage
///
/// Each thread should get its own [`RuntimeThreadLocal`] object, which acts as delegate object.
///
/// Use `Runtime::thread_local` to get the thread local object:
///
/// ```ignore
/// let runtime = RUNTIME.lock().unwrap().thread_local();
/// ```
pub(super) static RUNTIME: Lazy<Mutex<Runtime>> = Lazy::new(|| Mutex::new(Runtime::new()));

/// Runtime object that holds the runtime thread and a channel
/// to send jobs onto the worker queue.
pub struct Runtime {
    join_handle: Option<std::thread::JoinHandle<()>>,
    run_flag: Arc<AtomicBool>,
    work_tx: Sender<Work>,
}

impl Runtime {
    /// Acquire a thread local delegate for the runtime.
    pub(super) fn thread_local(&self) -> RuntimeThreadLocal {
        RuntimeThreadLocal::from_sender(self.work_tx.clone())
    }

    /// Create runtime.
    fn new() -> Self {
        let run_flag = Arc::new(AtomicBool::new(true));
        let (work_tx, work_rx) = channel::<Work>();

        let join_handle = std::thread::spawn({
            let run_flag = run_flag.clone();
            move || Self::worker(run_flag, work_rx)
        });

        Runtime {
            join_handle: Some(join_handle),
            run_flag,
            work_tx,
        }
    }

    /// Worker loop. Receives jobs from the worker queue and executes them until [`run_flag`]
    /// becomes `false`.
    ///
    /// # Arguments
    ///
    /// * `run_flag` - Atomic flag that indicates whether the worker should continue running.
    /// * `work_rx` - Receives work to execute.
    fn worker(run_flag: Arc<AtomicBool>, work_rx: Receiver<Work>) {
        while run_flag.load(Ordering::Relaxed) {
            match work_rx.recv() {
                Ok(work) => work.run(),
                Err(_) => break,
            }
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.run_flag.store(false, Ordering::Relaxed);

        // Put dummy workload into the queue to trigger the loop to continue and encounted the
        // `run_flag` that is now false, then stop. Note that if this fails, it means the underlying
        // channel is broken. It is not a problem, since that must mean the worker already quit
        // before, and it will join immediatly.
        let _ = self.work_tx.send(Work::new(|| {}));

        if let Some(join_handle) = self.join_handle.take() {
            join_handle
                .join()
                .expect("failed to join on runtime thread");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop() {
        let runtime = Runtime::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        drop(runtime);
    }

    #[test]
    fn test_it_does_work() {
        let runtime = Runtime::new();
        let (tx, rx) = std::sync::mpsc::channel();
        assert!(runtime
            .thread_local()
            .enqueue(Work::new(move || {
                assert!(tx.send(true).is_ok());
            }))
            .is_ok());
        assert!(matches!(
            rx.recv_timeout(std::time::Duration::from_millis(100)),
            Ok(true),
        ));
    }
}
