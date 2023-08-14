use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use crate::error::Error;
use crate::runtime::thread_local::RUNTIME_THREAD_LOCAL;
use crate::runtime::work::Work;
use crate::stream::Stream;

type Result<T> = std::result::Result<T, Error>;

/// Represents a closure that can be executed in the runtime.
pub type Closure<'closure> = Box<dyn FnOnce() + Send + 'closure>;

/// Future for CUDA operations.
///
/// Note that this future abstracts over two different asynchronousprimitives: dedicated-thread
/// semantics, and stream asynchrony.
///
/// # Dedicated-thread semantics
///
/// In this crate, all operations that use CUDA internally are off-loaded to a dedicated thread (the
/// runtime). This improves CUDA's ability to parallelize without being interrupted by the OS
/// scheduler or being affected by starvation when under load.
///
/// # Stream asynchrony
///
/// CUDA has internal asynchrony as well. Lots of CUDA operations are asynchronous with respect to
/// the host with regards to the stream they are bound to.
///
/// It is important to understand that most of the operations in this crate do *NOT* actually wait
/// for the CUDA asynchronous operation to complete. Instead, the operation is started and then the
/// future becomes ready. This means that if the caller must still synchronize the underlying CUDA
/// stream.
///
/// # Usage
///
/// To create a [`Future`], move the closure into with `Future::new`:
///
/// ```
/// # use async_cuda::runtime::Future;
/// # tokio_test::block_on(async {
/// let future = Future::new(move || {
///     ()
/// });
/// let return_value = future.await;
/// assert_eq!(return_value, ());
/// # })
/// ```
pub struct Future<'closure, T> {
    shared: Arc<Mutex<Shared<'closure, T>>>,
    completed: Arc<Condvar>,
    _phantom: std::marker::PhantomData<&'closure ()>,
}

impl<'closure, T> Future<'closure, T> {
    /// Wrap the provided function in this future. It will be sent to the runtime thread and
    /// executed there. The future resolves once the call on the runtime completes.
    ///
    /// # Arguments
    ///
    /// * `call` - Closure that contains relevant function call.
    ///
    /// # Example
    ///
    /// ```
    /// # use async_cuda::runtime::Future;
    /// # tokio_test::block_on(async {
    /// let return_value = Future::new(|| ()).await;
    /// assert_eq!(return_value, ());
    /// })
    /// ```
    #[inline]
    pub fn new<F>(call: F) -> Self
    where
        F: FnOnce() -> T + Send + 'closure,
        T: Send + 'closure,
    {
        let shared = Arc::new(Mutex::new(Shared::new()));
        let completed = Arc::new(Condvar::new());
        let closure = Box::new({
            let shared = shared.clone();
            let completed = completed.clone();
            move || {
                let return_value = call();
                let mut shared = shared.lock().unwrap();
                match shared.state {
                    State::Running => {
                        shared.complete(return_value);
                        // If the future was cancelled before the function finished, the drop
                        // function is now waiting for us to finish. Notify it here.
                        completed.notify_all();
                        // If the future is still active, then this will wake the executor and
                        // cause it to poll the future again. Since we changed the state to
                        // `State::Completed`, the future will return a result.
                        if let Some(waker) = shared.waker.take() {
                            waker.wake();
                        }
                    }
                    _ => {
                        panic!("unexpected state");
                    }
                }
            }
        });

        shared.lock().unwrap().initialize(closure);

        Self {
            shared,
            completed,
            _phantom: Default::default(),
        }
    }
}

impl<'closure, T> std::future::Future for Future<'closure, T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let mut shared = self.shared.lock().unwrap();
        match shared.state {
            State::New => Poll::Pending,
            // This is the first time that the future is polled. We take the function out and
            // enqueue it on the runtime, then change the state from `Initialized` to `Running`.
            State::Initialized => {
                shared.running(cx.waker().clone());
                let closure: Box<dyn FnOnce() + Send + 'closure> =
                    shared.closure.take().expect("initialized without function");
                let closure: Box<dyn FnOnce() + Send + 'static> = unsafe {
                    // SAFETY: This is safe because in `drop` we make sure to wait for the runtime
                    // thread closure to complete if it still exists. This ensures that the closure
                    // cannot outlive this future object. Because of this, we can simply erase the
                    // `'closure` lifetime bound here and pretend it is `'static`.
                    std::mem::transmute(closure)
                };
                RUNTIME_THREAD_LOCAL.with(|runtime| {
                    runtime.enqueue(Work::new(closure)).expect("runtime broken");
                });

                Poll::Pending
            }
            // The future is still running.
            State::Running => Poll::Pending,
            // The future has completed and a return value is available. We take out the return
            // value and change the state from `Completed` to `Done`.
            State::Completed => {
                shared.done();
                Poll::Ready(shared.return_value.take().unwrap())
            }
            // It is illegal to poll a future after it has become ready before.
            State::Done => {
                panic!("future polled after completion");
            }
        }
    }
}

impl<'closure, T> Drop for Future<'closure, T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        // SAFETY:
        //
        // Only if the state is `State::Running` there is a chance that the closure is currently
        // used and active. In that case we must wait for it to finish because we promised that the
        // closure outlives the future.
        //
        // Note that no race conditions can occur here because we currently have the lock on the
        // state and it is only released when waiting for the condition variable later. And even
        // after, the state is guaranteed only to change from the runtime thread i.e. the only
        // allowed state change is `State::Running` -> `State::Completed`.
        if let State::Running = shared.state {
            // SAFETY: This is where we wait for the closure to finish on the runtime thread. Since
            // the only allowed state change at this point it `State::Running` ->
            // `State::Completed`, we only need to check for that one.
            while !matches!(shared.state, State::Completed) {
                shared = self.completed.wait(shared).unwrap();
            }
        }
    }
}

/// Future for the stream synchronization operation.
///
/// Unlike the generic [`Future`] provided by this crate, this variant only becomes ready after all
/// operations on the given stream have completed.
///
/// # Usage
///
/// ```ignore
/// let null_stream = Stream::null();
/// let result = SynchronizeFuture::new(&null_stream).await;
/// ```
pub struct SynchronizeFuture<'closure>(Future<'closure, Result<()>>);

impl<'closure> SynchronizeFuture<'closure> {
    /// Create future that becomes ready only when all currently scheduled work on the given stream
    /// has completed.
    ///
    /// # Arguments
    ///
    /// * `stream` - Reference to stream to synchronize.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stream = Stream::new();
    /// SynchronizeFuture::new(&stream).await.unwrap();
    /// ```
    #[inline]
    pub(crate) fn new(stream: &'closure Stream) -> Self {
        let shared = Arc::new(Mutex::new(Shared::new()));
        let completed = Arc::new(Condvar::new());

        // Create a closure that will be sent to the runtime thread and then executed in the
        // dedicated thread.
        let closure = Box::new({
            let shared = shared.clone();
            let completed = completed.clone();
            move || {
                let callback = {
                    let shared = shared.clone();
                    let completed = completed.clone();
                    // Create a closure that will be executed after all work on the current CUDA
                    // stream has completed. This closure will wake the future and make it ready.
                    move || Self::complete(shared, completed, Ok(()))
                };
                if let Err(err) = stream.inner().add_callback(callback) {
                    // If for some reason CUDA can't add the callback, we must still ready the
                    // future or it will never complete.
                    Self::complete(shared, completed, Err(err));
                }
            }
        });

        shared.lock().unwrap().initialize(closure);

        Self(Future {
            shared,
            completed,
            _phantom: Default::default(),
        })
    }

    /// Set the future's shared state to reflect that the function has completed with the given
    /// return value.
    ///
    /// # Arguments
    ///
    /// * `shared` - Closure's shared state.
    /// * `return_value` - Closure's return value.
    #[inline]
    fn complete(
        shared: Arc<Mutex<Shared<Result<()>>>>,
        completed: Arc<Condvar>,
        return_value: Result<()>,
    ) {
        if let Ok(mut shared) = shared.lock() {
            match shared.state {
                State::Running => {
                    shared.complete(return_value);
                    // If the future was cancelled before the function finished, the drop
                    // function is now waiting for us to finish. Notify it here.
                    completed.notify_all();
                    // If the future is still active, then this will wake the executor and
                    // cause it to poll the future again. Since we changed the state to
                    // `State::Completed`, the future will return a result.
                    if let Some(waker) = shared.waker.take() {
                        waker.wake();
                    }
                }
                _ => {
                    panic!("unexpected state");
                }
            }
        }
    }
}

impl<'closure> std::future::Future for SynchronizeFuture<'closure> {
    type Output = Result<()>;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

/// Share state between the future and the closure that is sent over to the runtime.
struct Shared<'closure, T> {
    /// Current future state.
    state: State,
    /// Closure to execute on runtime.
    closure: Option<Closure<'closure>>,
    /// Waker that can be used to wake the future.
    waker: Option<Waker>,
    /// Return value of future.
    return_value: Option<T>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum State {
    /// Future has been created but not yet been polled.
    New,
    /// Future has been assigned a closure and has internal state. It has not yet been polled.
    Initialized,
    /// Future has been polled and is scheduled. It is running and the waker will wake it up at some
    /// point.
    Running,
    /// Future has completed and has a result.
    Completed,
    /// Future is done and result has been taken out.
    Done,
}

impl<'closure, T> Shared<'closure, T> {
    /// Create new [`Future`] shared state.
    fn new() -> Self {
        Shared {
            state: State::New,
            closure: None,
            waker: None,
            return_value: None,
        }
    }

    /// Initialize state and move function closure into shared state.
    ///
    /// # Arguments
    ///
    /// * `closure` - The function closure. Stil unscheduled at this point.
    ///
    /// # Safety
    ///
    /// This state change may only be performed from the thread that holds the future.
    #[inline]
    fn initialize(&mut self, closure: Closure<'closure>) {
        self.closure = Some(closure);
        self.state = State::Initialized;
    }

    /// Set running state and store waker.
    ///
    /// # Arguments
    ///
    /// * `waker` - Waker that can be used by runtime to wake future.
    ///
    /// # Safety
    ///
    /// This state change may only be performed from the thread that holds the future.
    #[inline]
    fn running(&mut self, waker: Waker) {
        self.waker = Some(waker);
        self.state = State::Running;
    }

    /// Complete state and set return value.
    ///
    /// # Arguments
    ///
    /// * `return_value` - Function closure return value.
    ///
    /// # Safety
    ///
    /// This state change may only be performed from the runtime thread.
    #[inline]
    fn complete(&mut self, return_value: T) {
        self.return_value = Some(return_value);
        self.state = State::Completed;
    }

    /// Set done state.
    ///
    /// # Safety
    ///
    /// This state change may only be performed from the runtime thread.
    #[inline]
    fn done(&mut self) {
        self.state = State::Done;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn test_future() {
        assert!(Future::new(|| true).await);
    }

    #[tokio::test]
    async fn test_future_order() {
        let first_future_completed = Arc::new(AtomicBool::new(false));
        Future::new({
            let first_future_completed = first_future_completed.clone();
            move || {
                first_future_completed.store(true, Ordering::Relaxed);
            }
        })
        .await;
        assert!(
            Future::new({
                let first_future_completed = first_future_completed.clone();
                move || first_future_completed.load(Ordering::Relaxed)
            })
            .await
        );
    }

    #[tokio::test]
    async fn test_future_order_simple() {
        let mut first_future_completed = false;
        Future::new(|| first_future_completed = true).await;
        assert!(Future::new(|| first_future_completed).await);
    }

    #[tokio::test]
    async fn test_future_outlives_closure() {
        let mut count_completed = 0;
        let mut count_cancelled = 0;
        for _ in 0..1_000 {
            let mut start_of_closure = false;
            let mut end_of_closure = false;
            let future = Future::new(|| {
                start_of_closure = true;
                std::thread::sleep(std::time::Duration::from_millis(1));
                end_of_closure = true;
            });
            let future_with_small_delay = async {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                future.await
            };
            let _ =
                tokio::time::timeout(std::time::Duration::from_nanos(0), future_with_small_delay)
                    .await;
            assert!((start_of_closure && end_of_closure) || (!start_of_closure && !end_of_closure));
            if end_of_closure {
                count_completed += 1;
            } else {
                count_cancelled += 1;
            }
        }
        println!("num completed: {count_completed}");
        println!("num cancelled: {count_cancelled}");
    }

    #[tokio::test]
    async fn test_future_outlives_closure_manual() {
        let mut start_of_closure = false;
        let mut end_of_closure = false;
        let future = Future::new(|| {
            start_of_closure = true;
            std::thread::sleep(std::time::Duration::from_nanos(1000));
            end_of_closure = true;
        });
        let future_with_small_delay = async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            future.await
        };
        let _ = tokio::time::timeout(std::time::Duration::ZERO, future_with_small_delay).await;
        assert!((!start_of_closure && !end_of_closure))
    }

    #[tokio::test]
    async fn test_future_does_not_run_if_cancelled_before_polling() {
        let mut start_of_closure = false;
        let mut end_of_closure = false;
        let future = Future::new(|| {
            start_of_closure = true;
            std::thread::sleep(std::time::Duration::from_nanos(1000));
            end_of_closure = true;
        });
        drop(future);
        assert!((!start_of_closure && !end_of_closure))
    }

    #[tokio::test]
    async fn test_synchronization_future() {
        let stream = crate::Stream::new().await.unwrap();
        assert!(SynchronizeFuture::new(&stream).await.is_ok());
    }
}
