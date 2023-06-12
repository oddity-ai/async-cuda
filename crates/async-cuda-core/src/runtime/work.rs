/// Represents a unit of work passed to the runtime. Holds a closure inside.
///
/// The closure is explictly [`Send`] because it will be sent over the thread boundary to be
/// executed in the runtime thread. For the same reason, the closure must be `'static`.
///
/// # Usage
///
/// ```ignore
/// let work = Work::new(|| {
///     // ...
/// });
/// work.run();
/// ```
pub struct Work(Box<dyn FnOnce() + Send + 'static>);

impl Work {
    /// Create a new work item.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure to execute.
    pub fn new(f: impl FnOnce() + Send + 'static) -> Self {
        Work(Box::new(f))
    }

    /// Execute work.
    pub fn run(self) {
        let Work(f) = self;
        f();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_runs() {
        let make_me_true = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let work = Work::new({
            let make_me_true = make_me_true.clone();
            move || {
                make_me_true.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        work.run();
        assert!(make_me_true.load(std::sync::atomic::Ordering::Relaxed));
    }
}
