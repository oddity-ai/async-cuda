#[cfg(not(no_native_deps))]
pub(crate) use cpp::cpp;

#[cfg(no_native_deps)]
macro_rules! cpp {
    {{ $($t:tt)* }} => {};
    {$(unsafe)? [$($a:tt)*] -> $ret:ty as $b:tt { $($t:tt)* } } => {
        $crate::ffi::utils::docs_build::panic::<$ret>()
    };
    { $($t:tt)* } => {
        $crate::ffi::utils::docs_build::panic::<()>()
    };
}

#[cfg(no_native_deps)]
pub(crate) use cpp;

#[cfg(no_native_deps)]
pub(crate) mod docs_build {
    pub fn panic<T>() -> T {
        panic!("docs-only build")
    }
}
