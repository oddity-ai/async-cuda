fn main() {
    let cuda_path = std::env::var("CUDA_PATH").map(std::path::PathBuf::from);

    #[cfg(not(windows))]
    let cuda_path = cuda_path.unwrap_or_else(|_| std::path::PathBuf::from("/usr/local/cuda"));
    #[cfg(windows)]
    let cuda_path = cuda_path.expect("Missing environment variable `CUDA_PATH`.");

    let cuda_include_path = std::env::var("CUDA_INCLUDE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| cuda_path.join("include"));

    let cuda_lib_path = std::env::var("CUDA_LIB_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            #[cfg(not(windows))]
            {
                cuda_path.join("lib64")
            }
            #[cfg(windows)]
            {
                cuda_path.join("lib").join("x64")
            }
        });

    cpp_build::Config::new()
        .include(cuda_include_path)
        .build("src/lib.rs");

    println!("cargo:rustc-link-search={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=cudart");

    #[cfg(feature = "npp")]
    link_npp_libraries();
}

#[cfg(feature = "npp")]
fn link_npp_libraries() {
    println!("cargo:rustc-link-lib=nppc");
    println!("cargo:rustc-link-lib=nppig");
    println!("cargo:rustc-link-lib=nppidei");
}
