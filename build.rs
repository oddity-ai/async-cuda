#[cfg(unix)]
fn main() {
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include_path =
        std::env::var("CUDA_INCLUDE_PATH").unwrap_or_else(|_| format!("{cuda_path}/include"));

    let cuda_lib_path =
        std::env::var("CUDA_LIB_PATH").unwrap_or_else(|_| format!("{cuda_path}/lib64"));

    cpp_build::Config::new()
        .include(cuda_include_path)
        .build("src/lib.rs");

    println!("cargo:rustc-link-search={cuda_lib_path}");
    println!("cargo:rustc-link-lib=cudart");

    #[cfg(feature = "npp")]
    link_npp_libraries();
}

#[cfg(windows)]
fn main() {
    let cuda_path = std::env::var("CUDA_PATH").expect("Missing environment variable `CUDA_PATH`.");
    let cuda_path = std::path::Path::new(&cuda_path);
    cpp_build::Config::new()
        .include(cuda_path.join("include"))
        .build("src/lib.rs");
    println!(
        "cargo:rustc-link-search={}",
        cuda_path.join("lib").join("x64").display()
    );
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
