#[cfg(unix)]
fn main() {
    handle_docs_rs();
    cpp_build::Config::new()
        .include("/usr/local/cuda/include")
        .build("src/lib.rs");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=nppc");
    println!("cargo:rustc-link-lib=nppig");
    println!("cargo:rustc-link-lib=nppidei");
}

#[cfg(windows)]
fn main() {
    handle_docs_rs();
    let cuda_path = std::env::var("CUDA_PATH").expect("Missing environment variable `CUDA_PATH`.");
    let cuda_path = std::path::Path::new(&cuda_path);
    cpp_build::Config::new()
        .include(cuda_path.join("include"))
        .build("src/lib.rs");
    println!(
        "cargo:rustc-link-search={}",
        cuda_path.join("lib").join("x64").display()
    );
    println!("cargo:rustc-link-lib=nppc");
    println!("cargo:rustc-link-lib=nppig");
    println!("cargo:rustc-link-lib=nppidei");
}

fn handle_docs_rs() {
    if std::env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-cfg=no_native_deps");
        std::process::exit(0);
    }
}
