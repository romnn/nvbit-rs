use std::env;
use std::path::PathBuf;

fn cuda_library_path() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        // CUDA_LIBRARY_PATH can be used to hardcode CUDA location
        let split_char = if cfg!(target_os = "windows") {
            ";"
        } else {
            ":"
        };
        path.split(split_char).map(|s| PathBuf::from(s)).collect()
    } else {
        vec![]
    }
}

pub fn find_cuda_windows() -> Vec<PathBuf> {
    let candidates = cuda_library_path();
    if candidates.len() > 0 {
        return candidates;
    }
    if let Ok(path) = env::var("CUDA_PATH") {
        // if CUDA_LIBRARY_PATH is not found, then CUDA_PATH will be used when
        // building for Windows to locate the Cuda installation.
        // CUDA installs the full Cuda SDK for 64-bit, but only a limited set of
        // libraries for 32-bit.

        // check to see which target we're building for.
        let target = env::var("TARGET").expect("cargo target");

        // targets use '-' separators. e.g. x86_64-pc-windows-msvc
        let target_components: Vec<_> = target.as_str().split("-").collect();

        // check that we're building for Windows. This code assumes that the layout in
        // CUDA_PATH matches Windows.
        if target_components[2] != "windows" {
            println!(
                "cargo:warning=CUDA_PATH env variable is used on Windows, yet build target is {}",
                target
            );
            return vec![];
        }

        // sanity check that the second component of 'target' is "pc"
        debug_assert_eq!(
            "pc", target_components[1],
            "Expected a Windows target to have the second component be 'pc'. Target: {}",
            target
        );

        // x86_64 should use the libs in the "lib/x64" directory.
        let lib_path = match target_components[0] {
            "x86_64" => "x64",
            _ => {
                println!(
                    "cargo:warning=unsupported architecture {}",
                    target_components[0]
                );
                return vec![];
            }
        };

        // i.e. $CUDA_PATH/lib/x64
        return vec![PathBuf::from(path).join("lib").join(lib_path)];
    }

    vec![]
}

pub fn find_cuda_unix() -> Vec<PathBuf> {
    let mut candidates = cuda_library_path();
    candidates.extend([PathBuf::from("/opt/cuda"), PathBuf::from("/usr/local/cuda")]);
    candidates.extend(
        glob::glob("/usr/local/cuda-*")
            .expect("glob cuda")
            .filter_map(|p| p.ok()),
    );

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.extend([lib.clone(), lib.join("stubs")]);
        }
        let base = base.join("targets/x86_64-linux");
        if base.join("include/cuda.h").is_file() {
            valid_paths.extend([base.join("lib"), base.join("lib/stubs")]);
            continue;
        }
    }
    valid_paths
}

pub fn find_cuda() -> Vec<PathBuf> {
    if cfg!(target_os = "windows") {
        find_cuda_windows()
    } else {
        find_cuda_unix()
    }
}
