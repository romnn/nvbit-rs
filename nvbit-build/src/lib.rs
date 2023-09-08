use std::path::PathBuf;
use std::process::{Command, Output};

/// Get the nvbit include dir.
///
/// **Note**: This function is intended to be used the build.rs context.
///
/// This can be useful when your crate uses nvbit and requires access to
/// the nvbit header files.
///
/// # Panics
/// When the `DEP_NVBIT_INCLUDE` environment variable is not set.
#[inline]
#[must_use]
pub fn nvbit_include() -> PathBuf {
    PathBuf::from(std::env::var("DEP_NVBIT_INCLUDE").expect("nvbit include path"))
        .canonicalize()
        .expect("canonicalize path")
}

/// Get the cargo output directory.
///
/// **Note**: This function is intended to be used the build.rs context.
///
/// # Panics
/// When the `OUT_DIR` environment variable is not set.
#[inline]
#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize path")
}

/// Get the cargo manifest directory.
///
/// **Note**: This function is intended to be used the build.rs context.
///
/// # Panics
/// When the `CARGO_MANIFEST_DIR` environment variable is not set.
#[inline]
#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize path")
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("Command failed")]
    Command(Output),
}

#[derive(Debug, Clone)]
pub struct Build {
    include_directories: Vec<PathBuf>,
    objects: Vec<PathBuf>,
    sources: Vec<PathBuf>,
    instrumentation_sources: Vec<PathBuf>,
    warnings: bool,
    warnings_as_errors: bool,
}

impl Default for Build {
    fn default() -> Self {
        Self::new()
    }
}

impl Build {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_directories: Vec::new(),
            objects: Vec::new(),
            sources: Vec::new(),
            instrumentation_sources: Vec::new(),
            warnings: false,
            warnings_as_errors: false,
        }
    }

    /// Compile and link static library with given name from inputs.
    ///
    /// # Errors
    /// When compilation fails, an error is returned.
    pub fn compile<O: AsRef<str>>(&self, output: O) -> Result<(), Error> {
        use std::ffi::OsStr;

        let mut objects = self.objects.clone();
        let include_args: Vec<_> = self
            .include_directories
            .iter()
            .map(|d| format!("-I{}", &d.to_string_lossy()))
            .collect();

        let mut compiler_flags = vec!["-Xcompiler", "-fPIC"];
        if self.warnings {
            compiler_flags.extend(["-Xcompiler", "-Wall"]);
        }
        if self.warnings_as_errors {
            compiler_flags.extend(["-Xcompiler", "-Werror"]);
        }

        // compile instrumentation functions
        for (i, src) in self.instrumentation_sources.iter().enumerate() {
            let default_name = format!("instr_src_{i}");
            let obj = output_path()
                .join(
                    src.file_name()
                        .and_then(OsStr::to_str)
                        .unwrap_or(&default_name),
                )
                .with_extension("o");
            let mut cmd = Command::new("nvcc");
            cmd.args(&include_args)
                .args([
                    "-maxrregcount=24",
                    "-Xptxas",
                    "-astoolspatch",
                    "--keep-device-functions",
                ])
                .args(&compiler_flags)
                .arg("-c")
                .arg(src)
                .arg("-o")
                .arg(&*obj.to_string_lossy());

            println!("cargo:warning={cmd:?}");
            let result = cmd.output()?;
            if !result.status.success() {
                return Err(Error::Command(result));
            }
            objects.push(obj);
        }

        // compile sources
        for (i, src) in self.sources.iter().enumerate() {
            let default_name = format!("src_{i}");
            let obj = output_path()
                .join(
                    src.file_name()
                        .and_then(OsStr::to_str)
                        .unwrap_or(&default_name),
                )
                .with_extension("o");
            let mut cmd = Command::new("nvcc");
            cmd.args(&include_args)
                .args(&compiler_flags)
                .args(["-dc", "-c"])
                .arg(src)
                .arg("-o")
                .arg(&*obj.to_string_lossy());
            println!("cargo:warning={cmd:?}");
            let result = cmd.output()?;
            if !result.status.success() {
                return Err(Error::Command(result));
            }
            objects.push(obj);
        }

        // link device functions
        let dev_link_obj = output_path().join("dev_link.o");
        let mut cmd = Command::new("nvcc");
        cmd.args(&include_args)
            .args(&compiler_flags)
            .arg("-dlink")
            .args(&objects)
            .arg("-o")
            .arg(&*dev_link_obj.to_string_lossy());
        println!("cargo:warning={cmd:?}");
        let result = cmd.output()?;
        if !result.status.success() {
            return Err(Error::Command(result));
        }
        objects.push(dev_link_obj);

        // link everything together
        let mut cmd = Command::new("ar");
        cmd.args([
            "cru",
            &output_path()
                .join(format!("lib{}.a", output.as_ref()))
                .to_string_lossy(),
        ])
        .args(&objects);
        println!("cargo:warning={cmd:?}");
        let result = cmd.output()?;
        if !result.status.success() {
            return Err(Error::Command(result));
        }

        println!("cargo:rustc-link-search=native={}", output_path().display());
        println!(
            "cargo:rustc-link-lib=static:+whole-archive={}",
            output.as_ref()
        );
        Ok(())
    }

    pub fn object<P: Into<PathBuf>>(&mut self, obj: P) -> &mut Self {
        self.objects.push(obj.into());
        self
    }

    pub fn objects<P>(&mut self, objects: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: Into<PathBuf>,
    {
        for obj in objects {
            self.object(obj);
        }
        self
    }

    pub fn instrumentation_source<P: Into<PathBuf>>(&mut self, src: P) -> &mut Self {
        self.instrumentation_sources.push(src.into());
        self
    }

    pub fn instrumentation_sources<P>(&mut self, sources: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: Into<PathBuf>,
    {
        for src in sources {
            self.instrumentation_source(src);
        }
        self
    }

    pub fn source<P: Into<PathBuf>>(&mut self, dir: P) -> &mut Self {
        self.sources.push(dir.into());
        self
    }

    pub fn sources<P>(&mut self, sources: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: Into<PathBuf>,
    {
        for src in sources {
            self.source(src);
        }
        self
    }

    pub fn include<P: Into<PathBuf>>(&mut self, dir: P) -> &mut Self {
        self.include_directories.push(dir.into());
        self
    }

    pub fn includes<P>(&mut self, dirs: P) -> &mut Self
    where
        P: IntoIterator,
        P::Item: Into<PathBuf>,
    {
        for dir in dirs {
            self.include(dir);
        }
        self
    }

    pub fn warnings(&mut self, enable: bool) -> &mut Self {
        self.warnings = enable;
        self
    }

    pub fn warnings_as_errors(&mut self, enable: bool) -> &mut Self {
        self.warnings_as_errors = enable;
        self
    }
}
