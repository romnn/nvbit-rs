use std::ffi::OsStr;
use std::path::{Path, PathBuf};
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
    host_compiler: Option<PathBuf>,
    nvcc_compiler: Option<PathBuf>,
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
            host_compiler: None,
            nvcc_compiler: None,
            warnings: false,
            warnings_as_errors: false,
        }
    }

    fn compile_instrumentation_functions(
        &self,
        nvcc_compiler: &Path,
        include_args: &[String],
        compiler_flags: &[&str],
        objects: &mut Vec<PathBuf>,
    ) -> Result<(), Error> {
        for (i, src) in self.instrumentation_sources.iter().enumerate() {
            let default_name = format!("instr_src_{i}");
            let obj = output_path()
                .join(
                    src.file_name()
                        .and_then(OsStr::to_str)
                        .unwrap_or(&default_name),
                )
                .with_extension("o");
            let mut cmd = Command::new(nvcc_compiler);
            if let Some(host_compiler) = &self.host_compiler {
                cmd.args(["-ccbin", &*host_compiler.to_string_lossy()]);
            }
            cmd.args(include_args);
            cmd.args([
                "-maxrregcount=24",
                "-Xptxas",
                "-astoolspatch",
                "--keep-device-functions",
            ])
            .args(compiler_flags)
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
        Ok(())
    }

    fn compile_sources(
        &self,
        nvcc_compiler: &Path,
        include_args: &[String],
        compiler_flags: &[&str],
        objects: &mut Vec<PathBuf>,
    ) -> Result<(), Error> {
        for (i, src) in self.sources.iter().enumerate() {
            let default_name = format!("src_{i}");
            let obj = output_path()
                .join(
                    src.file_name()
                        .and_then(OsStr::to_str)
                        .unwrap_or(&default_name),
                )
                .with_extension("o");
            let mut cmd = Command::new(nvcc_compiler);
            if let Some(host_compiler) = &self.host_compiler {
                cmd.args(["-ccbin", &*host_compiler.to_string_lossy()]);
            }
            cmd.args(include_args)
                .args(compiler_flags)
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
        Ok(())
    }

    /// Compile and link static library with given name from inputs.
    ///
    /// # Errors
    /// When compilation fails, an error is returned.
    pub fn compile<O: AsRef<str>>(&self, output: O) -> Result<(), Error> {
        let mut objects = self.objects.clone();
        let include_args: Vec<_> = self
            .include_directories
            .iter()
            .map(|d| format!("-I{}", &d.to_string_lossy()))
            .collect();

        let mut compiler_flags = vec!["-Xcompiler", "-fPIC"];
        // compiler_flags.extend(["-Xcompiler", "-Wl,--no-as-needed"]);
        if self.warnings {
            compiler_flags.extend(["-Xcompiler", "-Wall"]);
        }
        if self.warnings_as_errors {
            compiler_flags.extend(["-Xcompiler", "-Werror"]);
        }

        let default_nvcc_compiler = PathBuf::from("nvcc");
        let nvcc_compiler = self
            .nvcc_compiler
            .as_ref()
            .unwrap_or(&default_nvcc_compiler);

        // compile instrumentation functions
        self.compile_instrumentation_functions(
            nvcc_compiler,
            &include_args,
            &compiler_flags,
            &mut objects,
        )?;

        // compile sources
        self.compile_sources(nvcc_compiler, &include_args, &compiler_flags, &mut objects)?;

        // link device functions
        let dev_link_obj = output_path().join("dev_link.o");
        let mut cmd = Command::new(nvcc_compiler);
        if let Some(host_compiler) = &self.host_compiler {
            cmd.args(["-ccbin", &*host_compiler.to_string_lossy()]);
        }

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

    /// Configures the host compiler to be used to produce output.
    pub fn host_compiler<P: Into<PathBuf>>(&mut self, compiler: P) -> &mut Self {
        self.host_compiler = Some(compiler.into());
        self
    }

    /// Configures the host compiler to be used to produce output.
    pub fn nvcc_compiler<P: Into<PathBuf>>(&mut self, compiler: P) -> &mut Self {
        self.nvcc_compiler = Some(compiler.into());
        self
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
