use std::path::PathBuf;
use std::process::{Command, Output};

#[inline]
#[must_use]
pub fn nvbit_include() -> PathBuf {
    PathBuf::from(std::env::var("DEP_NVBIT_INCLUDE").expect("nvbit include path"))
        .canonicalize()
        .expect("canonicalize path")
}

#[inline]
#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize path")
}

#[inline]
#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir"))
        .canonicalize()
        .expect("canonicalize path")
}

#[derive(Debug, Clone)]
pub struct Build {
    include_directories: Vec<PathBuf>,
    objects: Vec<PathBuf>,
    sources: Vec<PathBuf>,
    instrumentation_sources: Vec<PathBuf>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("Command failed")]
    Command(Output),
}

impl Build {
    #[inline]
    pub fn new() -> Self {
        Self {
            include_directories: Vec::new(),
            objects: Vec::new(),
            sources: Vec::new(),
            instrumentation_sources: Vec::new(),
        }
    }

    pub fn compile<O: AsRef<str>>(&self, output: O) -> Result<(), Error> {
        use std::ffi::OsStr;

        let mut objects = self.objects.clone();
        let include_args: Vec<_> = self
            .include_directories
            .iter()
            .map(|d| format!("-I{}", &d.to_string_lossy()))
            .collect();

        // compile instrumentation functions
        for instrumentation_src in &self.instrumentation_sources {
            let obj = output_path()
                .join(
                    instrumentation_src
                        .file_name()
                        .and_then(OsStr::to_str)
                        .unwrap(),
                )
                .with_extension("o");
            let result = Command::new("nvcc")
                .args(&include_args)
                .args([
                    "-maxrregcount=24",
                    "-Xptxas",
                    "-astoolspatch",
                    "--keep-device-functions",
                    "-Xcompiler",
                    "-fPIC",
                    "-c",
                ])
                .arg(instrumentation_src)
                .arg("-o")
                .arg(&*obj.to_string_lossy())
                .output()?;
            if !result.status.success() {
                return Err(Error::Command(result));
            }
            objects.push(obj);
        }

        // compile sources
        for src in &self.sources {
            let obj = output_path()
                .join(src.file_name().and_then(OsStr::to_str).unwrap())
                .with_extension("o");
            let result = Command::new("nvcc")
                .args(&include_args)
                .args(["-Xcompiler", "-fPIC", "-dc", "-c"])
                .arg(src)
                .arg("-o")
                .arg(&*obj.to_string_lossy())
                .output()?;
            if !result.status.success() {
                return Err(Error::Command(result));
            }
            objects.push(obj);
        }

        // link device functions
        let dev_link_obj = output_path().join("dev_link.o");
        let result = Command::new("nvcc")
            .args(&include_args)
            .args(["-Xcompiler", "-fPIC", "-dlink"])
            .args(&objects)
            .arg("-o")
            .arg(&*dev_link_obj.to_string_lossy())
            .output()?;
        if !result.status.success() {
            return Err(Error::Command(result));
        }
        objects.push(dev_link_obj);

        // link everything together
        let result = Command::new("ar")
            .args([
                "cru",
                &output_path()
                    .join(format!("lib{}.a", output.as_ref()))
                    .to_string_lossy(),
            ])
            .args(&objects)
            .output()?;
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
}
