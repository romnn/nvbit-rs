## nvbit-rs

#### TODO
- we must include the CUDA inject funcs? and the `nvbit_tool.h` into the binary somehow.
  - maybe statically compile them in the build script
  - then link them with `nvbit-sys` crate
  - then, `nvbit_at_init` _should_ be called - hopefully

#### Example

The current goal is to get a working example of a tracer written in rust.
Usage should be:
```bash
# create a shared dynamic library that implements the nvbit hooks
cargo build --example tracer
# run the nvbit example CUDA application with the tracer
LD_PRELOAD=./target/debug/examples/libtracer.so ./nvbit_release/test-apps/vectoradd/vectoradd
```

#### Notes

check the clang versions installed
```bash
apt list --installed | grep clang
```

When running `clang nvbit.h`, it also complains about missing cassert.
-std=c++11 
-I$(NVBIT_PATH)

<cassert> is C++ STL, so we need: `clang++ -std=c++11 nvbit.h`.

`bindgen` does not work that well with C++ code, check [this](https://rust-lang.github.io/rust-bindgen/cpp.html).

we need some clang stuff so that bindgen can find `#include <cassert>`.

We will also need to include `nvbit.h`, `nvbit_tool.h`, and tracing injected functions, which require `.cu` files to be compiled and linked with the binary.

[this example](https://github.com/termoshtt/link_cuda_kernel) shows how `.cu` can be compiled and linked with the `cc` crate.

Make sure that the C function hooks of nvbit are not mangled in the shared library:
```bash
nm -D ./target/debug/examples/libtracer.so
```

Make sure that we link statically:
```bash
ldd ./target/debug/examples/libtracer.so
```
