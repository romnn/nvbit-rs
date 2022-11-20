## nvbit-rs

#### TODO:

- generate less bindings?
- finish safe abstractions in `nvbit-rs`
- write a better README.md
- rename project on github
- publish to crates.io

- write to a file using a compressed stream and `serde-cbor` for example
- migrate other examples to make the api really usable

#### DONE:
- use a single static for the accelsim example with all the context to avoid unsafe
- use the vendor `nvbit_release` only for downstream crates
- the read thread should push into a rust channel
- clean up some code
- alias types used for the nvbit callback API's
- improve the buffer implementation
- include `cuda-sys` crate in nvbit-rs

```bash
nvcc -D_FORCE_INLINES -dc -c -std=c++11 -I../nvbit_release/core -Xptxas -cloning=no -Xcompiler -w  -O3 -Xcompiler -fPIC tracer_tool.cu -o tracer_tool.o

nvcc -D_FORCE_INLINES -I../nvbit_release/core -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -c inject_funcs.cu -o inject_funcs.o

nvcc -D_FORCE_INLINES -O3 tracer_tool.o inject_funcs.o -L../nvbit_release/core -lnvbit -lcuda -shared -o tracer_tool.so
```

now is a good time to introduce workspaces
make the examples individual crates with cargo.toml and build.rs
write the custom tracing kernels per example 
this way we might finally include the symbol


#### TODO - we find that Rust and C++ interop is hard - e.g. `nvbit_get_related_functions` returns `std::vector<CUfunction>`, for which there is no easy binding, even using `&cxx::CxxVector<CUfunction>` does not work because `CUfunction` is a FFI struct (by value).
  - a possible way is to provide a wrapper that copies to a `cxx::Vec<CUfuncton>` i guess (see [this example](https://github.com/dtolnay/cxx/blob/master/book/src/binding/vec.md#example))
  - since we are tracing, and this would need to be performed for each unseen function, this copy overhead is not acceptable
    - TODO: find out how often it is called and maybe still do it and measure
      - UPDATE: get_related_functions is only called once, try it in rust

- other approach: only receive stuff from the channel, a simple struct...
  - if that works: how can we decide which tracing function to use
    - (since we cannot write new ones in rust)

- figure out if we can somehow reuse the same nvbit names by using a namespace??
- or wrap the calls in rust which calls the `ffi::rust_*` funcs.

- IMPORTANT OBSERVATION:
  - i almost gave up on `cxx`, because it was only giving me `Unsupported type` errors
  - i dont import `cxx::UniquePtr` or `cxx::CxxVector` in the `ffi` module, 
    so i was assuming i need to use `cxx::` to reference the types.
  - they dont in the docs, but use them in the top level module ...
  - turns out you *need* to omit the `cxx::` prefix because this is all macro magic ...

#### Done
- we must include the CUDA inject funcs? and the `nvbit_tool.h` into the binary somehow.
  - maybe statically compile them in the build script
  - then link them with `nvbit-sys` crate
  - then, `nvbit_at_init` _should_ be called - hopefully

#### Example

The current goal is to get a working example of a tracer written in rust.
Usage should be:
```bash
# install lld
sudo apt-get install -y lld
# create a shared dynamic library that implements the nvbit hooks
cargo build -p accelsim
# run the nvbit example CUDA application with the tracer
LD_PRELOAD=./target/debug/libaccelsim.so nvbit-sys/nvbit_release/test-apps/vectoradd/vectoradd
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
nm -D ./target/debug/build/nvbit-sys-08fdef510bde07a0/out/libinstrumentation.a
```

Problem: we need the `instrument_inst` function to be present in the binary, just like
for the example:
```bash
nm -D /home/roman/dev/nvbit-sys/tracer_nvbit/tracer_tool/tracer_tool.so | grep instrument

# for a static library:
nm --debug-syms target/debug/build/accelsim-a67c1762e4619dad/out/libinstrumentation.a | grep instrument
```
Currently, its not :(

Make sure that we link statically:
```bash
ldd ./target/debug/examples/libtracer.so
```

Check what includes `cxx` generated:
```bash
tre target/debug/build/nvbit-sys-*/out/cxxbridge/
```
