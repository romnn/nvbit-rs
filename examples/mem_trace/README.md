#### Dynamic (old name)

should be mem-trace from nvbit examples now

```bash
cargo build --release
LD_PRELOAD=./target/release/libdynamic.so ./nvbit-sys/nvbit_release/test-apps/vectoradd/vectoradd 100
# generates: examples/dynamic/traces/kernelslist
```
