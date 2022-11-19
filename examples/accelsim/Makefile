# please:
# 	# rustc -crate-name accelsim --edition=2021 --emit=obj examples/accelsim/src/lib.rs
# 	# ../../target/debug/deps
# 	# clang++ ../../target/debug/deps/accelsim.o ../../target/debug/build/accelsim-a67c1762e4619dad/out/libinstrumentation.so -shared -o please.so

# 	clang++ ../../target/debug/libaccelsim.a
# -L../nvbit_release/core \


mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(abspath $(dir $(mkfile_path)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_INCLUDE=$(mkfile_dir)/../../nvbit-sys/nvbit_release/core
TRACER_IMPL=$(mkfile_dir)/../../target/debug/libaccelsim.a

# build/tracer.a:
# 	mkdir -p build
# 	cargo build -p accelsim
# 	cp ../../target/debug/libaccelsim.a build/tracer.a

tracer.so: $(TRACER_IMPL)
	gcc \
		-L$(NVBIT_INCLUDE) \
		-g \
		-Wl,--whole-archive $(TRACER_IMPL) \
		-Wl,--no-whole-archive -lstdc++ -lnvbit -lcuda -lcudart -lcudadevrt \
		-shared -o tracer.so

tool: tracer.so

clean:
	rm -rf build/
	rm -f *.so *.a *.o

.PHONY: tool clean
