mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(abspath $(dir $(mkfile_path)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_INCLUDE=$(mkfile_dir)/../../nvbit-sys/nvbit_release/core
TRACER_IMPL=$(mkfile_dir)/../../target/release/libaccelsim.a

tracer.so: $(TRACER_IMPL)
	gcc \
		-L$(NVBIT_INCLUDE) \
		-g \
		-Wl,--whole-archive $(TRACER_IMPL) \
		-Wl,--no-whole-archive -lstdc++ -lcuda -lcudart -lcudadevrt \
		-shared -o tracer.so

tool: tracer.so

clean:
	rm -rf build/
	rm -f *.so *.a *.o

.PHONY: tool clean
