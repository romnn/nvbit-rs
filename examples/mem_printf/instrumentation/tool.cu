// every nvbit tool must include this once to initialize tracing
#include "nvbit_tool.h"

// force including
__global__ __noinline__ void dev_noop() {}

// force including
extern "C" __noinline__ void noop() {
  printf("hi from c\n");
  dev_noop<<<1, 1>>>();
}
