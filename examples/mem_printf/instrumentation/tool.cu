// every nvbit tool must include this once to initialize tracing
#include "nvbit_tool.h"

// force including
extern "C" __noinline__ void noop() {
  printf("hi from c\n");
}
