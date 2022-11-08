#include "nvbit.h"
#include "nvbit_tool.h"

extern "C" __noinline__ void init_nvbit() {}

extern "C" __noinline__ int this_must_be_present() {
  return 42;
}
