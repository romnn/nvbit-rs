#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

#include "common.h"

// Instrumentation function that we want to inject.
// Please note the use of extern "C" __device__ __noinline__
// to prevent "dead"-code elimination by the compiler.
extern "C" __device__ __noinline__ void instrument_inst(int pred, int opcode_id,
                                                        uint64_t addr) {
  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }
  /* uint64_t addr = 0; */
  /* uint64_t addr = 0; */
  printf(" 0x%016lx - opcode_id %d\n", addr, opcode_id);
}
