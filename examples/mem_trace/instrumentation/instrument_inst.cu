#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

// contains definition of the mem_access_t structure
#include "common.h"

// Instrumentation function that we want to inject.
// Please note the use of extern "C" __device__ __noinline__
// to prevent "dead"-code elimination by the compiler.
extern "C" __device__ __noinline__ void instrument_inst(
    int pred, int opcode_id, uint64_t addr, uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {

  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_access_t ma;

  /* collect memory address information from other threads */
  for (int i = 0; i < 32; i++) {
    ma.addrs[i] = __shfl_sync(active_mask, addr, i);
  }

  int4 cta = get_ctaid();
  ma.grid_launch_id = grid_launch_id;
  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warp_id = get_warpid();
  ma.opcode_id = opcode_id;

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_access_t));
  }
}
