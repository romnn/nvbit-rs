#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

// does not work
/* #include "nvbit_tool.h" */

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
    /* int pred, int opcode_id, int32_t vpc, bool is_mem, uint64_t addr, */
    /* int32_t width, int32_t desReg, int32_t srcReg1, int32_t srcReg2, */
    /* int32_t srcReg3, int32_t srcReg4, int32_t srcReg5, int32_t srcNum, */
    /* uint64_t pchannel_dev, uint64_t ptotal_dynamic_instr_counter, */
    /* uint64_t preported_dynamic_instr_counter, uint64_t pstop_report */
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
    /* ChannelDev* channel_dev = (ChannelDev*)pchannel_dev; */
    /* channel_devn->push(&ma, sizeof(mem_access_t)); */
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_access_t));
  }

  /* const int active_mask = __ballot_sync(__activemask(), 1); */
  /* const int predicate_mask = __ballot_sync(__activemask(), pred); */
  /* const int laneid = get_laneid(); */
  /* const int first_laneid = __ffs(active_mask) - 1; */

  /* if ((*((bool *)pstop_report))) { */
  /*   if (first_laneid == laneid) { */
  /*     atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1); */
  /*     return; */
  /*   } */
  /* } */

  /* inst_trace_t ma; */

  /* if (is_mem) { */
  /*   // collect memory address information */
  /*   for (int i = 0; i < 32; i++) { */
  /*     ma.addrs[i] = __shfl_sync(active_mask, addr, i); */
  /*   } */
  /*   ma.width = width; */
  /*   ma.is_mem = true; */
  /* } else { */
  /*   ma.is_mem = false; */
  /* } */

  /* int4 cta = get_ctaid(); */
  /* int unique_thread_id = threadIdx.z * blockDim.y * blockDim.x + */
  /*                        threadIdx.y * blockDim.x + threadIdx.x; */
  /* ma.warpid_tb = unique_thread_id / 32; */

  /* ma.cta_id_x = cta.x; */
  /* ma.cta_id_y = cta.y; */
  /* ma.cta_id_z = cta.z; */
  /* ma.warpid_sm = get_warpid(); */
  /* ma.opcode_id = opcode_id; */
  /* ma.vpc = vpc; */
  /* ma.GPRDst = desReg; */
  /* ma.GPRSrcs[0] = srcReg1; */
  /* ma.GPRSrcs[1] = srcReg2; */
  /* ma.GPRSrcs[2] = srcReg3; */
  /* ma.GPRSrcs[3] = srcReg4; */
  /* ma.GPRSrcs[4] = srcReg5; */
  /* ma.numSrcs = srcNum; */
  /* ma.active_mask = active_mask; */
  /* ma.predicate_mask = predicate_mask; */
  /* ma.sm_id = get_smid(); */

  /* // first active lane pushes information on the channel */
  /* if (first_laneid == laneid) { */
  /*   ChannelDev *channel_dev = (ChannelDev *)pchannel_dev; */
  /*   channel_dev->push(&ma, sizeof(inst_trace_t)); */
  /* } */
}
