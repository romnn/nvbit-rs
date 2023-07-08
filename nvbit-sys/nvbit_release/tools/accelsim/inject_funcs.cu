/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the trace_t structure */
#include "common.h"

extern "C" __device__ __noinline__ void instrument_mem(
    // uint32_t pred, uint32_t opcode_id, uint64_t addr,
    //            uint64_t grid_launch_id, uint64_t pchannel_dev,
    int pred, int instr_data_width, int instr_opcode_id, int instr_offset,
    int instr_idx, int line_num, int dest_reg, uint64_t src_regs,
    // int src_reg1, int src_reg2,
    // int src_reg3, int src_reg4, int src_reg5,
    int num_src_regs, uint64_t addr, uint64_t pchannel_dev,
    uint64_t kernel_id) {
  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  trace_t t;

  /* collect memory address information from other threads */
  for (int i = 0; i < 32; i++) {
    t.addrs[i] = __shfl_sync(active_mask, addr, i);
  }

  int4 cta = get_ctaid();
  t.grid_launch_id = kernel_id;
  t.cta_id_x = cta.x;
  t.cta_id_y = cta.y;
  t.cta_id_z = cta.z;
  t.warp_id = get_warpid();
  t.opcode_id = instr_opcode_id;

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&t, sizeof(trace_t));
  }
}
