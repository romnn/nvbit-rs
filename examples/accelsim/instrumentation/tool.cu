// every nvbit tool must include this once to initialize tracing
#include "nvbit_tool.h"

#include "utils/channel.hpp"

#include "common.h"

__global__ __noinline__ void flush_channel_kernel(void *channel_dev) {
  ChannelDev *channel = (ChannelDev *)channel_dev;
  printf("flush_channel\n");
  // push memory access with negative cta id to communicate
  // the kernel is completed
  inst_trace_t t;
  t.cta_id_x = -1;
  channel->push(&t, sizeof(inst_trace_t));

  // flush channel
  channel->flush();
}

extern "C" __noinline__ void flush_channel(void *channel_dev) {
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  // issue flush of channel so we are sure all the memory accesses
  // have been pushed
  flush_channel_kernel<<<1, 1>>>(channel_dev);
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);
}
