// every nvbit tool must include this once to initialize tracing
#include "nvbit_tool.h"

#include "utils/channel.hpp"

#include "common.h"

__global__ __noinline__ void flush_channel_kernel(ChannelDev *ch_dev) {
  /* set a CTA id = -1 to indicate communication thread that this is the
   * termination flag */
  mem_access_t ma;
  ma.cta_id_x = -1;
  ch_dev->push(&ma, sizeof(mem_access_t));
  ch_dev->flush();
}

extern "C" __noinline__ void flush_channel(void *channel_dev) {
  flush_channel_kernel<<<1, 1>>>((ChannelDev*)channel_dev);
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);
}

/* __global__ __noinline__ void flush_channel_kernel(void *channel_dev) { */
/*   ChannelDev* channel = (ChannelDev*)channel_dev; */
/*   printf("flush_channel\n"); */
/*   // push memory access with negative cta id to communicate */
/*   // the kernel is completed */
/*   inst_trace_t ma; */
/*   ma.cta_id_x = -1; */
/*   channel->push(&ma, sizeof(inst_trace_t)); */

/*   // flush channel */
/*   channel->flush(); */
/* } */

/* extern "C" __noinline__ void flush_channel(void *channel_dev) { */
/*   cudaDeviceSynchronize(); */
/*   assert(cudaGetLastError() == cudaSuccess); */

/*   // issue flush of channel so we are sure all the memory accesses */
/*   // have been pushed */
/*   flush_channel_kernel<<<1, 1>>>(channel_dev); */
/*   cudaDeviceSynchronize(); */
/*   assert(cudaGetLastError() == cudaSuccess); */
/* } */
