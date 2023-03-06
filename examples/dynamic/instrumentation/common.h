#include <stdint.h>

extern "C" void flush_channel(void *channel_dev);

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5

typedef struct {
  uint64_t grid_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;
  int opcode_id;
  uint64_t addrs[32];

  /* int cta_id_x; */
  /* int cta_id_y; */
  /* int cta_id_z; */
  /* int warpid_tb; */
  /* int warpid_sm; */
  /* int sm_id; */
  /* int opcode_id; */
  /* uint64_t addrs[32]; */
  /* uint32_t vpc; */
  /* bool is_mem; */
  /* int32_t GPRDst; */
  /* int32_t GPRSrcs[MAX_SRC]; */
  /* int32_t numSrcs; */
  /* int32_t width; */
  /* uint32_t active_mask; */
  /* uint32_t predicate_mask; */
} mem_access_t;
