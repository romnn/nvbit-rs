#include <stdint.h>

extern "C" void flush_channel(void *channel_dev);

typedef struct {
  uint64_t grid_launch_id;
  // use int here, so we identify end of the kernel with `cta_idx_id = -1`
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;
  // opcode_id is purely internal
  uint32_t opcode_id;
  // addr per thread of a warp?
  uint64_t addrs[32];
} mem_access_t;
