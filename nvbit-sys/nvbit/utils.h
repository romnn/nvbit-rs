#pragma once
// contains definitions such as a `rust::Vec`.
#include "rust/cxx.h"

/* #include "nvbit-sys/nvbit/utils.h" */
/* #include <stdint.h> */
/* #include "cuda.h" */
/* #include "cuda_runtime_api.h" */
/* #include "cuda_runtime.h" */
/* #include "nvbit-sys/nvbit_release/core/cuda.h" */

#include "nvbit-sys/nvbit_release/core/utils/utils.h"
#include "nvbit-sys/nvbit_release/core/utils/channel.hpp"

#include <memory> // std::unique_ptr

using c_void = void;

size_t dev_channel_size();

ChannelDev* new_managed_dev_channel();
std::unique_ptr<ChannelDev> new_dev_channel();

std::unique_ptr<ChannelHost> new_host_channel(int id, int buff_size, ChannelDev* ch_dev);
/* std::unique_ptr<ChannelHost> new_host_channel(); */
