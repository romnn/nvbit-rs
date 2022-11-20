#pragma once
#include "rust/cxx.h"

#include "utils/channel.hpp"
#include "utils/utils.h"
/* #include "nvbit-sys/nvbit_release/core/utils/channel.hpp" */
/* #include "nvbit-sys/nvbit_release/core/utils/utils.h" */

#include <memory> // for std::unique_ptr

using c_void = void;

class ManagedChannelDev : public Managed, public ChannelDev {};

std::unique_ptr<ManagedChannelDev> new_managed_dev_channel();
std::unique_ptr<ChannelDev> new_dev_channel();

std::unique_ptr<ChannelHost> new_host_channel(int id, int buff_size,
                                              ChannelDev *ch_dev);
