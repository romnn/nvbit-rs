#include "nvbit-sys/nvbit/utils_bridge.h"
#include "nvbit-sys/src/utils.rs.h"

std::unique_ptr<ManagedChannelDev> new_managed_dev_channel() {
  return std::make_unique<ManagedChannelDev>();
}

std::unique_ptr<ChannelDev> new_dev_channel() {
    return std::make_unique<ChannelDev>();
}

std::unique_ptr<ChannelHost> new_host_channel(int id, int buff_size, ChannelDev* ch_dev) {
    auto host_channel = std::make_unique<ChannelHost>();
    host_channel->init(id, buff_size, ch_dev, NULL);
    return host_channel;
}
