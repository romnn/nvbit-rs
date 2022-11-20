#include "nvbit-sys/nvbit/utils.h"

/* #include "nvbit-sys/nvbit_release/core/utils/utils.h" */
/* #include "nvbit-sys/nvbit_release/core/utils/channel.hpp" */

#include "nvbit-sys/src/utils.rs.h"

/* void ChannelHost::custom_init() const { */
/*   this->init(0, 0, NULL, NULL); */
/* } */

/* size_t dev_channel_size() { */
/*     return sizeof(ChannelDev); */
/* } */

/* class ManagedChannelDev: public Managed, public ChannelDev { */
  /* private: */
  /*   ChannelDev *channel; */

  /* public: */
  /*   void *operator new() { */
  /*       CUDA_SAFECALL(cudaMallocManaged(&this.channel, len)); */
  /*       new (this.channel) ChannelDev(); */
  /*   } */

  /*   // void Managed::operator delete(void *ptr) */
  /*   void operator delete(void *ptr) { cudaFree(ptr); } */

  /*   void *operator new[](size_t len) { */
  /*       void *ptr; */
  /*       cudaMallocManaged(&ptr, len); */
  /*       return ptr; */
  /*   } */
  /*   // void Managed::operator delete[] (void* ptr) { */
  /*   void operator delete[](void *ptr) { cudaFree(ptr); } */
/* }; */

std::unique_ptr<ManagedChannelDev> new_managed_dev_channel() {
/* ChannelDev* new_managed_dev_channel() { */
  /* ChannelDev* channel_dev; */
  /* CUDA_SAFECALL(cudaMallocManaged(&channel_dev, sizeof(ChannelDev))); */
  /* new (channel_dev) ChannelDev(); */
  /* /1* return channel_dev; *1/ */
  return std::make_unique<ManagedChannelDev>();
  /* return std::make_unique<ChannelDev>(channel_dev); */
}

std::unique_ptr<ChannelDev> new_dev_channel() {
    return std::make_unique<ChannelDev>();
}

std::unique_ptr<ChannelHost> new_host_channel(int id, int buff_size, ChannelDev* ch_dev) {
    auto host_channel = std::make_unique<ChannelHost>();
    host_channel->init(id, buff_size, ch_dev, NULL);
    return host_channel;
}
