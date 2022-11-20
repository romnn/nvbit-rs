#[cxx::bridge]
mod ffi {
    extern "Rust" {}

    unsafe extern "C++" {
        include!("nvbit-sys/nvbit/utils_bridge.h");

        type c_void;
        type ManagedChannelDev;
        type ChannelDev;
        type ChannelHost;

        unsafe fn new_dev_channel() -> UniquePtr<ChannelDev>;
        unsafe fn new_managed_dev_channel() -> UniquePtr<ManagedChannelDev>;

        unsafe fn new_host_channel(
            id: i32,
            buff_size: i32,
            channel_dev: *mut ChannelDev,
        ) -> UniquePtr<ChannelHost>;

        unsafe fn recv(self: Pin<&mut ChannelHost>, buff: *mut c_void, max_buff_size: u32) -> u32;
    }
}

pub use ffi::*;

unsafe impl Send for ffi::ChannelHost {}
unsafe impl Sync for ffi::ChannelHost {}

unsafe impl Send for ffi::ChannelDev {}
unsafe impl Sync for ffi::ChannelDev {}
