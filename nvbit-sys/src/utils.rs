#[cxx::bridge]
mod ffi {
    extern "Rust" {}

    unsafe extern "C++" {
        include!("nvbit-sys/nvbit/utils.h");
        // include!("nvbit-sys/nvbit/utils_bridge.cuh");
        // include!("nvbit-sys/nvbit/utils_bridge.h");

        type c_void;

        type ChannelDev;
        type ChannelHost;

        // fn new_host_channel() -> UniquePtr<ChannelHost>;

        fn new_dev_channel() -> UniquePtr<ChannelDev>;
        fn get_id(self: Pin<&mut ChannelDev>) -> i32;
        fn dev_channel_size() -> usize;

        // unsafe fn new_managed_dev_channel() -> UniquePtr<ChannelDev>;
        unsafe fn new_managed_dev_channel() -> *mut ChannelDev;

        unsafe fn new_host_channel(
            id: i32,
            buff_size: i32,
            channel_dev: *mut ChannelDev,
        ) -> UniquePtr<ChannelHost>;

        fn is_active(self: Pin<&mut ChannelHost>) -> bool;

        unsafe fn recv(self: Pin<&mut ChannelHost>, buff: *mut c_void, max_buff_size: u32) -> u32;
    }
}

pub use ffi::*;
