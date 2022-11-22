#[cxx::bridge]
mod ffi {
    extern "Rust" {}

    unsafe extern "C++" {
        include!("nvbit-sys/nvbit/utils_bridge.h");

        type c_void;
        type ManagedChannelDev;
        type ChannelDev;
        type ChannelHost;

        /// Allocate and initialize a new device channel.
        ///
        /// The channel can be accessed from the host only.
        #[must_use]
        fn new_dev_channel() -> UniquePtr<ChannelDev>;

        /// Allocate and initialize a new device channel in managed memory.
        ///
        /// The channel can be accessed from both the host and device.
        #[must_use]
        fn new_managed_dev_channel() -> UniquePtr<ManagedChannelDev>;

        /// Allocate and initialize a new host channel.
        ///
        /// The channel can be accessed from the host only.
        ///
        /// # Safety
        /// The user must ensure that `channel_dev` points to a valid 
        /// `ChannelDev` instance in host or managed memory.
        #[must_use]
        unsafe fn new_host_channel(
            id: i32,
            buff_size: i32,
            channel_dev: *mut ChannelDev,
        ) -> UniquePtr<ChannelHost>;

        /// Receive from channel, filling `buff` up to at most `max_buff_size`.
        ///
        /// The number of bytes received are returned.
        ///
        /// # Safety
        ///
        /// - `buff` mut point point to a buffer of
        ///   (unitialized) contiguous, heap allocated memory.
        /// - Size of `buff` buffer must be greater or equal to `max_buff_size`.
        /// - User should not read a number of bytes from buffer `buff` that exceeds
        ///   the number returned by this function per call.
        #[must_use]
        unsafe fn recv(self: Pin<&mut ChannelHost>, buff: *mut c_void, max_buff_size: u32) -> u32;
    }
}

pub use ffi::*;

unsafe impl Send for ffi::ChannelHost {}
unsafe impl Sync for ffi::ChannelHost {}

unsafe impl Send for ffi::ChannelDev {}
unsafe impl Sync for ffi::ChannelDev {}
