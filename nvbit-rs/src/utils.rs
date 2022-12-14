use nvbit_sys::utils::{ChannelHost, ManagedChannelDev};
use std::marker::PhantomData;
use std::sync::{atomic, mpsc, Arc, Mutex};

/// A device channel in managed memory.
///
/// This channel can be accessed by the host and device.
#[derive()]
pub struct DeviceChannel<T> {
    inner: cxx::UniquePtr<ManagedChannelDev>,
    packet: PhantomData<T>,
}

unsafe impl<T> Send for DeviceChannel<T> {}

impl<T> Default for DeviceChannel<T> {
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DeviceChannel<T> {
    /// Creates a new managed device channel.
    #[must_use]
    pub fn new() -> Self {
        let inner = nvbit_sys::utils::new_managed_dev_channel();
        Self {
            inner,
            packet: PhantomData,
        }
    }

    /// Converts a `DeviceChannel<T>` to a raw pointer.
    ///
    /// # Panics
    /// Panics if the owned channel [`cxx::UniquePtr<nvbit_sys::ManagedChannelDev>`]
    /// does not exist.
    #[must_use]
    pub fn as_ptr(&self) -> *const ManagedChannelDev {
        self.inner.as_ref().unwrap() as *const _
    }

    /// Converts a `DeviceChannel<T>` to a raw pointer.
    ///
    /// # Panics
    /// Panics if the owned channel [`cxx::UniquePtr<nvbit_sys::ManagedChannelDev>`]
    /// does not exist.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut ManagedChannelDev {
        let channel = unsafe { self.inner.as_mut().unwrap().get_unchecked_mut() };
        channel as *mut _
    }
}

#[derive()]
struct HostChannelInner<T> {
    channel: cxx::UniquePtr<ChannelHost>,
    buffer: super::buffer::Buffer,
    packet: PhantomData<T>,
}

impl<T> HostChannelInner<T>
where
    T: Send + 'static,
{
    pub fn read(&mut self, tx: &mpsc::Sender<T>, shutdown: &Arc<atomic::AtomicBool>) {
        let packet_size = std::mem::size_of::<T>();
        while !shutdown.load(atomic::Ordering::Relaxed) {
            let buffer_ptr = self.buffer.as_mut_ptr();
            let buffer_size = self.buffer.len();

            assert!(u32::try_from(buffer_size).is_ok());
            let num_recv_bytes = unsafe {
                self.channel
                    .pin_mut()
                    .recv(buffer_ptr.cast(), buffer_size.try_into().unwrap())
            };
            if num_recv_bytes > 0 {
                let mut num_processed_bytes: usize = 0;
                while num_processed_bytes < num_recv_bytes as usize {
                    let packet_bytes =
                        &self.buffer[num_processed_bytes..num_processed_bytes + packet_size];
                    assert_eq!(packet_size, packet_bytes.len());
                    let packet: T = unsafe { std::ptr::read(packet_bytes.as_ptr().cast()) };
                    tx.send(packet).expect("send packet");
                    num_processed_bytes += packet_size;
                }
            }
        }
    }
}

/// Errors that can occur when using an NVBIT managed channel.
#[derive(thiserror::Error, Debug)]
pub enum HostChannelError {
    #[error("failed to create buffer")]
    Buffer(
        #[from]
        #[source]
        super::buffer::Error,
    ),
}

/// A host channel.
///
/// This channel can be accessed only by the host.
#[derive()]
pub struct HostChannel<T>
where
    T: Send + 'static,
{
    inner: Arc<Mutex<HostChannelInner<T>>>,
    receiver_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<atomic::AtomicBool>,
}

impl<T> HostChannel<T>
where
    T: Send + 'static,
{
    /// Creates a new host channel.
    ///
    /// # Errors
    /// Returns an error if a buffer of size `buffer_size` cannot be allocated.
    ///
    /// # Panics
    /// * Panics if the buffer size is larger than `i32::MAX`.
    pub fn new(
        id: i32,
        buffer_size: usize,
        dev_channel: &mut DeviceChannel<T>,
    ) -> Result<Self, HostChannelError> {
        assert!(i32::try_from(buffer_size).is_ok());
        let channel = unsafe {
            nvbit_sys::utils::new_host_channel(
                id,
                buffer_size.try_into().unwrap(),
                dev_channel.as_mut_ptr().cast(),
            )
        };
        let buffer = super::buffer::Buffer::new(buffer_size)?;
        let inner = Arc::new(Mutex::new(HostChannelInner {
            channel,
            buffer,
            packet: PhantomData,
        }));
        Ok(Self {
            inner,
            shutdown: Arc::new(atomic::AtomicBool::new(false)),
            receiver_thread: None,
        })
    }

    /// Starts reading from the channel.
    ///
    /// This function is non-blocking and spawns a new thread to read from the channel
    /// in the background.
    /// Received packets of type `T` are pushed into a channel which can be read
    /// using the [`std::sync::mpsc::Receiver`] returned.
    ///
    /// # Panics
    /// Panics if the receiver thread cannot be created.
    ///
    #[must_use]
    pub fn read(&mut self) -> mpsc::Receiver<T> {
        let (tx, rx) = mpsc::channel();
        let inner = self.inner.clone();
        let shutdown = self.shutdown.clone();

        self.receiver_thread = Some(std::thread::spawn(move || {
            let mut lock = inner.lock().unwrap();
            lock.read(&tx, &shutdown);
        }));
        rx
    }

    /// Stops reading from the channel.
    ///
    /// Sends shutdown signal to the receiving thread and waits for it to exit.
    ///
    /// # Errors
    /// If the receiving thread panicked, a [`std::thread::Result`] is returned
    /// containing the boxed parameter given to `panic!`.
    pub fn stop(&mut self) -> std::thread::Result<()> {
        println!("stopping receiver thread");
        self.shutdown.store(true, atomic::Ordering::Relaxed);
        // wait for the receiver thread to complete
        match self.receiver_thread.take() {
            Some(thread) => thread.join(),
            _ => Ok(()),
        }
    }
}

impl<T> Drop for HostChannel<T>
where
    T: Send + 'static,
{
    fn drop(&mut self) {
        self.stop().expect("stop host channel");
    }
}
