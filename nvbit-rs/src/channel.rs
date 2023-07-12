use nvbit_sys::utils::{ChannelHost, ManagedChannelDev};
use std::marker::PhantomData;
use std::sync::{atomic, mpsc, Arc, Mutex};

/// A device channel in managed memory.
///
/// This channel can be accessed by the host and device.
#[derive()]
pub struct Device<T> {
    inner: cxx::UniquePtr<ManagedChannelDev>,
    packet: PhantomData<T>,
}

unsafe impl<T> Send for Device<T> {}

impl<T> Default for Device<T> {
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Device<T> {
    /// Creates a new managed device channel.
    #[must_use]
    pub fn new() -> Self {
        let inner = nvbit_sys::utils::new_managed_dev_channel();
        Self {
            inner,
            packet: PhantomData,
        }
    }

    /// Converts a `Device<T>` to a raw pointer.
    ///
    /// # Panics
    /// Panics if the owned channel [`cxx::UniquePtr<nvbit_sys::ManagedChannelDev>`]
    /// does not exist.
    #[must_use]
    pub fn as_ptr(&self) -> *const ManagedChannelDev {
        self.inner.as_ref().unwrap() as *const _
    }

    /// Converts a `Device<T>` to a raw pointer.
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

struct HostInner<T> {
    channel: cxx::UniquePtr<ChannelHost>,
    buffer: super::buffer::Buffer,
    packet: PhantomData<T>,
}

impl<T> HostInner<T>
where
    T: Send + 'static,
{
    pub fn read(&mut self, tx: &mpsc::Sender<T>, shutdown: &Arc<atomic::AtomicBool>) {
        let packet_size = std::mem::size_of::<T>();
        while !shutdown.load(atomic::Ordering::Relaxed) {
            let buffer_ptr = self.buffer.as_mut_ptr();
            let buffer_size = u32::try_from(self.buffer.len()).unwrap_or(u32::MAX);

            let num_recv_bytes =
                unsafe { self.channel.pin_mut().recv(buffer_ptr.cast(), buffer_size) };
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
pub enum HostError {
    #[error(transparent)]
    Buffer(#[from] super::buffer::Error),
}

/// A host channel.
///
/// This channel can be accessed only by the host.
#[derive()]
pub struct Host<T>
where
    T: Send + 'static,
{
    inner: Arc<Mutex<HostInner<T>>>,
    receiver_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<atomic::AtomicBool>,
}

impl<T> Host<T>
where
    T: Send + 'static,
{
    /// Creates a new host channel.
    ///
    /// # Errors
    /// Returns an error if a buffer of size `buffer_size` cannot be allocated.
    pub fn new(id: i32, buffer_size: u32, dev_channel: &mut Device<T>) -> Result<Self, HostError> {
        let channel = unsafe {
            nvbit_sys::utils::new_host_channel(
                id,
                buffer_size.try_into().unwrap_or(i32::MAX),
                dev_channel.as_mut_ptr().cast(),
            )
        };
        let buffer = super::buffer::Buffer::new(buffer_size as usize)?;
        let inner = HostInner {
            channel,
            buffer,
            packet: PhantomData,
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
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
        match self.receiver_thread.take() {
            Some(thread) => {
                self.shutdown.store(true, atomic::Ordering::Relaxed);

                // wait for the receiver thread to complete
                thread.join()
            }
            _ => Ok(()),
        }
    }
}

impl<T> Drop for Host<T>
where
    T: Send + 'static,
{
    fn drop(&mut self) {
        self.stop().expect("stop host channel");
    }
}
