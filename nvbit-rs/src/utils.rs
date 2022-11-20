use nvbit_sys::utils::{ChannelHost, ManagedChannelDev};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{atomic, mpsc, Arc, Mutex};

#[derive()]
pub struct DeviceChannel<T> {
    inner: cxx::UniquePtr<ManagedChannelDev>,
    packet: PhantomData<T>,
}

unsafe impl<T> Send for DeviceChannel<T> {}

impl<T> DeviceChannel<T> {
    pub fn new() -> Self {
        let inner = unsafe { nvbit_sys::utils::new_managed_dev_channel() };
        Self {
            inner,
            packet: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const ManagedChannelDev {
        self.inner.as_ref().unwrap() as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut ManagedChannelDev {
        let channel = unsafe { self.inner.as_mut().unwrap().get_unchecked_mut() };
        channel as *mut _
    }
}

#[derive()]
struct HostChannelInner<T> {
    channel: cxx::UniquePtr<nvbit_sys::utils::ChannelHost>,
    buffer: super::buffer::Buffer,
    packet: PhantomData<T>,
}

impl<T> HostChannelInner<T>
where
    T: Send + 'static,
{
    pub fn read(&mut self, tx: mpsc::Sender<T>, shutdown: Arc<atomic::AtomicBool>) {
        let packet_size = std::mem::size_of::<T>();
        while !shutdown.load(atomic::Ordering::Relaxed) {
            let buffer_ptr = self.buffer.as_mut_ptr();
            let buffer_size = self.buffer.len();

            let num_recv_bytes = unsafe {
                self.channel.pin_mut().recv(
                    buffer_ptr as *mut _, // nvbit_sys::utils::c_void,
                    buffer_size as u32,
                )
            };
            if num_recv_bytes > 0 {
                let mut num_processed_bytes: usize = 0;
                while num_processed_bytes < num_recv_bytes as usize {
                    let packet_bytes =
                        &self.buffer[num_processed_bytes..num_processed_bytes + packet_size];
                    assert_eq!(packet_size, packet_bytes.len());
                    let packet: T = unsafe { std::ptr::read(packet_bytes.as_ptr() as *const _) };
                    tx.send(packet);
                    num_processed_bytes += packet_size;
                }
            }
        }
    }
}

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
    pub fn new(id: usize, buffer_size: usize, dev_channel: &mut DeviceChannel<T>) -> Self {
        let channel = unsafe {
            nvbit_sys::utils::new_host_channel(
                id as i32,
                buffer_size as i32,
                dev_channel.as_mut_ptr() as *mut _,
            )
        };
        let buffer = super::buffer::Buffer::new(buffer_size);
        let inner = Arc::new(Mutex::new(HostChannelInner {
            channel,
            buffer,
            packet: PhantomData,
        }));
        Self {
            inner,
            shutdown: Arc::new(atomic::AtomicBool::new(false)),
            receiver_thread: None,
        }
    }

    /// Read from the channel
    pub fn read(&mut self) -> mpsc::Receiver<T> {
        let (tx, rx) = mpsc::channel();
        let inner = self.inner.clone();
        let shutdown = self.shutdown.clone();
        self.receiver_thread = Some(std::thread::spawn(move || {
            let mut lock = inner.lock().unwrap();
            lock.read(tx, shutdown);
        }));
        rx
    }

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
        self.stop();
    }
}
