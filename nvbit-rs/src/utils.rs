use nvbit_sys::utils::{ChannelHost, ManagedChannelDev};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{atomic, mpsc, Arc, Mutex};
// use std::sync::mpsc::{channel, Receiver, Sender};

// share stuff between managed and host alloc channeld dev?
// trait DevChannel {}
// nvbit_sys::utils::ManagedChannelDev

#[derive()]
pub struct DeviceChannel<T> {
    inner: cxx::UniquePtr<ManagedChannelDev>,
    packet: PhantomData<T>,
}

unsafe impl<T> Send for DeviceChannel<T> {}

impl<T> DeviceChannel<T> {
    // pub fn new() -> Self {
    //     let inner = unsafe { nvbit_sys::utils::new_dev_channel() };
    //     Self { inner }
    // }

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
    // running:
}

impl<T> HostChannelInner<T>
where
    // T: Send + std::fmt::Debug + 'static,
    T: Send + 'static,
{
    pub fn stop(&self) {}

    pub fn read(&mut self, tx: mpsc::Sender<T>, shutdown: Arc<atomic::AtomicBool>) {
        // let (tx, rx) = channel();
        // let buffer_ptr = self.buffer.as_mut_ptr(); //  as *mut ffi::c_void;
        // let buffer = self.buffer.clone(); // as_mut_ptr(); //  as *mut ffi::c_void;

        // let channel = self.inner.clone(); // pin_mut();
        // let buffer_size = self.buffer_size as u32;

        // no self in the thread
        // self.receiver_thread = Some(std::thread::spawn(move || {
        let packet_size = std::mem::size_of::<T>();
        let mut packet_count = 0;
        // let running =
        // while *self.recv_thread_started.lock().unwrap() {
        while !shutdown.load(atomic::Ordering::Relaxed) {
            // let mut buffer_lock = buffer.lock().unwrap();
            // let buffer_ptr = buffer_lock.as_mut_ptr();
            let buffer_ptr = self.buffer.as_mut_ptr();
            let buffer_size = self.buffer.len();
            // let mut channel_lock = channel.lock().unwrap(); // .pin_mut();
            // let channel_ptr = channel_lock.pin_mut();

            let num_recv_bytes = unsafe {
                self.channel.pin_mut().recv(
                    // self.host_channel.lock().unwrap().pin_mut().recv(
                    buffer_ptr as *mut _, // nvbit_sys::utils::c_void,
                    // *recv_buffer.as_mut_ptr() as *mut c_void,
                    buffer_size as u32,
                )
            };
            // let receiving = true;
            // if receiving && num_recv_bytes > 0 {
            if num_recv_bytes > 0 {
                // println!("received {} bytes", num_recv_bytes);
                let mut num_processed_bytes: usize = 0;
                while num_processed_bytes < num_recv_bytes as usize {
                    let packet_bytes =
                        &self.buffer[num_processed_bytes..num_processed_bytes + packet_size];
                    // println!("{:02X?}", packet_bytes);
                    assert_eq!(packet_size, packet_bytes.len());
                    let packet: T = unsafe { std::ptr::read(packet_bytes.as_ptr() as *const _) };
                    // println!("received from channel: {:#?}", &packet);

                    tx.send(packet);

                    // this should be done externally
                    // when we get this cta_id_x it means the kernel has completed
                    // if (ma.cta_id_x == -1) {
                    //     *self.recv_thread_receiving.lock().unwrap() = false;
                    //     break;
                    // }
                    // println!("size of common::inst_trace_t: {}", packet_size);
                    num_processed_bytes += packet_size;
                    packet_count += 1;
                }
            }
        }
        println!("received {} packets", packet_count);
        // }));
        // rx
    }
}

#[derive()]
pub struct HostChannel<T>
where
    T: Send + 'static,
{
    inner: Arc<Mutex<HostChannelInner<T>>>,
    // inner: Arc<Mutex<cxx::UniquePtr<nvbit_sys::utils::ChannelHost>>>,
    // buffer: Arc<Mutex<super::buffer::Buffer>>,
    receiver_thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<atomic::AtomicBool>,
    // buffer_size: usize,
    // tx: Sender<T>,
    // rx: Receiver<T>,
}

// unsafe impl<T> Send for HostChannel<T> {}

impl<T> HostChannel<T>
where
    // T: Send + std::fmt::Debug + 'static,
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

    // pub fn pin_mut(&mut self) -> Pin<&mut ChannelHost> {
    //     self.inner.pin_mut()
    // }

    // pub fn as_ptr(&self) -> *const ChannelHost {
    //     self.inner.lock().unwrap().as_ref().unwrap() as *const _
    // }

    // pub fn as_mut_ptr(&mut self) -> *mut ChannelHost {
    //     let mut lock = self.inner.lock().unwrap();
    //     let channel = unsafe { lock.as_mut().unwrap().get_unchecked_mut() };
    //     channel as *mut _
    // }

    /// Read from the channel
    pub fn read(&mut self) -> mpsc::Receiver<T> {
        let (tx, rx) = mpsc::channel();
        // let buffer_ptr = self.buffer.as_mut_ptr(); //  as *mut ffi::c_void;
        // let buffer = self.buffer.clone(); // as_mut_ptr(); //  as *mut ffi::c_void;

        // let channel = self.inner.clone(); // pin_mut();
        // let buffer_size = self.buffer_size as u32;

        // no self in the thread
        let inner = self.inner.clone();
        let shutdown = self.shutdown.clone();
        self.receiver_thread = Some(std::thread::spawn(move || {
            let mut lock = inner.lock().unwrap();
            lock.read(tx, shutdown);
        }));
        rx
        // self.receiver_thread = Some(std::thread::spawn(move || {
        //     let packet_size = std::mem::size_of::<T>();
        //     let mut packet_count = 0;
        //     // let running =
        //     // while *self.recv_thread_started.lock().unwrap() {
        //     loop {
        //         let mut buffer_lock = buffer.lock().unwrap();
        //         let buffer_ptr = buffer_lock.as_mut_ptr();
        //         let mut channel_lock = channel.lock().unwrap(); // .pin_mut();
        //         let channel_ptr = channel_lock.pin_mut();

        //         let num_recv_bytes = unsafe {
        //             channel_ptr.recv(
        //                 // self.host_channel.lock().unwrap().pin_mut().recv(
        //                 buffer_ptr as *mut _, // nvbit_sys::utils::c_void,
        //                 // *recv_buffer.as_mut_ptr() as *mut c_void,
        //                 buffer_size,
        //             )
        //         };
        //         let receiving = true;
        //         if receiving && num_recv_bytes > 0 {
        //             // println!("received {} bytes", num_recv_bytes);
        //             let mut num_processed_bytes: usize = 0;
        //             while num_processed_bytes < num_recv_bytes as usize {
        //                 let packet_bytes =
        //                     &buffer_lock[num_processed_bytes..num_processed_bytes + packet_size];
        //                 // println!("{:02X?}", packet_bytes);
        //                 assert_eq!(packet_size, packet_bytes.len());
        //                 let packet: T =
        //                     unsafe { std::ptr::read(packet_bytes.as_ptr() as *const _) };
        //                 println!("received from channel: {:#?}", &packet);

        //                 tx.send(packet);

        //                 // this should be done externally
        //                 // when we get this cta_id_x it means the kernel has completed
        //                 // if (ma.cta_id_x == -1) {
        //                 //     *self.recv_thread_receiving.lock().unwrap() = false;
        //                 //     break;
        //                 // }
        //                 // println!("size of common::inst_trace_t: {}", packet_size);
        //                 num_processed_bytes += packet_size;
        //                 packet_count += 1;
        //             }
        //         }
        //     }
        //     println!("received {} packets", packet_count);
        // }));
        // rx
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

    // pub fn recv(&mut self) -> Result<T, RecvError> {
    // pub fn recv(&mut self) -> Result<T, RecvError> {
    //     self.rx.recv()
    //     // self.pin_mut().recv()
    //     // let channel = unsafe { self.inner.as_mut().unwrap().get_unchecked_mut() };
    //     // channel as *mut _
    // }
}

impl<T> Drop for HostChannel<T>
where
    T: Send + 'static,
{
    fn drop(&mut self) {
        self.stop();
        // println!("Dropping Bar!")
    }
}

// impl<T> std::ops::Deref for HostChannel<T> {
//     type Target = Receiver<T>;

//     fn deref(&self) -> &Self::Target {
//         &self.rx
//     }
// }

// impl<T> std::ops::DerefMut for HostChannel<T> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.rx
//     }
// }
