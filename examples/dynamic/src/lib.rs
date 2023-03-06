#![allow(warnings)]
#![allow(clippy::missing_panics_doc)]

use lazy_static::lazy_static;
use nvbit_rs::{DeviceChannel, HostChannel};
use rustacuda::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// const PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/inject_funcs.1.sm_30.ptx"));

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
mod common {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use common::mem_access_t;

#[allow(non_snake_case)]
#[derive(Debug, Default, Clone)]
pub struct Args {
    pub opcode_id: std::ffi::c_int,
    pub mref_idx: u64,
    pub pchannel_dev: u64,
}

impl Args {
    pub fn instrument<'a>(&self, instr: &mut nvbit_rs::Instruction<'a>) {
        // predicate value
        instr.add_call_arg_guard_pred_val();
        // opcode id
        instr.add_call_arg_const_val32(self.opcode_id.try_into().unwrap_or_default());
        // memory reference 64 bit address
        instr.add_call_arg_mref_addr64(self.mref_idx.try_into().unwrap_or_default());
        // add "space" for kernel function pointer,
        // that will be set at launch time
        // (64 bit value at offset 0 of the dynamic arguments)
        instr.add_call_arg_launch_val64(0);
        instr.add_call_arg_const_val64(self.pchannel_dev);
    }
}

// 1 MiB = 2**20
const CHANNEL_SIZE: usize = 1 << 20;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<mem_access_t>>,
    host_channel: Arc<Mutex<HostChannel<mem_access_t>>>,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    start: Instant,
    grid_launch_id: Mutex<u64>,
    instr_begin_interval: usize,
    instr_end_interval: usize,
    skip_flag: Mutex<bool>,
}

impl<'c> Instrumentor<'c> {
    fn new(ctx: nvbit_rs::Context<'c>) -> Self {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = Arc::new(Mutex::new(
            HostChannel::<mem_access_t>::new(42, CHANNEL_SIZE, &mut dev_channel).unwrap(),
        ));

        let rx = host_channel.lock().unwrap().read();
        let example_dir = PathBuf::from(file!())
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let traces_dir = std::env::var("TRACES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| example_dir.join("traces"));

        // make sure trace dir exists
        std::fs::create_dir_all(&traces_dir).unwrap();

        let host_channel_clone = host_channel.clone();
        let recv_thread = Mutex::new(Some(std::thread::spawn(move || {
            let trace_file_path = traces_dir.join("kernelslist");
            let mut file = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&trace_file_path)
                    .unwrap(),
            );
            let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
            let mut serializer = serde_json::Serializer::with_formatter(&mut file, formatter);
            let mut encoder = nvbit_rs::Encoder::new(&mut serializer);

            let mut packet_count = 0;
            while let Ok(packet) = rx.recv() {
                // when cta_id_x == -1, the kernel has completed
                if packet.cta_id_x == -1 {
                    host_channel_clone
                        .lock()
                        .unwrap()
                        .stop()
                        .expect("stop host channel");
                    break;
                }
                packet_count += 1;

                // todo: use a new struct for serialization here
                // ss << "CTX " << HEX(ctx) << " - grid_launch_id "
                //    << ma->grid_launch_id << " - CTA " << ma->cta_id_x << ","
                //    << ma->cta_id_y << "," << ma->cta_id_z << " - warp "
                //    << ma->warp_id << " - " << id_to_opcode_map[ma->opcode_id]
                //    << " - ";

                // for (int i = 0; i < 32; i++) {
                //     ss << HEX(ma->addrs[i]) << " ";
                // }

                encoder.encode::<mem_access_t>(packet).unwrap();
            }

            encoder.finalize().unwrap();
            println!(
                "wrote {} packets to {}",
                &packet_count,
                &trace_file_path.display()
            );
        })));

        Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            host_channel,
            recv_thread,
            grid_launch_id: Mutex::new(0),
            start: Instant::now(),
            instr_begin_interval: 0,
            instr_end_interval: usize::MAX,
            skip_flag: Mutex::new(false),
        }
    }
}

type Contexts = RwLock<HashMap<nvbit_rs::ContextHandle<'static>, Instrumentor<'static>>>;

lazy_static! {
    static ref CONTEXTS: Contexts = RwLock::new(HashMap::new());
}

impl<'c> Instrumentor<'c> {
    fn at_cuda_event(
        &self,
        is_exit: bool,
        cbid: nvbit_sys::nvbit_api_cuda_t,
        _event_name: &str,
        params: *mut ffi::c_void,
        _pstatus: *mut nvbit_sys::CUresult,
    ) {
        use nvbit_rs::CudaEventParams;
        if *self.skip_flag.lock().unwrap() {
            return;
        }

        let params = CudaEventParams::new(cbid, params);

        if let Some(CudaEventParams::KernelLaunch { mut func, .. }) = params {
            // make sure GPU is idle
            unsafe { nvbit_sys::cuCtxSynchronize() };

            if !is_exit {
                self.instrument_function_if_needed(&mut func);

                let ctx = &mut self.ctx.lock().unwrap();
                let nregs = func.num_registers().unwrap();
                let shmem_static_nbytes = func.shared_memory_bytes().unwrap();
                let func_name = func.name(ctx);
                let pc = func.addr();

                *self.grid_launch_id.lock().unwrap() += 1;

                // enable instrumented code to run
                nvbit_rs::enable_instrumented(ctx, &mut func, true, true);
            }
        }
    }

    fn instrument_instruction<'f>(&self, instr: &mut nvbit_rs::Instruction<'f>) {
        instr.print_decoded();

        let mut opcode_to_id_map: HashMap<String, usize> = HashMap::new();
        let mut id_to_opcode_map: HashMap<usize, String> = HashMap::new();

        let opcode = instr.opcode().expect("has opcode");

        if !opcode_to_id_map.contains_key(opcode) {
            let opcode_id = opcode_to_id_map.len();
            opcode_to_id_map.insert(opcode.to_string(), opcode_id);
            id_to_opcode_map.insert(opcode_id, opcode.to_string());
        }

        let opcode_id = opcode_to_id_map[opcode];
        let mut mref_idx = 0;

        // iterate on the operands
        for operand in instr.operands().collect::<Vec<_>>() {
            if operand.kind() == nvbit_rs::OperandKind::MemRef {
                instr.insert_call("instrument_inst", nvbit_rs::InsertionPoint::Before);
                let mut inst_args = Args {
                    opcode_id: opcode_id.try_into().unwrap(),
                    mref_idx,
                    pchannel_dev: self.dev_channel.lock().unwrap().as_mut_ptr() as u64,
                };
                inst_args.instrument(instr);
                mref_idx += 1;
            }
        }
    }

    fn instrument_function_if_needed<'f>(&self, func: &mut nvbit_rs::Function<'f>) {
        let mut related_functions =
            nvbit_rs::get_related_functions(&mut self.ctx.lock().unwrap(), func);

        for f in related_functions.iter_mut().chain([func]) {
            let mut f = nvbit_rs::Function::wrap(f.as_mut_ptr());

            let func_name = nvbit_rs::get_func_name(&mut self.ctx.lock().unwrap(), &mut f);
            let func_addr = nvbit_rs::get_func_addr(&mut f);

            if !self.already_instrumented.lock().unwrap().insert(f.handle()) {
                println!(
                    "already instrumented function {} at address {:#X}",
                    func_name, func_addr
                );
                continue;
            }

            println!(
                "inspecting function {} at address {:#X}",
                func_name, func_addr
            );

            let mut instrs = nvbit_rs::get_instrs(&mut self.ctx.lock().unwrap(), &mut f);

            // iterate on all the static instructions in the function
            for (cnt, instr) in &mut instrs.iter_mut().enumerate() {
                if cnt < self.instr_begin_interval || cnt >= self.instr_end_interval {
                    continue;
                }
                if let nvbit_rs::MemorySpace::None | nvbit_rs::MemorySpace::Constant =
                    instr.memory_space()
                {
                    continue;
                }

                self.instrument_instruction(instr);
            }
        }
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    println!("nvbit_at_init");

    // Set up CUDA context and load the instrumentation module
    // we already have the context somehow
    // rustacuda::init(CudaFlags::empty()).unwrap();
    // let device = Device::get_device(0).unwrap();
    // let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
    // .unwrap();

    // let ptx = CString::new(include_str!("../resources/add.ptx"))?;

    println!("nvbit_at_init success");
}

#[no_mangle]
#[inline(never)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: nvbit_sys::nvbit_api_cuda_t,
    event_name: *const ffi::c_char,
    params: *mut ffi::c_void,
    pstatus: *mut nvbit_sys::CUresult,
) {
    let is_exit = is_exit != 0;
    let event_name = unsafe { ffi::CStr::from_ptr(event_name).to_str().unwrap() };
    println!(
        "nvbit_at_cuda_event: {:?} (is_exit = {})",
        event_name, is_exit
    );

    let lock = CONTEXTS.read().unwrap();
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };
    instrumentor.at_cuda_event(is_exit, cbid, event_name, params, pstatus);
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_init");

    // let module = Module::load_from_string(&ffi::CString::new(PTX).unwrap()).unwrap();
    CONTEXTS
        .write()
        .unwrap()
        .entry(ctx.handle())
        .or_insert_with(|| Instrumentor::new(ctx));

    println!("nvbit_at_ctx_init success");
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    let lock = CONTEXTS.read().unwrap();
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };

    *instrumentor.skip_flag.lock().unwrap() = true;
    unsafe {
        // flush channel

        common::flush_channel(instrumentor.dev_channel.lock().unwrap().as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };
    *instrumentor.skip_flag.lock().unwrap() = false;

    // stop the host channel
    instrumentor
        .host_channel
        .lock()
        .unwrap()
        .stop()
        .expect("stop host channel");

    // finish receiving packets
    if let Some(recv_thread) = instrumentor.recv_thread.lock().unwrap().take() {
        recv_thread.join().expect("join receiver thread");
    }

    println!(
        "done after {:?}",
        Instant::now().duration_since(instrumentor.start)
    );
}
