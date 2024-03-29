#![allow(clippy::missing_panics_doc, clippy::missing_safety_doc)]

use nvbit_rs::{model, DeviceChannel, HostChannel};
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

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

fn app_prefix() -> String {
    let mut args: Vec<_> = std::env::args().collect();
    if let Some(executable) = args.get_mut(0) {
        *executable = PathBuf::from(&*executable)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
    }
    args.join("-")
}

fn traces_dir() -> PathBuf {
    let example_dir = PathBuf::from(file!()).parent().unwrap().join("../../../");
    let traces_dir =
        std::env::var("TRACES_DIR").map_or_else(|_| example_dir.join("traces"), PathBuf::from);
    // make sure trace dir exists
    std::fs::create_dir_all(&traces_dir).ok();
    traces_dir
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MemAccessTraceEntry<'a> {
    pub cuda_ctx: u64,
    pub grid_launch_id: u64,
    pub cta_id: model::Dim,
    pub warp_id: u32,
    pub instr_opcode: &'a str,
    pub instr_offset: u32,
    pub instr_idx: u32,
    pub instr_predicate: model::Predicate,
    pub instr_mem_space: model::MemorySpace,
    pub instr_is_load: bool,
    pub instr_is_store: bool,
    pub instr_is_extended: bool,
    /// Address per thread in warp
    pub addrs: [u64; 32],
}

#[allow(non_snake_case, clippy::struct_excessive_bools)]
#[derive(Debug, Default, Clone)]
struct Args {
    instr_opcode_id: std::ffi::c_int,
    instr_offset: u32,
    instr_idx: u32,
    instr_predicate_num: std::ffi::c_int,
    instr_predicate_is_neg: bool,
    instr_predicate_is_uniform: bool,
    instr_mem_space: u8,
    instr_is_load: bool,
    instr_is_store: bool,
    instr_is_extended: bool,
    mref_idx: u64,
    pchannel_dev: u64,
}

impl Args {
    pub fn instrument(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.instr_opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_offset);
        instr.add_call_arg_const_val32(self.instr_idx);
        instr.add_call_arg_const_val32(self.instr_predicate_num.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_predicate_is_neg.into());
        instr.add_call_arg_const_val32(self.instr_predicate_is_uniform.into());
        instr.add_call_arg_const_val32(self.instr_mem_space.into());
        instr.add_call_arg_const_val32(self.instr_is_load.into());
        instr.add_call_arg_const_val32(self.instr_is_store.into());
        instr.add_call_arg_const_val32(self.instr_is_extended.into());

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
const CHANNEL_SIZE: u32 = 1 << 20;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<mem_access_t>>,
    host_channel: Mutex<HostChannel<mem_access_t>>,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    opcode_to_id_map: RwLock<HashMap<String, usize>>,
    id_to_opcode_map: RwLock<HashMap<usize, String>>,
    start: Instant,
    grid_launch_id: Mutex<u64>,
    instr_begin_interval: usize,
    instr_end_interval: usize,
    skip_flag: Mutex<bool>,
}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = HostChannel::new(42, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let instrumentor = Arc::new(Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            host_channel: Mutex::new(host_channel),
            recv_thread: Mutex::new(None),
            opcode_to_id_map: RwLock::new(HashMap::new()),
            id_to_opcode_map: RwLock::new(HashMap::new()),
            grid_launch_id: Mutex::new(0),
            start: Instant::now(),
            instr_begin_interval: 0,
            instr_end_interval: usize::MAX,
            skip_flag: Mutex::new(false), // skip re-entry into intrumention logic
        });

        // start receiving from the channel
        let instrumentor_clone = instrumentor.clone();
        *instrumentor.recv_thread.lock().unwrap() = Some(std::thread::spawn(move || {
            instrumentor_clone.read_channel();
        }));

        instrumentor
    }

    fn read_channel(self: Arc<Self>) {
        let rx = self.host_channel.lock().unwrap().read();

        let trace_file_path = traces_dir().join(format!("{}-trace", &app_prefix()));
        dbg!(&trace_file_path);
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
        let mut encoder = nvbit_io::Encoder::new(&mut serializer).unwrap();

        // start the thread here
        let mut packet_count = 0;
        while let Ok(packet) = rx.recv() {
            // when cta_id_x == -1, the kernel has completed
            if packet.cta_id_x == -1 {
                self.host_channel
                    .lock()
                    .unwrap()
                    .stop()
                    .expect("stop host channel");
                break;
            }
            packet_count += 1;

            // we keep the read lock for as long as encoding takes
            // so we avoid copying the opcode string
            let cuda_ctx = self.ctx.lock().unwrap().as_ptr() as u64;
            let lock = self.id_to_opcode_map.read().unwrap();
            let opcode = &lock[&(packet.instr_opcode_id as usize)];
            let cta_id = model::Dim {
                x: packet.cta_id_x.unsigned_abs(),
                y: packet.cta_id_y.unsigned_abs(),
                z: packet.cta_id_z.unsigned_abs(),
            };
            let instr_predicate = model::Predicate {
                num: packet.instr_predicate_num,
                is_neg: packet.instr_predicate_is_neg,
                is_uniform: packet.instr_predicate_is_uniform,
            };
            let instr_mem_space: model::MemorySpace = unsafe {
                let variant = u8::try_from(packet.instr_mem_space).unwrap();
                std::mem::transmute(variant)
            };

            let entry = MemAccessTraceEntry {
                cuda_ctx,
                grid_launch_id: packet.grid_launch_id,
                cta_id,
                warp_id: packet.warp_id.unsigned_abs(),
                instr_opcode: opcode,
                instr_offset: packet.instr_offset,
                instr_idx: packet.instr_idx,
                instr_predicate,
                instr_mem_space,
                instr_is_load: packet.instr_is_load,
                instr_is_store: packet.instr_is_store,
                instr_is_extended: packet.instr_is_extended,
                addrs: packet.addrs,
            };
            encoder.encode::<MemAccessTraceEntry>(entry).unwrap();
        }

        encoder.finalize().unwrap();
        println!(
            "wrote {} packets to {}",
            &packet_count,
            &trace_file_path.display()
        );
    }
}

type Contexts = HashMap<nvbit_rs::ContextHandle<'static>, Arc<Instrumentor<'static>>>;

static mut CONTEXTS: Lazy<Contexts> = Lazy::new(HashMap::new);

impl<'c> Instrumentor<'c> {
    fn at_cuda_event(
        &self,
        is_exit: bool,
        cbid: nvbit_sys::nvbit_api_cuda_t,
        _event_name: &str,
        params: *mut ffi::c_void,
        _pstatus: *mut nvbit_sys::CUresult,
    ) {
        if *self.skip_flag.lock().unwrap() {
            return;
        }

        let params = nvbit_rs::EventParams::new(cbid, params);

        match params {
            Some(nvbit_rs::EventParams::MemAlloc {
                device_ptr,
                num_bytes,
            }) => {
                if is_exit {
                    // addresses are only valid on exit
                    println!("allocated {num_bytes} bytes at {device_ptr:#06x} ({device_ptr})");
                }
            }
            Some(nvbit_rs::EventParams::KernelLaunch {
                mut func,
                grid,
                block,
                shared_mem_bytes,
                h_stream,
                ..
            }) => {
                // make sure GPU is idle
                // unsafe { nvbit_sys::cuCtxSynchronize() };

                dbg!(func.as_mut_ptr());
                if is_exit {
                    return;
                }
                self.instrument_function_if_needed(&mut func);

                let ctx = &mut self.ctx.lock().unwrap();
                let mut grid_launch_id = self.grid_launch_id.lock().unwrap();

                let nregs = func.num_registers().unwrap();
                let shmem_static_nbytes = func.shared_memory_bytes().unwrap();
                let func_name = func.name(ctx);
                let pc = func.addr();

                println!("MEMTRACE: CTX {:#06x} - LAUNCH", ctx.as_ptr() as u64);
                println!("\tKernel pc: {pc:#06x}");
                println!("\tKernel name: {func_name}");
                println!("\tGrid launch id: {grid_launch_id}");
                println!("\tGrid size: {grid}");
                println!("\tBlock size: {block}");
                println!("\tNum registers: {nregs}");
                println!(
                    "\tShared memory bytes: {}",
                    shmem_static_nbytes + shared_mem_bytes as usize
                );
                println!("\tCUDA stream id: {}", h_stream.as_ptr() as u64);

                *grid_launch_id += 1;

                // enable instrumented code to run
                func.enable_instrumented(ctx, true, true);
            }
            _ => {}
        }
    }

    fn instrument_instruction(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        // instr.print_decoded();

        let opcode = instr.opcode().expect("has opcode");

        // if let model::MemorySpace::Constant = instr.memory_space() {
        //     // skip constant memory instructions
        //     return;
        // }

        let opcode_id = {
            let mut opcode_to_id_map = self.opcode_to_id_map.write().unwrap();
            let mut id_to_opcode_map = self.id_to_opcode_map.write().unwrap();

            if !opcode_to_id_map.contains_key(opcode) {
                let opcode_id = opcode_to_id_map.len();
                opcode_to_id_map.insert(opcode.to_string(), opcode_id);
                id_to_opcode_map.insert(opcode_id, opcode.to_string());
            }

            opcode_to_id_map[opcode]
        };

        let mut instrumented = false;

        if opcode.to_lowercase() == "exit" {
            instr.insert_call("instrument_inst", model::InsertionPoint::Before);
            let mut pchannel_dev_lock = self.dev_channel.lock().unwrap();
            let predicate = instr.predicate().unwrap_or_default();
            let inst_args = Args {
                instr_opcode_id: opcode_id.try_into().unwrap(),
                instr_offset: instr.offset(),
                instr_idx: instr.idx(),
                instr_predicate_num: predicate.num,
                instr_predicate_is_neg: predicate.is_neg,
                instr_predicate_is_uniform: predicate.is_uniform,
                instr_mem_space: instr.memory_space() as u8,
                instr_is_load: instr.is_load(),
                instr_is_store: instr.is_store(),
                instr_is_extended: instr.is_extended(),
                mref_idx: 0,
                pchannel_dev: pchannel_dev_lock.as_mut_ptr() as u64,
            };
            inst_args.instrument(instr);
            instrumented = true;
        }

        if instr.memory_space() != model::MemorySpace::None {
            let mut mref_idx = 0;

            // iterate on the operands
            for operand in instr.operands().collect::<Vec<_>>() {
                println!("operand {} kind: {:?}", &operand.name(), &operand.kind());

                if let model::OperandKind::MemRef { .. } = operand.kind() {
                    instr.insert_call("instrument_inst", model::InsertionPoint::Before);
                    let mut pchannel_dev_lock = self.dev_channel.lock().unwrap();
                    let predicate = instr.predicate().unwrap_or_default();
                    let inst_args = Args {
                        instr_opcode_id: opcode_id.try_into().unwrap(),
                        instr_offset: instr.offset(),
                        instr_idx: instr.idx(),
                        instr_predicate_num: predicate.num,
                        instr_predicate_is_neg: predicate.is_neg,
                        instr_predicate_is_uniform: predicate.is_uniform,
                        instr_mem_space: instr.memory_space() as u8,
                        instr_is_load: instr.is_load(),
                        instr_is_store: instr.is_store(),
                        instr_is_extended: instr.is_extended(),
                        mref_idx,
                        pchannel_dev: pchannel_dev_lock.as_mut_ptr() as u64,
                    };
                    inst_args.instrument(instr);
                    mref_idx += 1;
                    instrumented = true;
                }
            }
        }

        if instrumented {
            let instr_idx = instr.idx();
            let instr_offset = instr.offset();
            let line_info = instr.line_info(&mut self.ctx.lock().unwrap()).map(|info| {
                format!(
                    "{}:{}",
                    PathBuf::from(info.dir_name).join(info.file_name).display(),
                    info.line
                )
            });
            println!(
                "[{}] instrumented instruction {} at index {} (offset {})\n\n",
                line_info.unwrap_or_default(),
                instr,
                instr_idx,
                instr_offset,
            );
        }
    }

    fn instrument_function_if_needed<'f: 'c>(&self, func: &mut nvbit_rs::Function<'f>) {
        let mut related_functions = func.related_functions(&mut self.ctx.lock().unwrap());
        for f in related_functions.iter_mut().chain([func]) {
            let func_name = f.name(&mut self.ctx.lock().unwrap());
            let func_addr = f.addr();

            if !self.already_instrumented.lock().unwrap().insert(f.handle()) {
                println!("already instrumented function {func_name} at address {func_addr:#X}");
                continue;
            }

            println!("inspecting function {func_name} at address {func_addr:#X}");

            let mut instrs = f.instructions(&mut self.ctx.lock().unwrap());

            // iterate on all the static instructions in the function
            for (cnt, instr) in instrs.iter_mut().enumerate() {
                if cnt < self.instr_begin_interval || cnt >= self.instr_end_interval {
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
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: nvbit_sys::nvbit_api_cuda_t,
    event_name: nvbit_rs::CudaEventName,
    params: *mut ffi::c_void,
    pstatus: *mut nvbit_sys::CUresult,
) {
    let is_exit = is_exit != 0;
    println!("nvbit_at_cuda_event: {event_name} (is_exit = {is_exit})");

    if let Some(trace_ctx) = unsafe { CONTEXTS.get(&ctx.handle()) } {
        trace_ctx.at_cuda_event(is_exit, cbid, &event_name, params, pstatus);
    };
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(mut ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_init ({:?})", ctx.as_mut_ptr());
    unsafe {
        CONTEXTS
            .entry(ctx.handle())
            .or_insert_with(|| Instrumentor::new(ctx));
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    let Some(trace_ctx) = (unsafe { CONTEXTS.get(&ctx.handle()) }) else {
        return;
    };

    *trace_ctx.skip_flag.lock().unwrap() = true;
    unsafe {
        // flush channel
        let mut dev_channel = trace_ctx.dev_channel.lock().unwrap();
        common::flush_channel(dev_channel.as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };

    // stop the host channel
    trace_ctx
        .host_channel
        .lock()
        .unwrap()
        .stop()
        .expect("stop host channel");

    // finish receiving packets
    if let Some(recv_thread) = trace_ctx.recv_thread.lock().unwrap().take() {
        recv_thread.join().expect("join receiver thread");
    }

    println!("done after {:?}", trace_ctx.start.elapsed());
}
