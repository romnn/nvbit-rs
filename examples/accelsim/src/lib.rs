#![allow(clippy::missing_panics_doc, clippy::missing_safety_doc)]

mod instrument_inst;

use nvbit_rs::{model, DeviceChannel, HostChannel};
use once_cell::sync::Lazy;
use rustacuda::memory::UnifiedBox;
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

use common::inst_trace_t;

fn traces_dir() -> PathBuf {
    let example_dir = PathBuf::from(file!()).join("../../");
    let traces_dir =
        std::env::var("TRACES_DIR").map_or_else(|_| example_dir.join("traces"), PathBuf::from);
    // make sure trace dir exists
    std::fs::create_dir_all(&traces_dir).ok();
    traces_dir
}

// 1 MiB = 2**20
const CHANNEL_SIZE: u32 = 1 << 20;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<inst_trace_t>>,
    host_channel: Mutex<HostChannel<inst_trace_t>>,
    total_dyn_instr_counter: Mutex<UnifiedBox<u64>>,
    reported_dyn_instr_counter: Mutex<UnifiedBox<u64>>,
    stop_report: Mutex<UnifiedBox<bool>>,
    kernelid: Mutex<u64>,
    terminate_after_limit_number_of_kernels_reached: bool,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    opcode_to_id_map: RwLock<HashMap<String, usize>>,
    id_to_opcode_map: RwLock<HashMap<usize, String>>,
    instr_begin_interval: usize,
    instr_end_interval: usize,
    active_from_start: bool,
    active_region: Mutex<bool>,
    skip_flag: Mutex<bool>,
    dynamic_kernel_limit_start: u64,
    dynamic_kernel_limit_end: u64,
    start: Instant,
}

unsafe impl<'c> Send for Instrumentor<'c> {}
unsafe impl<'c> Sync for Instrumentor<'c> {}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = HostChannel::new(42, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let total_dyn_instr_counter = UnifiedBox::new(0u64).unwrap();
        let reported_dyn_instr_counter = UnifiedBox::new(0u64).unwrap();
        let stop_report = UnifiedBox::new(false).unwrap();

        let instrumentor = Arc::new(Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            host_channel: Mutex::new(host_channel),
            total_dyn_instr_counter: Mutex::new(total_dyn_instr_counter),
            reported_dyn_instr_counter: Mutex::new(reported_dyn_instr_counter),
            stop_report: Mutex::new(stop_report),
            kernelid: Mutex::new(1),
            terminate_after_limit_number_of_kernels_reached: false,
            recv_thread: Mutex::new(None),
            opcode_to_id_map: RwLock::new(HashMap::new()),
            id_to_opcode_map: RwLock::new(HashMap::new()),
            instr_begin_interval: 0,
            instr_end_interval: usize::MAX,
            active_from_start: true,
            active_region: Mutex::new(true),
            skip_flag: Mutex::new(false), // skip re-entry into intrumention logic
            dynamic_kernel_limit_start: 0, // start from the begging kernel
            dynamic_kernel_limit_end: 0,  // no limit
            start: Instant::now(),
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

        let trace_file_path = traces_dir().join("kernelslist");
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
            // println!("{:?}", packet);
            encoder.encode::<inst_trace_t>(packet).unwrap();
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KernelMetadata<'a> {
    name: &'a str,
    kernel_id: u64,
    grid: model::Dim,
    block: model::Dim,
    shared_mem_bytes: usize,
    nregs: i32,
    binary_version: i32,
    cuda_stream_id: u64,
    shared_mem_base_addr: u64,
    local_mem_base_addr: u64,
    nvbit_version: &'a str,
}

impl<'c> Instrumentor<'c> {
    fn instrument_instruction(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.print_decoded();

        let opcode = instr.opcode().expect("has opcode");

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

        instr.insert_call("instrument_inst", model::InsertionPoint::Before);

        let mut inst_args = instrument_inst::Args {
            opcode_id: opcode_id.try_into().unwrap(),
            vpc: instr.offset(),
            ..instrument_inst::Args::default()
        };

        // check all operands
        // For now, we ignore constant, TEX, predicates and unified registers.
        // We only report vector registers
        let mut src_oprd: [ffi::c_int; common::MAX_SRC] = [-1; common::MAX_SRC];
        let mut src_num: usize = 0;
        let mut dst_oprd: ffi::c_int = -1;
        let mut mem_oper_idx: ffi::c_int = -1;

        // find dst reg and handle the special case if the oprd[0] is mem
        // (e.g. store and RED)
        if let Some(first_operand) = instr.operand(0) {
            match first_operand.kind() {
                model::OperandKind::Register { num, .. } => {
                    dst_oprd = num;
                }
                model::OperandKind::MemRef { ra_num, .. } => {
                    src_oprd[0] = ra_num;
                    mem_oper_idx = 0;
                    src_num += 1;
                }
                _ => {
                    // skip anything else (constant and predicates)
                }
            }
        }

        // find src regs and mem
        let num_operands = instr.num_operands();
        let remaining_operands =
            (1..common::MAX_SRC.min(num_operands)).filter_map(|i| instr.operand(i));
        for operand in remaining_operands {
            match operand.kind() {
                model::OperandKind::MemRef { ra_num, .. } => {
                    // mem is found
                    assert!(src_num < common::MAX_SRC);
                    src_oprd[src_num] = ra_num;
                    src_num += 1;
                    // TODO: handle LDGSTS with two mem refs
                    assert!(mem_oper_idx == -1); // ensure one memory operand per inst
                    mem_oper_idx += 1;
                }
                model::OperandKind::Register { num, .. } => {
                    // reg is found
                    assert!(src_num < common::MAX_SRC);
                    src_oprd[src_num] = num;
                    src_num += 1;
                }
                _ => {
                    // skip anything else (constant and predicates)
                }
            };
        }

        // mem addresses info
        if mem_oper_idx >= 0 {
            inst_args.is_mem = true;
            inst_args.addr = 0;
            inst_args.width = instr.size().try_into().unwrap();
        } else {
            inst_args.is_mem = false;
            inst_args.addr = 1; // todo: was -1
            inst_args.width = 1; // todo: was -1
        }

        // reg info
        inst_args.desReg = dst_oprd;
        // we set the default value to -1,
        // however that will become 0 using unwrap_or_default()
        inst_args.srcReg1 = src_oprd[0];
        inst_args.srcReg2 = src_oprd[1];
        inst_args.srcReg3 = src_oprd[2];
        inst_args.srcReg4 = src_oprd[3];
        inst_args.srcReg5 = src_oprd[4];
        inst_args.srcNum = src_num.try_into().unwrap();

        inst_args.pchannel_dev = self.dev_channel.lock().unwrap().as_mut_ptr() as u64;

        inst_args.ptotal_dynamic_instr_counter = self
            .total_dyn_instr_counter
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.preported_dynamic_instr_counter = self
            .reported_dyn_instr_counter
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.pstop_report = self
            .stop_report
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.instrument(instr);
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
pub unsafe extern "C" fn nvbit_at_cuda_event(
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
    }
}

impl<'c> Instrumentor<'c> {
    #[allow(clippy::too_many_lines)]
    fn at_cuda_event(
        &self,
        is_exit: bool,
        cbid: nvbit_sys::nvbit_api_cuda_t,
        _event_name: &str,
        params: *mut ffi::c_void,
        _pstatus: *mut nvbit_sys::CUresult,
    ) {
        use nvbit_rs::EventParams;
        if *self.skip_flag.lock().unwrap() {
            return;
        }

        // if self.first_call {
        //     self.first_call = false;
        // }

        if self.active_from_start {
            *self.active_region.lock().unwrap() = self.dynamic_kernel_limit_start <= 1;
        }

        // kernelsFile = fopen(kernelslist_location.c_str(), "w");
        // statsFile = fopen(stats_location.c_str(), "w");
        // fprintf(statsFile,
        //         "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
        //         "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
        //         "total_insts, total_reported_insts\n");
        // fclose(statsFile);

        let params = EventParams::new(cbid, params);
        match params {
            Some(EventParams::MemCopyHostToDevice { .. }) => {
                if !is_exit {
                    // char buffer[1024];
                    // kernelsFile = fopen(kernelslist_location.c_str(), "a");
                    // sprintf(buffer, "MemcpyHtoD,0x%016lx,%lld", p->dstDevice, p->ByteCount);
                    // fprintf(kernelsFile, buffer);
                    // fprintf(kernelsFile, "\n");
                    // fclose(kernelsFile);
                }
            }
            Some(EventParams::KernelLaunch {
                mut func,
                grid,
                block,
                shared_mem_bytes,
                h_stream,
                ..
            }) => {
                if is_exit {
                    *self.skip_flag.lock().unwrap() = true;
                    unsafe {
                        common::flush_channel(self.dev_channel.lock().unwrap().as_mut_ptr().cast());
                    };

                    *self.skip_flag.lock().unwrap() = false;

                    // while *self.recv_thread_receiving.lock().unwrap() {
                    //     std::thread::yield_now();
                    // }

                    // if !stop_report {
                    //     fclose(resultsFile);
                    // }

                    if self.active_from_start
                        && self.dynamic_kernel_limit_end > 0
                        && *self.kernelid.lock().unwrap() > self.dynamic_kernel_limit_end
                    {
                        *self.active_region.lock().unwrap() = false;
                    }
                } else {
                    if self.active_from_start
                        && self.dynamic_kernel_limit_start > 0
                        && *self.kernelid.lock().unwrap() == self.dynamic_kernel_limit_start
                    {
                        *self.active_region.lock().unwrap() = true;
                    }

                    if self.terminate_after_limit_number_of_kernels_reached
                        && self.dynamic_kernel_limit_end > 0
                        && *self.kernelid.lock().unwrap() > self.dynamic_kernel_limit_end
                    {
                        std::process::exit(0);
                    }

                    self.instrument_function_if_needed(&mut func);

                    let enable_instrumentation = *self.active_region.lock().unwrap();
                    func.enable_instrumented(
                        &mut self.ctx.lock().unwrap(),
                        enable_instrumentation,
                        true, // apply to related functions
                    );
                    **self.stop_report.lock().unwrap() = !enable_instrumentation;

                    if !**self.stop_report.lock().unwrap() {
                        let kernel_id = *self.kernelid.lock().unwrap();
                        let kernel_metadata_path =
                            traces_dir().join(format!("kernel-{kernel_id}.trace"));
                        let kernel_metadata_file = std::io::BufWriter::new(
                            std::fs::OpenOptions::new()
                                .write(true)
                                .create(true)
                                .truncate(true)
                                .open(&kernel_metadata_path)
                                .unwrap(),
                        );

                        println!(
                            "Writing kernel metadata to {}",
                            kernel_metadata_path.display()
                        );

                        let metadata = {
                            let mut ctx = self.ctx.lock().unwrap();
                            let name = func.name(&mut ctx);
                            let nregs = func.num_registers().unwrap();
                            let shmem_static_nbytes = func.shared_memory_bytes().unwrap();
                            let binary_version = func.binary_version().unwrap();
                            let cuda_stream_id = h_stream.as_ptr() as u64;
                            let shared_mem_base_addr = nvbit_rs::shmem_base_addr(&mut ctx);
                            let local_mem_base_addr = nvbit_rs::local_mem_base_addr(&mut ctx);
                            KernelMetadata {
                                name,
                                kernel_id,
                                grid,
                                block,
                                shared_mem_bytes: shmem_static_nbytes + shared_mem_bytes as usize,
                                nregs,
                                binary_version,
                                cuda_stream_id,
                                shared_mem_base_addr,
                                local_mem_base_addr,
                                nvbit_version: nvbit_rs::version(),
                            }
                        };

                        serde_json::to_writer(kernel_metadata_file, &metadata).unwrap();

                        // fprintf(resultsFile,
                        //         "#traces format = threadblock_x threadblock_y threadblock_z "
                        //         "warpid_tb PC mask dest_num [reg_dests] opcode src_num "
                        //         "[reg_srcs] mem_width [adrrescompress?] [mem_addresses]\n");
                    }

                    *self.kernelid.lock().unwrap() += 1;
                }
            }
            Some(EventParams::ProfilerStart) => {
                if is_exit && !self.active_from_start {
                    *self.active_region.lock().unwrap() = true;
                }
            }
            Some(EventParams::ProfilerStop) => {
                if is_exit && !self.active_from_start {
                    *self.active_region.lock().unwrap() = false;
                }
            }
            _ => {
                // do nothing
            }
        }
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_init");
    unsafe {
        CONTEXTS
            .entry(ctx.handle())
            .or_insert_with(|| Instrumentor::new(ctx));
    }
}

/// NVBIT callback when CUDA context is terminated.
#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    let Some(trace_ctx) = (unsafe { CONTEXTS.get(&ctx.handle()) }) else {
        return;
    };

    // stop the host channel
    trace_ctx
        .host_channel
        .lock()
        .unwrap()
        .stop()
        .expect("stop host channel");

    dbg!(**trace_ctx.stop_report.lock().unwrap());
    dbg!(**trace_ctx.total_dyn_instr_counter.lock().unwrap());
    dbg!(**trace_ctx.reported_dyn_instr_counter.lock().unwrap());

    // finish receiving packets
    if let Some(recv_thread) = trace_ctx.recv_thread.lock().unwrap().take() {
        recv_thread.join().expect("join receiver thread");
    }

    println!("done after {:?}", trace_ctx.start.elapsed());
}
