#![allow(clippy::missing_panics_doc)]

mod instrument_inst;

use lazy_static::lazy_static;
use nvbit_rs::{DeviceChannel, HostChannel};
use nvbit_sys::bindings;
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

// 1 MiB = 2**20
const CHANNEL_SIZE: usize = 1 << 20;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<inst_trace_t>>,
    host_channel: Arc<Mutex<HostChannel<inst_trace_t>>>,
    ptotal_dynamic_instr_counter: Mutex<UnifiedBox<u64>>,
    preported_dynamic_instr_counter: Mutex<UnifiedBox<u64>>,
    pstop_report: Mutex<UnifiedBox<bool>>,
    kernelid: Mutex<u64>,
    terminate_after_limit_number_of_kernels_reached: bool,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    instr_begin_interval: u32,
    instr_end_interval: u32,
    active_from_start: bool,
    active_region: Mutex<bool>,
    skip_flag: Mutex<bool>,
    dynamic_kernel_limit_start: u64,
    dynamic_kernel_limit_end: u64,
    start: Instant,
}

impl<'c> Instrumentor<'c> {
    fn new(ctx: nvbit_rs::Context<'c>) -> Self {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = Arc::new(Mutex::new(
            HostChannel::<inst_trace_t>::new(42, CHANNEL_SIZE, &mut dev_channel).unwrap(),
        ));

        let rx = host_channel.lock().unwrap().read();
        let traces_dir = PathBuf::from(
            std::env::var("TRACES_DIR").expect("missing TRACES_DIR environment variable"),
        );

        // make sure trace dir exists
        std::fs::create_dir_all(&traces_dir).unwrap();

        let host_channel_clone = host_channel.clone();
        let recv_thread = Mutex::new(Some(std::thread::spawn(move || {
            let mut file = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(traces_dir.join("kernelslist"))
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
                // println!("{:?}", packet);
                encoder.encode::<inst_trace_t>(packet).unwrap();
            }

            encoder.finalize().unwrap();
            println!("received {} packets", packet_count);
        })));

        let ptotal_dynamic_instr_counter = Mutex::new(UnifiedBox::new(0u64).unwrap());
        let preported_dynamic_instr_counter = Mutex::new(UnifiedBox::new(0u64).unwrap());
        let pstop_report = Mutex::new(UnifiedBox::new(false).unwrap());

        Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            host_channel,
            ptotal_dynamic_instr_counter,
            preported_dynamic_instr_counter,
            pstop_report,
            kernelid: Mutex::new(1),
            terminate_after_limit_number_of_kernels_reached: false,
            recv_thread,
            instr_begin_interval: 0,
            instr_end_interval: u32::MAX,
            active_from_start: true,
            active_region: Mutex::new(true), // region of interest
            skip_flag: Mutex::new(false),
            dynamic_kernel_limit_start: 0, // start from the begging kernel
            dynamic_kernel_limit_end: 0,   // // no limit
            start: Instant::now(),
        }
    }
}

unsafe impl<'c> Send for Instrumentor<'c> {}
unsafe impl<'c> Sync for Instrumentor<'c> {}

type Contexts = RwLock<HashMap<nvbit_rs::ContextHandle<'static>, Instrumentor<'static>>>;

lazy_static! {
    static ref CONTEXTS: Contexts = RwLock::new(HashMap::new());
}

impl<'c> Instrumentor<'c> {
    fn instrument_instruction<'f>(&self, instr: &mut nvbit_rs::Instruction<'f>) {
        // instr.print_decoded();

        let mut opcode_to_id_map: HashMap<String, usize> = HashMap::new();
        let mut id_to_opcode_map: HashMap<usize, String> = HashMap::new();

        let opcode = instr.opcode().expect("has opcode");

        if !opcode_to_id_map.contains_key(opcode) {
            let opcode_id = opcode_to_id_map.len();
            opcode_to_id_map.insert(opcode.to_string(), opcode_id);
            id_to_opcode_map.insert(opcode_id, opcode.to_string());
        }

        let opcode_id = opcode_to_id_map[opcode];

        instr.insert_call("instrument_inst", nvbit_rs::InsertionPoint::Before);

        let mut inst_args = instrument_inst::Args {
            opcode_id: opcode_id.try_into().unwrap(),
            vpc: instr.offset(),
            ..instrument_inst::Args::default()
        };

        // check all operands
        // For now, we ignore constant, TEX, predicates and unified registers.
        // We only report vector registers
        let mut src_oprd: [ffi::c_int; common::MAX_SRC as usize] = [-1; common::MAX_SRC as usize];
        let mut src_num: usize = 0;
        let mut dst_oprd: ffi::c_int = -1;
        let mut mem_oper_idx: ffi::c_int = -1;

        // find dst reg and handle the special case if the oprd[0] is mem
        // (e.g. store and RED)
        let num_operands = instr.num_operands();
        if num_operands > 0 {
            let first_operand = instr.operand(0).unwrap().into_owned();
            match first_operand.type_ {
                bindings::InstrType_OperandType::REG => {
                    dst_oprd = unsafe { first_operand.u.reg.num };
                }
                bindings::InstrType_OperandType::MREF => {
                    src_oprd[0] = unsafe { first_operand.u.mref.ra_num };
                    mem_oper_idx = 0;
                    src_num += 1;
                }
                _ => {
                    // skip anything else (constant and predicates)
                }
            }
        }

        // find src regs and mem
        for i in 1..(common::MAX_SRC.min(num_operands.try_into().unwrap())) {
            assert!(i < common::MAX_SRC);
            assert!((i as usize) < num_operands);
            let op = instr.operand(i as usize).unwrap().into_owned();
            match op.type_ {
                bindings::InstrType_OperandType::MREF => {
                    // mem is found
                    assert!(src_num < common::MAX_SRC as usize);
                    src_oprd[src_num] = unsafe { op.u.mref.ra_num };
                    src_num += 1;
                    // TODO: handle LDGSTS with two mem refs
                    assert!(mem_oper_idx == -1); // ensure one memory operand per inst
                    mem_oper_idx += 1;
                }
                bindings::InstrType_OperandType::REG => {
                    // reg is found
                    assert!(src_num < common::MAX_SRC as usize);
                    src_oprd[src_num] = unsafe { op.u.reg.num };
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
            .ptotal_dynamic_instr_counter
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.preported_dynamic_instr_counter = self
            .preported_dynamic_instr_counter
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.pstop_report = self
            .pstop_report
            .lock()
            .unwrap()
            .as_unified_ptr()
            .as_raw_mut() as u64;

        inst_args.instrument(instr);
    }

    fn instrument_function_if_needed<'f>(&self, func: &mut nvbit_rs::Function<'f>) {
        let mut related_functions =
            nvbit_rs::get_related_functions(&mut self.ctx.lock().unwrap(), func);

        for f in related_functions.iter_mut().chain([func]) {
            let mut f = nvbit_rs::Function::new(f.as_mut_ptr());

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

            let mut instr_count: u32 = 0;

            // iterate on all the static instructions in the function
            for instr in &mut instrs {
                instr_count += 1;
                if instr_count < self.instr_begin_interval || instr_count >= self.instr_end_interval
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
}

#[no_mangle]
#[inline(never)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: bindings::nvbit_api_cuda_t,
    event_name: *const ffi::c_char,
    params: *mut ffi::c_void,
    pstatus: *mut bindings::CUresult,
) {
    let is_exit = is_exit != 0;
    let event_name = unsafe { ffi::CStr::from_ptr(event_name).to_str().unwrap() };
    println!(
        "nvbit_at_cuda_event: {:?} (is_exit = {})",
        event_name, is_exit
    );
    if let Some(instrumentor) = CONTEXTS.read().unwrap().get(&ctx.handle()) {
        instrumentor.at_cuda_event(is_exit, cbid, event_name, params, pstatus);
    }
}

impl<'c> Instrumentor<'c> {
    fn at_cuda_event(
        &self,
        is_exit: bool,
        cbid: bindings::nvbit_api_cuda_t,
        _event_name: &str,
        params: *mut ffi::c_void,
        _pstatus: *mut bindings::CUresult,
    ) {
        if *self.skip_flag.lock().unwrap() {
            return;
        }

        // if self.first_call {
        //     self.first_call = false;
        // }

        // if (mkdir("traces", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        //   if (errno == EEXIST) {
        //     // alredy exists
        //   } else {
        //     // something else
        //     std::cout << "cannot create folder error:" << strerror(errno)
        //               << std::endl;
        //     return;
        //   }
        // }

        if self.active_from_start {
            if self.dynamic_kernel_limit_start <= 1 {
                *self.active_region.lock().unwrap() = true;
            } else {
                *self.active_region.lock().unwrap() = false;
            }
        }

        // if(user_defined_folders == 1)
        // {
        //   std::string usr_folder = std::getenv("TRACES_FOLDER");
        //   std::string temp_traces_location = usr_folder;
        //   std::string temp_kernelslist_location = usr_folder + "/kernelslist";
        //   std::string temp_stats_location = usr_folder + "/stats.csv";
        //   traces_location.resize(temp_traces_location.size());
        //   kernelslist_location.resize(temp_kernelslist_location.size());
        //   stats_location.resize(temp_stats_location.size());
        //   traces_location.replace(traces_location.begin(), traces_location.end(),temp_traces_location);
        //   kernelslist_location.replace(kernelslist_location.begin(), kernelslist_location.end(),temp_kernelslist_location);
        //   stats_location.replace(stats_location.begin(), stats_location.end(),temp_stats_location);
        //   printf("\n Traces location is %s \n", traces_location.c_str());
        //   printf("Kernelslist location is %s \n", kernelslist_location.c_str());
        //   printf("Stats location is %s \n", stats_location.c_str());
        // }

        // kernelsFile = fopen(kernelslist_location.c_str(), "w");
        // statsFile = fopen(stats_location.c_str(), "w");
        // fprintf(statsFile,
        //         "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
        //         "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
        //         "total_insts, total_reported_insts\n");
        // fclose(statsFile);
        // }

        match cbid {
            bindings::nvbit_api_cuda_t::API_CUDA_cuMemcpyHtoD_v2 => {
                if !is_exit {
                    let p = unsafe { &mut *params.cast::<bindings::cuMemcpyHtoD_v2_params>() };
                    // let p: &mut bindings::cuMemcpyHtoD_v2_params =
                    //     unsafe { &mut *(params as *mut bindings::cuMemcpyHtoD_v2_params) };

                    dbg!(&p);
                    // char buffer[1024];
                    // kernelsFile = fopen(kernelslist_location.c_str(), "a");
                    // sprintf(buffer, "MemcpyHtoD,0x%016lx,%lld", p->dstDevice, p->ByteCount);
                    // fprintf(kernelsFile, buffer);
                    // fprintf(kernelsFile, "\n");
                    // fclose(kernelsFile);
                }
            }
            bindings::nvbit_api_cuda_t::API_CUDA_cuLaunchKernel_ptsz
            | bindings::nvbit_api_cuda_t::API_CUDA_cuLaunchKernel => {
                let p = unsafe { &mut *params.cast::<bindings::cuLaunchKernel_params>() };
                dbg!(&p);
                let mut pf = nvbit_rs::Function::new(p.f);

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

                    let _nregs = pf.num_registers().unwrap();
                    let _shmem_static_nbytes = pf.shared_memory_bytes().unwrap();
                    let _binary_version = pf.binary_version().unwrap();

                    self.instrument_function_if_needed(&mut pf);

                    if *self.active_region.lock().unwrap() {
                        nvbit_rs::enable_instrumented(
                            &mut self.ctx.lock().unwrap(),
                            &mut pf,
                            true,
                            true,
                        );
                        **self.pstop_report.lock().unwrap() = false;
                    } else {
                        nvbit_rs::enable_instrumented(
                            &mut self.ctx.lock().unwrap(),
                            &mut pf,
                            false,
                            true,
                        );
                        **self.pstop_report.lock().unwrap() = true;
                    }

                    // char buffer[1024];
                    // sprintf(buffer, std::string(traces_location+"/kernel-%d.trace").c_str(), kernelid);

                    // if (!stop_report) {
                    //   resultsFile = fopen(buffer, "w");

                    //   printf("Writing results to %s\n", buffer);

                    //   fprintf(resultsFile, "-kernel name = %s\n",
                    //           nvbit_get_func_name(ctx, p->f, true));
                    //   fprintf(resultsFile, "-kernel id = %d\n", kernelid);
                    //   fprintf(resultsFile, "-grid dim = (%d,%d,%d)\n", p->gridDimX,
                    //           p->gridDimY, p->gridDimZ);
                    //   fprintf(resultsFile, "-block dim = (%d,%d,%d)\n", p->blockDimX,
                    //           p->blockDimY, p->blockDimZ);
                    //   fprintf(resultsFile, "-shmem = %d\n",
                    //           shmem_static_nbytes + p->sharedMemBytes);
                    //   fprintf(resultsFile, "-nregs = %d\n", nregs);
                    //   fprintf(resultsFile, "-binary version = %d\n", binary_version);
                    //   fprintf(resultsFile, "-cuda stream id = %lu\n", (uint64_t)p->hStream);
                    //   fprintf(resultsFile, "-shmem base_addr = 0x%016lx\n",
                    //           (uint64_t)nvbit_get_shmem_base_addr(ctx));
                    //   fprintf(resultsFile, "-local mem base_addr = 0x%016lx\n",
                    //           (uint64_t)nvbit_get_local_mem_base_addr(ctx));
                    //   fprintf(resultsFile, "-nvbit version = %s\n", NVBIT_VERSION);
                    //   fprintf(resultsFile, "-accelsim tracer version = %s\n", TRACER_VERSION);
                    //   fprintf(resultsFile, "\n");

                    //   fprintf(resultsFile,
                    //           "#traces format = threadblock_x threadblock_y threadblock_z "
                    //           "warpid_tb PC mask dest_num [reg_dests] opcode src_num "
                    //           "[reg_srcs] mem_width [adrrescompress?] [mem_addresses]\n");
                    //   fprintf(resultsFile, "\n");
                    // }

                    *self.kernelid.lock().unwrap() += 1;
                }
            }
            bindings::nvbit_api_cuda_t::API_CUDA_cuProfilerStart => {
                // if is_exit && !*self.active_from_start.lock().unwrap() {
                if is_exit && !self.active_from_start {
                    *self.active_region.lock().unwrap() = true;
                }
            }
            bindings::nvbit_api_cuda_t::API_CUDA_cuProfilerStop => {
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
    CONTEXTS
        .write()
        .unwrap()
        .entry(ctx.handle())
        .or_insert_with(|| Instrumentor::new(ctx));
}

/// NVBIT callback when CUDA context is terminated.
#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    if let Some(instrumentor) = CONTEXTS.read().unwrap().get(&ctx.handle()) {
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

        instrumentor.at_ctx_term();

        println!(
            "done after {:?}",
            Instant::now().duration_since(instrumentor.start)
        );
    }
    // this will lead to problems:
    // CONTEXTS.write().unwrap().remove(&ctx.handle());
}

impl<'c> Instrumentor<'c> {
    fn at_ctx_term(&self) {
        dbg!(**self.pstop_report.lock().unwrap());
        dbg!(**self.ptotal_dynamic_instr_counter.lock().unwrap());
        dbg!(**self.preported_dynamic_instr_counter.lock().unwrap());
    }
}
