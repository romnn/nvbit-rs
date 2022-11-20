#![allow(warnings)]

mod instrument_inst;

use anyhow::Result;
use lazy_static::lazy_static;
use nvbit_sys::bindings;
use parking_lot::ReentrantMutex;
use rustacuda::memory::UnifiedBox;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};

mod common {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// 1 MiB = 2**20
const CHANNEL_SIZE: usize = 1 << 20;
// const CHANNEL_SIZE: usize = 1 << 10;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    // ctx: nvbit_rs::Context<'c>,
    already_instrumented: Mutex<HashSet<nvbit_rs::Function<'c>>>,
    // pchannel_dev: *mut nvbit_sys::utils::ChannelDev,
    dev_channel: Mutex<nvbit_rs::DeviceChannel<common::inst_trace_t>>,
    host_channel: Arc<Mutex<nvbit_rs::HostChannel<common::inst_trace_t>>>,
    // host_channel: nvbit_rs::HostChannel<common::inst_trace_t>,
    // channel_host: Mutex<cxx::UniquePtr<nvbit_sys::utils::ChannelHost>>,
    ptotal_dynamic_instr_counter: Mutex<UnifiedBox<u64>>,
    preported_dynamic_instr_counter: Mutex<UnifiedBox<u64>>,
    pstop_report: Mutex<UnifiedBox<bool>>,
    // pchannel_host: Pin<&'a mut ChannelHost>,
    kernelid: Mutex<u64>,
    // first_call: bool,
    terminate_after_limit_number_of_kernels_reached: bool,
    // recv_thread_started: Mutex<bool>,
    // recv_thread_receiving: Mutex<bool>,
    // recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    instr_begin_interval: u32,
    instr_end_interval: u32,
    active_from_start: bool,
    active_region: Mutex<bool>,
    skip_flag: Mutex<bool>,
    dynamic_kernel_limit_start: u64,
    dynamic_kernel_limit_end: u64,
}

impl<'c> Instrumentor<'c> {
    // #[inline(never)]
    // pub fn init(&self) {
    //     println!("stats are ready");
    // }

    pub fn new(ctx: nvbit_rs::Context<'c>) -> Self {
        // let mut devPtr: bindings::CUdeviceptr = std::ptr::null_mut::<u64>() as _;
        // let mut devPtr: *mut u64 = std::ptr::null_mut::<u64>() as _;
        // let total_dynamic_instr_counter: u64 = 0;
        // let pinned: Pin<&mut u64> = Pin::new(&mut total_dynamic_instr_counter);
        // // avoid dropping?
        // std::mem::forget(pinned);
        // // let mut total_dynamic_instr_counter: Box<u64> = Box::new(0);
        // let mut devPtr: *mut u64 = &mut *total_dynamic_instr_counter as _;

        // let pchannel_dev = unsafe { nvbit_sys::utils::new_managed_dev_channel() };
        let mut dev_channel = nvbit_rs::DeviceChannel::new();

        // let pchannel_dev =
        //     unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };
        // panic!("size of channel dev: {}", dev_channel_size());

        // let pchannel_dev =
        //     unsafe { cuda_malloc_unified::<ChannelDev>(1).expect("cuda malloc unified") };

        // panic!("size of channel dev: {}", std::mem::size_of(pchannel_dev.as_ref().unwrap()));

        // let pchannel_dev = unsafe { &mut channel_dev as *mut ChannelDev };
        // let mut channel_host =
        //     unsafe { nvbit_sys::utils::new_host_channel(42, CHANNEL_SIZE as i32, pchannel_dev) };
        let mut host_channel = Arc::new(Mutex::new(
            nvbit_rs::HostChannel::<common::inst_trace_t>::new(42, CHANNEL_SIZE, &mut dev_channel),
        ));

        // Arc::new(Mutex::new());
        let rx = host_channel.lock().unwrap().read();
        // // let host_channel_clone = host_channel.clone();
        let host_channel_clone = host_channel.clone();
        let recv_thread = Mutex::new(Some(std::thread::spawn(move || {
            while let Ok(packet) = rx.recv() {
                // when we get this cta_id_x it means the kernel has completed
                if (packet.cta_id_x == -1) {
                    // *self.recv_thread_receiving.lock().unwrap() = false;
                    host_channel_clone.lock().unwrap().stop();
                    // host_channel.lock().unwrap().stop();
                    // break;
                }
                // println!("{:?}", packet);
            }
        })));

        // pchannel_dev should be initialized now???

        // channel_host.init(0, CHANNEL_SIZE, pchannel_dev, NULL);
        // channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);

        let ptotal_dynamic_instr_counter = UnifiedBox::new(0u64).expect("cuda malloc unified");
        // unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };

        let preported_dynamic_instr_counter = UnifiedBox::new(0u64).expect("cuda malloc unified");
        // unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };

        let pstop_report = UnifiedBox::new(false).expect("cuda malloc unified");
        // unsafe { cuda_malloc_unified::<bool>(1).expect("cuda malloc unified") };

        // unsafe { *ptotal_dynamic_instr_counter = 0 };
        // unsafe { *preported_dynamic_instr_counter = 0 };
        // unsafe { *pstop_report = false };

        // let mut devPtr: *mut u64 = std::ptr::null_mut::<u64>() as _;
        // todo: run the cuda managed alloc here already

        // let pchannel_host = pchannel_host.pin_mut();

        Self {
            ctx: Mutex::new(ctx),
            // ctx,
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            // host_channel: Mutex::new(host_channel),
            host_channel,
            ptotal_dynamic_instr_counter: Mutex::new(ptotal_dynamic_instr_counter),
            preported_dynamic_instr_counter: Mutex::new(preported_dynamic_instr_counter),
            pstop_report: Mutex::new(pstop_report),
            kernelid: Mutex::new(1),
            // first_call: true,
            terminate_after_limit_number_of_kernels_reached: false,
            // recv_thread_started: Mutex::new(false),
            // recv_thread_receiving: Mutex::new(false),
            // recv_thread: Mutex::new(None),
            recv_thread, // : Mutex::new(None),
            instr_begin_interval: 0,
            instr_end_interval: u32::MAX,
            active_from_start: true,
            active_region: Mutex::new(true), // region of interest when active from start is 0
            skip_flag: Mutex::new(false),
            dynamic_kernel_limit_start: 0, // start from the begging kernel
            dynamic_kernel_limit_end: 0,   // // no limit
        }
    }
}

unsafe impl<'c> Send for Instrumentor<'c> {}
unsafe impl<'c> Sync for Instrumentor<'c> {}

// std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

type Contexts = RwLock<HashMap<nvbit_rs::ContextHandle<'static>, Instrumentor<'static>>>;

lazy_static! {
    // static ref instrumentor: Instrumentor = Stats::new();
    // static ref contexts: Mutex<HashMap<nvbit_rs::ContextHandle<'static>, Instrumentor<'static>>> =
    static ref contexts: Contexts =
        RwLock::new(HashMap::new());
        // ReentrantMutex::new(RefCell::new(HashMap::new()));
}

// lazy_static! {
//     static ref ALREADY_INSTRUMENTED: Mutex<HashSet<nvbit_rs::Function<'static>>> =
//         Mutex::new(HashSet::new());
// }

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    println!("nvbit_at_init");
}

impl<'c> Instrumentor<'c> {
    // pub fn instrument_function_if_needed<'c, 'f>(
    pub fn instrument_function_if_needed<'f>(
        &self,
        // ctx: &mut nvbit_rs::Context<'c>,
        func: &mut nvbit_rs::Function<'f>,
    ) {
        println!("instrument_function_if_needed");

        // test getting the id
        // let id = unsafe { Pin::new_unchecked(&mut *stats.pchannel_dev as &mut ChannelDev) }.get_id();
        // println!("dev channel id: {}", id);

        let mut related_functions =
            nvbit_rs::get_related_functions(&mut self.ctx.lock().unwrap(), func);

        for f in related_functions.iter_mut().chain([func]) {
            // "recording" function was instrumented,

            let mut f = nvbit_rs::Function::new(f.as_mut_ptr());
            let func_name = nvbit_rs::get_func_name(&mut self.ctx.lock().unwrap(), &mut f);
            let func_addr = nvbit_rs::get_func_addr(&mut f);

            // let mut instrumented_lock = self.already_instrumented; // .lock().unwrap();

            // todo: is it okay to clone a function?
            // this does allow concurrent mutable access :(
            if !self.already_instrumented.lock().unwrap().insert(f.clone()) {
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
            println!("found {} instructions", instrs.len());
            println!("instructions {:#?} ", instrs);

            let mut instr_count: u32 = 0;

            // iterate on all the static instructions in the function
            for instr in instrs.iter_mut() {
                if instr_count < self.instr_begin_interval || instr_count >= self.instr_end_interval
                {
                    instr_count += 1;
                    continue;
                }

                // instr.print_decoded();

                let mut opcode_to_id_map: HashMap<String, usize> = HashMap::new();
                let mut id_to_opcode_map: HashMap<usize, String> = HashMap::new();

                let opcode = instr.opcode().expect("has opcode");
                // dbg!(&opcode);

                if !opcode_to_id_map.contains_key(opcode) {
                    let opcode_id = opcode_to_id_map.len();
                    opcode_to_id_map.insert(opcode.to_string(), opcode_id);
                    id_to_opcode_map.insert(opcode_id, opcode.to_string());
                }

                let opcode_id = opcode_to_id_map[opcode];
                // dbg!(&opcode_id);

                instr.insert_call("instrument_inst", nvbit_rs::InsertionPoint::Before);

                let mut inst_args = instrument_inst::InstrumentInstArgs::default();
                // dbg!(&inst_args);

                inst_args.opcode_id = opcode_id as i32;

                inst_args.vpc = instr.offset();

                // check all operands.
                // For now, we ignore constant, TEX, predicates and unified registers.
                // We only report vector registers
                let mut src_oprd: [ffi::c_int; common::MAX_SRC as usize] =
                    [-1; common::MAX_SRC as usize];
                let mut srcNum: usize = 0;
                let mut dst_oprd: ffi::c_int = -1;
                let mut mem_oper_idx: ffi::c_int = -1;

                // find dst reg and handle the special case if the oprd[0] is mem
                // (e.g. store and RED)
                let num_operands = instr.num_operands();
                if num_operands > 0 {
                    let first_operand = instr.operand(0).unwrap().test();
                    match first_operand.type_ {
                        bindings::InstrType_OperandType::REG => {
                            dst_oprd = unsafe { first_operand.u.reg.num };
                        }
                        bindings::InstrType_OperandType::MREF => {
                            src_oprd[0] = unsafe { first_operand.u.mref.ra_num };
                            mem_oper_idx = 0;
                            srcNum += 1;
                        }
                        _ => {
                            // skip anything else (constant and predicates)
                        }
                    }
                }

                // find src regs and mem
                for i in 1..(common::MAX_SRC.min(num_operands as u32)) {
                    assert!(i < common::MAX_SRC);
                    assert!(i < num_operands as u32);
                    let op = instr.operand(i as usize).unwrap().test();
                    match op.type_ {
                        bindings::InstrType_OperandType::MREF => {
                            // mem is found
                            assert!(srcNum < common::MAX_SRC as usize);
                            src_oprd[srcNum] = unsafe { op.u.mref.ra_num };
                            srcNum += 1;
                            // TODO: handle LDGSTS with two mem refs
                            assert!(mem_oper_idx == -1); // ensure one memory operand per inst
                            mem_oper_idx += 1;
                        }
                        bindings::InstrType_OperandType::REG => {
                            // reg is found
                            assert!(srcNum < common::MAX_SRC as usize);
                            src_oprd[srcNum] = unsafe { op.u.reg.num };
                            srcNum += 1;
                        }
                        _ => {
                            // skip anything else (constant and predicates)
                        }
                    };
                }

                // mem addresses info
                if (mem_oper_idx >= 0) {
                    inst_args.is_mem = true;
                    inst_args.addr = 0;
                    inst_args.width = instr.size() as i32;
                } else {
                    inst_args.is_mem = false;
                    inst_args.addr = 1; // todo: was -1
                    inst_args.width = 1; // todo: was -1
                }

                // reg info
                inst_args.desReg = dst_oprd;
                // we set the default value to -1
                inst_args.srcReg1 = src_oprd[0];
                inst_args.srcReg2 = src_oprd[1];
                inst_args.srcReg3 = src_oprd[2];
                inst_args.srcReg4 = src_oprd[3];
                inst_args.srcReg5 = src_oprd[4];
                inst_args.srcNum = srcNum as i32;

                // add pointer to channel_dev and other counters
                // inst_args.pchannel_dev = (self.pchannel_dev as *mut ffi::c_void) as u64;
                inst_args.pchannel_dev = self.dev_channel.lock().unwrap().as_mut_ptr() as u64;

                inst_args.ptotal_dynamic_instr_counter = self
                    .ptotal_dynamic_instr_counter
                    .lock()
                    .unwrap()
                    .as_unified_ptr()
                    .as_raw_mut() as u64;
                // *mut ffi::c_void) as u64;
                // (stats.ptotal_dynamic_instr_counter as *mut ffi::c_void) as u64;

                inst_args.preported_dynamic_instr_counter =
                    self.preported_dynamic_instr_counter
                        .lock()
                        .unwrap()
                        .as_unified_ptr()
                        .as_raw_mut() as u64;
                // as *mut ffi::c_void) as u64;

                inst_args.pstop_report = self
                    .pstop_report
                    .lock()
                    .unwrap()
                    .as_unified_ptr()
                    .as_raw_mut() as u64;
                // (stats.pstop_report.as_unified_ptr() as *mut ffi::c_void) as u64;
                // dbg!(&inst_args);

                inst_args.instrument(instr);
                instr_count += 1;
            }
        }
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    // mut ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: bindings::nvbit_api_cuda_t,
    event_name: *const ffi::c_char,
    params: *mut ffi::c_void,
    pStatus: *mut bindings::CUresult,
) {
    let is_exit = is_exit != 0;
    let event_name = unsafe { ffi::CStr::from_ptr(event_name).to_str().unwrap() };
    println!(
        "nvbit_at_cuda_event: {:?} (is_exit = {})",
        event_name, is_exit
    );
    // let mut ctx_map = ;
    // let ctx_lock = contexts.lock();
    // let mut test = lock.get_mut();
    // println!("ctx: {:p}", ctx.as_ptr());
    // let mut ctx_map = ctx_lock.borrow();
    // println!("got the lock for {}", event_name);

    if let Some(instrumentor) = contexts.read().unwrap().get(&ctx.handle()) {
        // .expect("unknown context");
        // println!("ctx: {:p}", instrumentor.ctx.as_ptr());
        instrumentor.at_cuda_event(is_exit, cbid, event_name, params, pStatus);
        println!("handled cuda event {}", event_name);
    } else {
        println!("skip event {}", event_name);
    }
}

impl<'c> Instrumentor<'c> {
    pub fn at_cuda_event(
        &self,
        // mut ctx: nvbit_rs::Context<'static>,
        is_exit: bool,
        cbid: bindings::nvbit_api_cuda_t,
        event_name: &str,
        params: *mut ffi::c_void,
        pStatus: *mut bindings::CUresult,
    ) {
        // println!(
        //     "nvbit_at_cuda_event: {:?} (is_exit = {})",
        //     event_name, is_exit
        // );

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

        if self.active_from_start && self.dynamic_kernel_limit_start <= 1 {
            *self.active_region.lock().unwrap() = true;
        } else {
            if self.active_from_start {
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
                    let p: &mut bindings::cuMemcpyHtoD_v2_params =
                        unsafe { &mut *(params as *mut bindings::cuMemcpyHtoD_v2_params) };

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
                let p: &mut bindings::cuLaunchKernel_params =
                    unsafe { &mut *(params as *mut bindings::cuLaunchKernel_params) };
                dbg!(&p);
                let mut pf = nvbit_rs::Function::new(p.f);

                if !is_exit {
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
                        println!("i decided to terminate");
                        std::process::exit(0);
                    }

                    use rustacuda::function::FunctionAttribute;

                    // let function = module.get_function(&name)?;
                    // let function = rustacuda::function::Function::from_raw(0); // new(0, 0);
                    // let shared_memory =
                    //     function.get_attribute(FunctionAttribute::SharedMemorySizeBytes)?;

                    let mut nregs: ffi::c_int = 0;
                    let result = unsafe {
                        bindings::cuFuncGetAttribute(
                            &mut nregs,
                            bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                            p.f,
                        )
                    };
                    dbg!(&result);
                    dbg!(&nregs);

                    let mut shmem_static_nbytes: ffi::c_int = 0;
                    let result = unsafe {
                        bindings::cuFuncGetAttribute(
                            &mut shmem_static_nbytes,
                            bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            p.f,
                        )
                    };
                    dbg!(&result);
                    dbg!(&shmem_static_nbytes);

                    let mut binary_version: ffi::c_int = 0;
                    let result = unsafe {
                        bindings::cuFuncGetAttribute(
                            &mut binary_version,
                            bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                            p.f,
                        )
                    };
                    dbg!(&result);
                    dbg!(&binary_version);

                    // self.instrument_function_if_needed(&mut self.ctx, &mut pf);
                    println!("instrumenting");
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
                    // *self.recv_thread_receiving.lock().unwrap() = true;
                } else {
                    // is exit
                    *self.skip_flag.lock().unwrap() = true;
                    // make sure current kernel is completed
                    // unsafe { common::flush_channel(self.dev_channel.as_mut_ptr() as *mut ffi::c_void) };
                    unsafe {
                        common::flush_channel(
                            self.dev_channel.lock().unwrap().as_mut_ptr() as *mut _
                        )
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

// pub unsafe fn cuda_malloc_unified<T>(count: usize) -> Result<*mut T> {
//     // CudaResult<UnifiedPointer<T>> {
//     let size = count.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
//     dbg!(&size);
//     if size == 0 {
//         panic!("InvalidMemoryAllocation");
//     }

//     let mut ptr: *mut ffi::c_void = std::ptr::null_mut();
//     let result = unsafe {
//         bindings::cuMemAllocManaged(
//             &mut ptr as *mut *mut ffi::c_void as *mut u64,
//             size,
//             bindings::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL as u32,
//         )
//     };
//     if result != bindings::cudaError_enum::CUDA_SUCCESS {
//         panic!("CUDA ERROR: {:?}", result);
//     }
//     // .to_result()?;
//     Ok(ptr as *mut T)
//     // Ok(UnifiedPointer::wrap(ptr as *mut T))
// }

impl<'c> Instrumentor<'c> {
    pub fn read_channel(&mut self) {
        println!("read channel thread started");

        // println!(
        //     "creating a buffer of size {} ({} MiB)",
        //     CHANNEL_SIZE as usize,
        //     CHANNEL_SIZE as f32 / 1024.0f32.powi(2)
        // );
        // // let mut recv_buffer = buffer::Buffer::with_size(CHANNEL_SIZE as usize);
        // // let mut recv_buffer = buffer::alloc_box_buffer(CHANNEL_SIZE as usize);
        // let mut recv_buffer = nvbit_rs::buffer::Buffer::new(CHANNEL_SIZE);
        // println!("{:02X?}", &recv_buffer[0..10]);
        // println!("{:02X?}", &recv_buffer[(CHANNEL_SIZE - 10)..CHANNEL_SIZE]);
        // // let recv_buffer_ptr =
        // // std::mem::forget(recv_buffer);

        // // let recv_buffer: [*mut ffi::c_void; CHANNEL_SIZE as usize] =
        // //     [std::ptr::null_mut() as *mut ffi::c_void; CHANNEL_SIZE as usize];

        // let packet_size = std::mem::size_of::<common::inst_trace_t>();
        // let mut packet_count = 0;

        // while *self.recv_thread_started.lock().unwrap() {
        //     // loop {
        //     // println!("receiving");
        //     // let num_recv_bytes: u32 = 0;
        //     // while (recv_thread_receiving &&
        //     let num_recv_bytes = 0;
        //     // let num_recv_bytes = unsafe {
        //     //     self.host_channel.lock().unwrap().pin_mut().recv(
        //     //         recv_buffer.as_mut_ptr() as *mut nvbit_sys::utils::c_void,
        //     //         // *recv_buffer.as_mut_ptr() as *mut c_void,
        //     //         CHANNEL_SIZE as u32,
        //     //     )
        //     // };
        //     if *self.recv_thread_receiving.lock().unwrap() && num_recv_bytes > 0 {
        //         // println!("received {} bytes", num_recv_bytes);
        //         let mut num_processed_bytes: usize = 0;
        //         while num_processed_bytes < num_recv_bytes as usize {
        //             // let ma: &mut common::inst_trace_t = unsafe {
        //             let packet_bytes =
        //                 &recv_buffer[num_processed_bytes..num_processed_bytes + packet_size];
        //             // println!("{:02X?}", packet_bytes);

        //             assert_eq!(
        //                 std::mem::size_of::<common::inst_trace_t>(),
        //                 packet_bytes.len()
        //             );
        //             let ma: common::inst_trace_t =
        //                 unsafe { std::ptr::read(packet_bytes.as_ptr() as *const _) };
        //             println!("received from channel: {:#?}", &ma);

        //             // when we get this cta_id_x it means the kernel has completed
        //             if (ma.cta_id_x == -1) {
        //                 *self.recv_thread_receiving.lock().unwrap() = false;
        //                 break;
        //             }

        //             println!("size of common::inst_trace_t: {}", packet_size);
        //             num_processed_bytes += packet_size;
        //             packet_count += 1;
        //         }
        //     }
        // }
        // println!("received {} packets", packet_count);
        // todo: currently our buffer is mem::forget so we must free here
    }
}

// void *recv_thread_fun(void *) {
//   char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

//   while (recv_thread_started) {
//     uint32_t num_recv_bytes = 0;
//     if (recv_thread_receiving &&
//         (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
//       uint32_t num_processed_bytes = 0;
//       while (num_processed_bytes < num_recv_bytes) {
//         inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

//         // when we get this cta_id_x it means the kernel has completed
//         if (ma->cta_id_x == -1) {
//           recv_thread_receiving = false;
//           break;
//         }

//         fprintf(resultsFile, "%d ", ma->cta_id_x);
//         fprintf(resultsFile, "%d ", ma->cta_id_y);
//         fprintf(resultsFile, "%d ", ma->cta_id_z);
//         fprintf(resultsFile, "%d ", ma->warpid_tb);
//         if (print_core_id) {
//           fprintf(resultsFile, "%d ", ma->sm_id);
//           fprintf(resultsFile, "%d ", ma->warpid_sm);
//         }
//         fprintf(resultsFile, "%04x ", ma->vpc); // Print the virtual PC
//         fprintf(resultsFile, "%08x ", ma->active_mask & ma->predicate_mask);
//         if (ma->GPRDst >= 0) {
//           fprintf(resultsFile, "1 ");
//           fprintf(resultsFile, "R%d ", ma->GPRDst);
//         } else
//           fprintf(resultsFile, "0 ");

//         // Print the opcode.
//         fprintf(resultsFile, "%s ", id_to_opcode_map[ma->opcode_id].c_str());
//         unsigned src_count = 0;
//         for (int s = 0; s < MAX_SRC; s++) // GPR srcs count.
//           if (ma->GPRSrcs[s] >= 0)
//             src_count++;
//         fprintf(resultsFile, "%d ", src_count);

//         for (int s = 0; s < MAX_SRC; s++) // GPR srcs.
//           if (ma->GPRSrcs[s] >= 0)
//             fprintf(resultsFile, "R%d ", ma->GPRSrcs[s]);

//         // print addresses
//         std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
//         if (ma->is_mem) {
//           std::istringstream iss(id_to_opcode_map[ma->opcode_id]);
//           std::vector<std::string> tokens;
//           std::string token;
//           while (std::getline(iss, token, '.')) {
//             if (!token.empty())
//               tokens.push_back(token);
//           }
//           fprintf(resultsFile, "%d ", get_datawidth_from_opcode(tokens));

//           bool base_stride_success = false;
//           uint64_t base_addr = 0;
//           int stride = 0;
//           std::vector<long long> deltas;

//           if (enable_compress) {
//             // try base+stride format
//             base_stride_success =
//                 base_stride_compress(ma->addrs, mask, base_addr, stride);
//             if (!base_stride_success) {
//               // if base+stride fails, try base+delta format
//               base_delta_compress(ma->addrs, mask, base_addr, deltas);
//             }
//           }

//           if (base_stride_success && enable_compress) {
//             // base + stride format
//             fprintf(resultsFile, "%u 0x%llx %d ", address_format::base_stride,
//                     base_addr, stride);
//           } else if (!base_stride_success && enable_compress) {
//             // base + delta format
//             fprintf(resultsFile, "%u 0x%llx ", address_format::base_delta,
//                     base_addr);
//             for (int s = 0; s < deltas.size(); s++) {
//               fprintf(resultsFile, "%lld ", deltas[s]);
//             }
//           } else {
//             // list all the addresses
//             fprintf(resultsFile, "%u ", address_format::list_all);
//             for (int s = 0; s < 32; s++) {
//               if (mask.test(s))
//                 fprintf(resultsFile, "0x%016lx ", ma->addrs[s]);
//             }
//           }
//         } else {
//           fprintf(resultsFile, "0 ");
//         }

//         fprintf(resultsFile, "\n");

//         num_processed_bytes += sizeof(inst_trace_t);
//       }
//     }
//   }
//   free(recv_buffer);
//   return NULL;
// }

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(mut ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_init");
    // let ctx_lock = contexts.write().unwrap();

    // // setup stats
    // stats.init();

    // // cudaMallocManaged((void**)&(elm->name), sizeof(char) * (strlen("hello") + 1) );
    // // start reading from the channel
    // unsafe { recv_thread_started = true };
    // unsafe { recv_thread = Some(std::thread::spawn(read_channel)) };
    // let ctx = Arc::new(ctx);
    // println!("nvbit_at_ctx_init");
    // let ctx_lock = contexts.lock();
    // let mut ctx_map = ctx_lock.borrow_mut();
    // println!("ctx: {:p}", ctx.as_ptr());
    // let instrumentor = contexts
    contexts
        .write()
        .unwrap()
        .entry(ctx.handle())
        .or_insert_with(|| Instrumentor::new(ctx));
    // println!("ctx: {:p}", instrumentor.ctx.as_ptr());
    // instrumentor.at_ctx_init();
}

impl<'c> Instrumentor<'c> {
    pub fn at_ctx_init(&self) {
        // setup stats
        // stats.init();

        // cudaMallocManaged((void**)&(elm->name), sizeof(char) * (strlen("hello") + 1) );
        // start reading from the channel
        // *self.recv_thread_started.lock().unwrap() = true;
        // *self.recv_thread.lock().unwrap() = Some(std::thread::spawn(|| self.read_channel()));
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(mut ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    // let ctx_map = contexts.read().unwrap();
    // let ctx_lock = contexts.lock();
    // must borrow as mut, but will also cause at event to be called, which then cannot borrow
    // let mut ctx_map = ctx_lock.borrow();
    // println!("ctx: {:p}", ctx.as_ptr());
    if let Some(instrumentor) = contexts.read().unwrap().get(&ctx.handle()) {
        // .expect("unknown context");
        // println!("ctx: {:p}", instrumentor.ctx.as_ptr());
        // instrumentor.at_ctx_term();
        // drop of the channels will run here
        // IMPORTANT: skip
        instrumentor.host_channel.lock().unwrap().stop();
        if let Some(recv_thread) = instrumentor.recv_thread.lock().unwrap().take() {
            recv_thread.join().expect("join receiver thread");
        }
    }
    // ctx_lock.borrow_mut().remove(&ctx.handle());
    // contexts.write().unwrap().remove(&ctx.handle());
    println!("done");
}

impl<'c> Instrumentor<'c> {
    pub fn at_ctx_term(&self) {
        // unsafe {
        // if self.recv_thread_started {
        //     self.recv_thread_started = false;
        // }
        // self.host_channel.stop().expect("finish receiving from channel");
        // if let Some(handle) = self.recv_thread.lock().unwrap().take() {
        //     handle.join().expect("join receiver thread");
        // }
        // }
        dbg!(**self.pstop_report.lock().unwrap());
        dbg!(**self.ptotal_dynamic_instr_counter.lock().unwrap());
        dbg!(**self.preported_dynamic_instr_counter.lock().unwrap());
    }
}
