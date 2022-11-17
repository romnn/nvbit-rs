#![allow(warnings)]

mod buffer;
mod instrument_inst;

use anyhow::Result;
use lazy_static::lazy_static;
use libc;
use libloading::{Library, Symbol};
use nvbit_sys::bindings::{self, CUcontext, CUfunction};
use nvbit_sys::nvbit::*;
use nvbit_sys::utils::*;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

mod common {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct CUfunctionKey {
    // inner: Arc<CUfunction>,
    // inner: Mutex<CUfunction>,
    // inner: *const bindings::CUfunc_st,
    // inner: Arc<*const bindings::CUfunc_st>,
    // inner: std::ptr::NonNull<*mut bindings::CUfunc_st>,
    inner: std::ptr::NonNull<bindings::CUfunc_st>,
}

// todo: make sure this is done in the safe wrapper at some point
unsafe impl Send for CUfunctionKey {}

lazy_static! {
    static ref ALREADY_INSTRUMENTED: Mutex<HashSet<CUfunctionKey>> = Mutex::new(HashSet::new());
}

static mut kernelid: u64 = 1;
static mut first_call: bool = true;
static mut terminate_after_limit_number_of_kernels_reached: bool = false;

static mut recv_thread_started: bool = false;
static mut recv_thread_receiving: bool = false;
static mut recv_thread: Option<std::thread::JoinHandle<()>> = None;

static mut instr_begin_interval: u32 = 0;
static mut instr_end_interval: u32 = u32::MAX;
static mut active_from_start: bool = true;
// used to select region of interest when active from start is 0
static mut active_region: bool = true;
static mut skip_flag: bool = false;

// 0 means start from the begging kernel
static mut dynamic_kernel_limit_start: u64 = 0;
// 0 means no limit
static mut dynamic_kernel_limit_end: u64 = 0;

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    println!("nvbit_at_init");
}

// fn instrument_function_if_needed(ctx: CUcontext , func: CUfunction) {
// fn instrument_function_if_needed(ctx: *mut bindings::CUctx_st, func: *mut bindings::CUfunc_st) {
fn instrument_function_if_needed(ctx: CUcontext, func: CUfunction) {
    println!("instrument_function_if_needed");

    // test getting the id
    // let id = unsafe { Pin::new_unchecked(&mut *stats.pchannel_dev as &mut ChannelDev) }.get_id();
    // println!("dev channel id: {}", id);

    // let testctx = TestCUcontext { inner: ctx };
    // let testfunc = TestCUfunction { inner: func };
    let related_functions = unsafe { rust_nvbit_get_related_functions(ctx, func) };
    // let related_functions = nvbit_get_related_functions(ctx, func);
    // std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    // add kernel itself to the related function vector */
    // let mut already_instrumented = std::collections::HashSet::new();
    // std::unordered_set<CUfunction> already_instrumented;

    // todo: perform this in the safe wrapper
    let mut related_functions: Vec<&CUfunctionShim> = related_functions.into_iter().collect();

    let current_func = unsafe { CUfunctionShim::wrap(func) };
    related_functions.push(unsafe { &current_func });
    println!("number of related functions: {:?}", related_functions.len());

    for f in &mut related_functions {
        // "recording" function was instrumented,
        // if set insertion failed we have already encountered this function
        let func_name: *const libc::c_char = unsafe {
            // bindings::nvbit_get_func_name(ctx.as_mut_ptr(), f.as_mut_ptr(), false)
            bindings::nvbit_get_func_name(ctx, f.as_mut_ptr(), false)
        };

        let func_name = unsafe { ffi::CStr::from_ptr(func_name).to_string_lossy() };

        let func_addr: u64 = unsafe {
            // bindings::nvbit_get_func_addr(f.ptr)
            bindings::nvbit_get_func_addr(f.as_mut_ptr())
        };

        let mut instrumented_lock = ALREADY_INSTRUMENTED.lock().unwrap();
        let key = unsafe {
            CUfunctionKey {
                inner: std::ptr::NonNull::new_unchecked(f.as_mut_ptr()),
            }
        };
        if !instrumented_lock.insert(key) {
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

        // const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        // const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        let instrs = unsafe { rust_nvbit_get_instrs(ctx, func) };
        // println!("found {} instructions", instrs.len());
        // println!("instructions {:#?} ", instrs);

        let mut instr_count: u32 = 0;

        // iterate on all the static instructions in the function
        for instr in instrs.iter() {
            if instr_count < unsafe { instr_begin_interval }
                || instr_count >= unsafe { instr_end_interval }
            {
                instr_count += 1;
                continue;
            }

            // unsafe { instr.as_mut_ref().printDecoded() };

            // std::map<std::string, int> opcode_to_id_map;
            let mut opcode_to_id_map: HashMap<String, usize> = HashMap::new();
            let mut id_to_opcode_map: HashMap<usize, String> = HashMap::new();

            let opcode: String = {
                let op = unsafe { instr.as_mut_ref().getOpcode() };
                unsafe { ffi::CStr::from_ptr(op).to_string_lossy().to_string() }
            };

            if !opcode_to_id_map.contains_key(&opcode) {
                let opcode_id = opcode_to_id_map.len();
                opcode_to_id_map.insert(opcode.clone(), opcode_id);
                id_to_opcode_map.insert(opcode_id, opcode.clone());
            }

            let opcode_id = opcode_to_id_map[&opcode];

            // insert call to the instrumentation function
            unsafe {
                bindings::nvbit_insert_call(
                    unsafe { instr.as_ptr() },
                    ffi::CString::new("instrument_inst").unwrap().as_ptr(),
                    bindings::ipoint_t::IPOINT_BEFORE,
                );
            }
            let mut inst_args = instrument_inst::InstrumentInstArgs::default();

            // pass predicate value */
            // unsafe {
            //     bindings::nvbit_add_call_arg_guard_pred_val(instr.as_ptr(), false);
            // }
            // inst_args.pred = false as libc::c_int;

            // send opcode and pc */
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val32(
            //         // instr.as_ref().get_ref() as *const Instr,
            //         unsafe { instr.as_ptr() },
            //         opcode_id as u32,
            //         false,
            //     );
            // }
            inst_args.opcode_id = opcode_id as i32;

            inst_args.vpc = unsafe { instr.as_mut_ref().getOffset() };
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val32(unsafe { instr.as_ptr() }, offset, false);
            // }

            // check all operands.
            // For now, we ignore constant, TEX, predicates and unified registers.
            // We only report vector registers
            let mut src_oprd: [libc::c_int; common::MAX_SRC as usize] =
                [-1; common::MAX_SRC as usize];
            let mut srcNum: usize = 0;
            let mut dst_oprd: libc::c_int = -1;
            let mut mem_oper_idx: libc::c_int = -1;

            // find dst reg and handle the special case if the oprd[0] is mem
            // (e.g. store and RED)

            let num_operands = unsafe { instr.as_mut_ref().getNumOperands() };
            if num_operands > 0 {
                let first_operand = unsafe { *instr.as_mut_ref().getOperand(0) };
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
                let op = unsafe { *instr.as_mut_ref().getOperand(i as i32) };
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
                inst_args.width = unsafe { instr.as_mut_ref().getSize() };
                // unsafe {
                //     bindings::nvbit_add_call_arg_const_val32(unsafe { instr.as_ptr() }, 1, false);
                //     // this is why mref function must be part of the binary
                //     bindings::nvbit_add_call_arg_mref_addr64(unsafe { instr.as_ptr() }, 0, false);
                //     let size =
                //     bindings::nvbit_add_call_arg_const_val32(
                //         unsafe { instr.as_ptr() },
                //         size as u32,
                //         false,
                //     );
                // }
            } else {
                inst_args.is_mem = false;
                inst_args.addr = 1; // todo: was -1
                inst_args.width = 1; // todo: was -1

                // unsafe {
                //     bindings::nvbit_add_call_arg_const_val32(unsafe { instr.as_ptr() }, 0, false);
                //     bindings::nvbit_add_call_arg_const_val64(unsafe { instr.as_ptr() }, 1, false);
                //     // todo: should be -1
                //     bindings::nvbit_add_call_arg_const_val32(unsafe { instr.as_ptr() }, 1, false);
                // }
            }

            // reg info
            inst_args.desReg = dst_oprd;
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val32(
            //         unsafe { instr.as_ptr() },
            //         dst_oprd as u32,
            //         false,
            //     );
            // }
            // we set the default value to -1
            inst_args.srcReg1 = src_oprd[0];
            inst_args.srcReg2 = src_oprd[1];
            inst_args.srcReg3 = src_oprd[2];
            inst_args.srcReg4 = src_oprd[3];
            inst_args.srcReg5 = src_oprd[4];
            inst_args.srcNum = srcNum as i32;

            // for i in 0..srcNum {
            //     unsafe {
            //         bindings::nvbit_add_call_arg_const_val32(
            //             unsafe { instr.as_ptr() },
            //             src_oprd[i] as u32,
            //             false,
            //         );
            //     }
            // }
            // for i in srcNum..MAX_SRC {
            //     unsafe {
            //         // todo: should be -1
            //         bindings::nvbit_add_call_arg_const_val32(unsafe { instr.as_ptr() }, 1, false);
            //     }
            // }

            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val32(
            //         unsafe { instr.as_ptr() },
            //         srcNum as u32,
            //         false,
            //     );
            // }

            // add pointer to channel_dev and other counters

            // let pchannel_dev = unsafe { &mut channel_dev as *mut ChannelDev };
            inst_args.pchannel_dev = (stats.pchannel_dev as *mut libc::c_void) as u64;

            inst_args.ptotal_dynamic_instr_counter =
                (stats.ptotal_dynamic_instr_counter as *mut libc::c_void) as u64;
            inst_args.preported_dynamic_instr_counter =
                (stats.preported_dynamic_instr_counter as *mut libc::c_void) as u64;

            inst_args.pstop_report = (stats.pstop_report as *mut libc::c_void) as u64;
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val64(
            //         unsafe { instr.as_ptr() },
            //         0,
            //         // (uint64_t) & channel_dev,
            //         false,
            //     );
            // }
            // todo: only the the atomics for now, but define them in the
            //
            // CUdeviceptr = ::std::os::raw::c_ulonglong

            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val64(
            //         unsafe { instr.as_ptr() },
            //         // (uint64_t) & total_dynamic_instr_counter,
            //         // (&mut stats.total_dynamic_instr_counter as *mut bindings::CUdeviceptr) as u64,
            //         // &mut stats.total_dynamic_instr_counter as *mut *mut libc::c_void as u64, // libc::c_void),
            //         (stats.total_dynamic_instr_counter as *mut libc::c_void) as u64, // libc::c_void),
            //         false,
            //     );
            // }
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val64(
            //         unsafe { instr.as_ptr() },
            //         0,
            //         // (uint64_t) & reported_dynamic_instr_counter,
            //         false,
            //     );
            // }
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val64(
            //         unsafe { instr.as_ptr() },
            //         0,
            //         // (uint64_t) & stop_report,
            //         false,
            //     );
            // }
            unsafe {
                inst_args.instrument(unsafe { instr.as_ptr() });
            };
            instr_count += 1;
        }
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_cuda_event(
    // ctx: *mut bindings::CUctx_st,
    ctx: CUcontext,
    is_exit: libc::c_int,
    cbid: bindings::nvbit_api_cuda_t,
    event_name: *const libc::c_char,
    params: *mut ffi::c_void,
    pStatus: *mut bindings::CUresult,
) {
    let is_exit = is_exit != 0;
    println!("nvbit_at_cuda_event");
    // unsafe { say_hi() };
    // println!("is exit: {:?}", is_exit);
    let event_name = unsafe { ffi::CStr::from_ptr(event_name).to_string_lossy() };
    // println!("name: {:?} (is_exit = {})", event_name, is_exit);

    if unsafe { skip_flag } {
        return;
    }

    if unsafe { first_call } {
        unsafe { first_call = false }
    }

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

    if unsafe { active_from_start } && unsafe { dynamic_kernel_limit_start } <= 1 {
        unsafe { active_region = true }
    } else {
        if unsafe { active_from_start } {
            unsafe { active_region = false }
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

                // let p: &bindings::cuMemcpyHtoD_v2_params = (cuMemcpyHtoD_v2_params *)params;
                // dbg!(p);
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
            // dbg!(&p);
            // (bindings::cuMemcpyHtoD_v2_params *)params;
            // cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

            if !is_exit {
                if unsafe { active_from_start }
                    && unsafe { dynamic_kernel_limit_start } > 0
                    && unsafe { kernelid } == unsafe { dynamic_kernel_limit_start }
                {
                    unsafe { active_region = true }
                }

                if unsafe { terminate_after_limit_number_of_kernels_reached }
                    && unsafe { dynamic_kernel_limit_end } > 0
                    && unsafe { kernelid } > unsafe { dynamic_kernel_limit_end }
                {
                    println!("i decided to terminate");
                    std::process::exit(0);
                }

                let mut nregs: ffi::c_int = 0;
                unsafe {
                    bindings::cuFuncGetAttribute(
                        &mut nregs,
                        bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                        p.f,
                    );
                }
                // dbg!(&nregs);

                // CUDA_SAFECALL(
                //     cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

                let mut shmem_static_nbytes: ffi::c_int = 0;
                unsafe {
                    bindings::cuFuncGetAttribute(
                        &mut shmem_static_nbytes,
                        bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                        p.f,
                    );
                }
                // dbg!(&shmem_static_nbytes);

                // int shmem_static_nbytes;
                // CUDA_SAFECALL(cuFuncGetAttribute(
                //     &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

                let mut binary_version: ffi::c_int = 0;
                unsafe {
                    bindings::cuFuncGetAttribute(
                        &mut binary_version,
                        bindings::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                        p.f,
                    );
                }
                // dbg!(&binary_version);

                // int binary_version;
                // CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                //                                  CU_FUNC_ATTRIBUTE_BINARY_VERSION, p->f));
                // instrument_function_if_needed(ctx, CUfunction::wrap(p.f));
                instrument_function_if_needed(ctx, p.f);

                // nvbit_enable_instrumented(ctx, p.f, true, true);

                if unsafe { active_region } {
                    unsafe {
                        bindings::nvbit_enable_instrumented(ctx, p.f, true, true);
                    }
                    unsafe { *stats.pstop_report = false };
                } else {
                    unsafe {
                        bindings::nvbit_enable_instrumented(ctx, p.f, false, true);
                    }
                    unsafe { *stats.pstop_report = true };
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

                unsafe { kernelid += 1 }
                unsafe { recv_thread_receiving = true };
            } else {
                // is exit
                unsafe { skip_flag = true };
                // make sure current kernel is completed
                unsafe { common::flush_channel(stats.pchannel_dev as *mut libc::c_void) };

                unsafe { skip_flag = false };

                while unsafe { recv_thread_receiving } {
                    // why do we need this
                    unsafe {
                        libc::sched_yield();
                    }
                }

                // if !stop_report {
                //     fclose(resultsFile);
                // }

                if unsafe { active_from_start }
                    && unsafe { dynamic_kernel_limit_end } > 0
                    && unsafe { kernelid } > unsafe { dynamic_kernel_limit_end }
                {
                    unsafe { active_region = false }
                }
            }
        }
        bindings::nvbit_api_cuda_t::API_CUDA_cuProfilerStart => {
            if is_exit && !unsafe { active_from_start } {
                unsafe { active_region = true }
            }
        }
        bindings::nvbit_api_cuda_t::API_CUDA_cuProfilerStop => {
            if is_exit && !unsafe { active_from_start } {
                unsafe { active_region = false }
            }
        }
        _ => {
            // do nothing
        }
    }
}

// static *const u64 total_dynamic_instr_counter = 0;
// static &'static u64 total_dynamic_instr_counter; //  = 0;

// 1 MiB = 2**20
// const CHANNEL_SIZE: i32 = 1 << 20;
const CHANNEL_SIZE: u64 = 1 << 10;

// struct Stats<'a> {
struct Stats {
    // total_dynamic_instr_counter: *mut bindings::CUdeviceptr,
    pchannel_dev: *mut ChannelDev,
    // pchannel_host: Pin<&'a mut ChannelHost>,
    channel_host: Mutex<cxx::UniquePtr<ChannelHost>>,
    ptotal_dynamic_instr_counter: *mut u64,
    preported_dynamic_instr_counter: *mut u64,
    pstop_report: *mut bool,
}

// impl<'a> Stats<'a> {
impl Stats {
    #[inline(never)]
    pub fn init(&self) {
        println!("stats are ready");
    }

    pub fn new() -> Self {
        // let mut devPtr: bindings::CUdeviceptr = std::ptr::null_mut::<u64>() as _;
        // let mut devPtr: *mut u64 = std::ptr::null_mut::<u64>() as _;
        // let total_dynamic_instr_counter: u64 = 0;
        // let pinned: Pin<&mut u64> = Pin::new(&mut total_dynamic_instr_counter);
        // // avoid dropping?
        // std::mem::forget(pinned);
        // // let mut total_dynamic_instr_counter: Box<u64> = Box::new(0);
        // let mut devPtr: *mut u64 = &mut *total_dynamic_instr_counter as _;

        let pchannel_dev = unsafe { new_managed_dev_channel() };

        // let pchannel_dev =
        //     unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };
        // panic!("size of channel dev: {}", dev_channel_size());

        // let pchannel_dev =
        //     unsafe { cuda_malloc_unified::<ChannelDev>(1).expect("cuda malloc unified") };

        // panic!("size of channel dev: {}", std::mem::size_of(pchannel_dev.as_ref().unwrap()));

        // let pchannel_dev = unsafe { &mut channel_dev as *mut ChannelDev };
        let mut channel_host = unsafe { new_host_channel(42, CHANNEL_SIZE as i32, pchannel_dev) };

        // pchannel_dev should be initialized now???

        // channel_host.init(0, CHANNEL_SIZE, pchannel_dev, NULL);
        // channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);

        let ptotal_dynamic_instr_counter =
            unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };

        let preported_dynamic_instr_counter =
            unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };

        let pstop_report = unsafe { cuda_malloc_unified::<bool>(1).expect("cuda malloc unified") };

        unsafe { *ptotal_dynamic_instr_counter = 0 };
        unsafe { *preported_dynamic_instr_counter = 0 };
        unsafe { *pstop_report = false };

        // let mut devPtr: *mut u64 = std::ptr::null_mut::<u64>() as _;
        // todo: run the cuda managed alloc here already

        // let pchannel_host = pchannel_host.pin_mut();
        Self {
            pchannel_dev,
            channel_host: Mutex::new(channel_host),
            ptotal_dynamic_instr_counter,
            preported_dynamic_instr_counter,
            pstop_report,
            // total_dynamic_instr_counter: unsafe { &mut devPtr as _ },
        }
    }
}

// unsafe impl Send for Stats<'_> {}
unsafe impl Send for Stats {}
// unsafe impl Sync for Stats<'_> {}
unsafe impl Sync for Stats {}

lazy_static! {
    static ref stats: Stats = Stats::new();
    // static ref stats: Stats<'static> = Stats::new();
    // static ref total_dynamic_instr_counter: bindings::CUdeviceptr =
        // std::ptr::null_mut::<u64>() as _;
}

pub unsafe fn cuda_malloc_unified<T>(count: usize) -> Result<*mut T> {
    // CudaResult<UnifiedPointer<T>> {
    let size = count.checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
    dbg!(&size);
    if size == 0 {
        panic!("InvalidMemoryAllocation");
    }

    let mut ptr: *mut libc::c_void = std::ptr::null_mut();
    let result = unsafe {
        bindings::cuMemAllocManaged(
            &mut ptr as *mut *mut libc::c_void as *mut u64,
            size,
            bindings::CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL as u32,
        )
    };
    if result != bindings::cudaError_enum::CUDA_SUCCESS {
        panic!("CUDA ERROR: {:?}", result);
    }
    // .to_result()?;
    Ok(ptr as *mut T)
    // Ok(UnifiedPointer::wrap(ptr as *mut T))
}

fn read_channel() {
    println!("read channel thread started");

    println!(
        "creating a buffer of size {} ({} MiB)",
        CHANNEL_SIZE as usize,
        CHANNEL_SIZE as f32 / 1024.0f32.powi(2)
    );
    // let mut recv_buffer = buffer::Buffer::with_size(CHANNEL_SIZE as usize);
    let mut recv_buffer = buffer::alloc_box_buffer(CHANNEL_SIZE as usize);
    println!("{:02X?}", &recv_buffer[0..10]);
    println!(
        "{:02X?}",
        &recv_buffer[(CHANNEL_SIZE as usize - 10)..CHANNEL_SIZE as usize]
    );
    // let recv_buffer_ptr =
    // std::mem::forget(recv_buffer);

    // let recv_buffer: [*mut libc::c_void; CHANNEL_SIZE as usize] =
    //     [std::ptr::null_mut() as *mut libc::c_void; CHANNEL_SIZE as usize];

    let packet_size = std::mem::size_of::<common::inst_trace_t>();
    let mut packet_count = 0;

    while unsafe { recv_thread_started } {
        // println!("receiving");
        // let num_recv_bytes: u32 = 0;
        // while (recv_thread_receiving &&
        let num_recv_bytes = unsafe {
            stats.channel_host.lock().unwrap().pin_mut().recv(
                recv_buffer.as_mut_ptr() as *mut c_void,
                // *recv_buffer.as_mut_ptr() as *mut c_void,
                CHANNEL_SIZE as u32,
            )
        };
        if unsafe { recv_thread_receiving } && num_recv_bytes > 0 {
            // println!("received {} bytes", num_recv_bytes);
            let mut num_processed_bytes: usize = 0;
            while num_processed_bytes < num_recv_bytes as usize {
                // let ma: &mut common::inst_trace_t = unsafe {
                let packet_bytes =
                    &recv_buffer[num_processed_bytes..num_processed_bytes + packet_size];
                // println!("{:02X?}", packet_bytes);

                assert_eq!(
                    std::mem::size_of::<common::inst_trace_t>(),
                    packet_bytes.len()
                );
                let ma: common::inst_trace_t = unsafe {
                    std::ptr::read(packet_bytes.as_ptr() as *const _)
                    // recv_buffer[num_processed_bytes]
                    // &mut recv_buffer[num_processed_bytes as *mut common::inst_trace_t)
                    // recv_buffer[num_processed_bytes] as *mut common::inst_trace_t
                    // recv_buffer[num_processed_bytes] as *mut common::inst_trace_t
                };
                println!("received from channel: {:#?}", &ma);

                // when we get this cta_id_x it means the kernel has completed
                if (ma.cta_id_x == -1) {
                    unsafe { recv_thread_receiving = false };
                    break;
                }

                // inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];
                println!("size of common::inst_trace_t: {}", packet_size);
                num_processed_bytes += packet_size;
                packet_count += 1;
            }
        }
    }
    println!("received {} packets", packet_count);
    // todo: currently our buffer is mem::forget so we must free here
}

// void *recv_thread_fun(void *) {
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
// pub extern "C" fn nvbit_at_ctx_init(ctx: *mut CUctx_st) {
pub extern "C" fn nvbit_at_ctx_init(ctx: CUcontext) {
    // setup stats here
    stats.init();

    // cudaMallocManaged((void**)&(elm->name), sizeof(char) * (strlen("hello") + 1) );

    unsafe { recv_thread_started = true };
    unsafe { recv_thread = Some(std::thread::spawn(read_channel)) };
    println!("nvbit_at_ctx_init");

    // start receiving from the channel

    // pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);

    // std::ptr::NonNull::dangling().as_ptr() as *mut u64;

    // let mut total_dynamic_instr_counter: Box<u64> = Box::new(0);
    // let mut devPtr: *mut u64 = &mut *total_dynamic_instr_counter as _;
    // let mut devPtrPtr: *mut *mut u64 = &mut devPtr as _;
    // dbg!(unsafe { **devPtrPtr });
    // dbg!(unsafe {*devPtr});

    //unsafe {
    //    // bindings::cudaMallocManaged((void**)&(elm->name), sizeof(char) * (strlen("hello") + 1) );
    //    // let devPtr: *mut bindings::CUdeviceptr = &mut stats.total_dynamic_instr_counter as *mut _;
    //    // let devPtr: *mut bindings::CUdeviceptr =
    //    //     unsafe { &mut stats.total_dynamic_instr_counter as *mut _ };
    //    //
    //    // bindings::cuMemAllocManaged(
    //    //     stats.total_dynamic_instr_counter as *mut bindings::CUdeviceptr,
    //    //     std::mem::size_of::<u64>(),
    //    //     0,
    //    // );
    //    *stats.total_dynamic_instr_counter = 0;
    //    dbg!(*stats.total_dynamic_instr_counter);
    //}
    // unsafe { say_hi() };
    // unsafe { say_hi() };
    // dynamic loading does not work either
    // unsafe {
    //     // ::<Symbol<extern "C" fn(i32) -> i32>>
    //     let _lib = Library::new("/home/roman/dev/nvbit-sys/target/debug/build/accelsim-a67c1762e4619dad/out/libinstrumentation.so").unwrap();
    //     // let _foo = lib
    //     //     .get(b"foo")
    //     //     .unwrap();

    //     // println!("{}", foo(1));
    // }
}

#[no_mangle]
#[inline(never)]
// pub extern "C" fn nvbit_at_ctx_term(ctx: *mut CUctx_st) {
pub extern "C" fn nvbit_at_ctx_term(ctx: CUcontext) {
    println!("nvbit_at_ctx_term");
    unsafe {
        if recv_thread_started {
            recv_thread_started = false;
        }
        if let Some(handle) = recv_thread.take() {
            // println!("would wait for receiver thread here");
            handle.join().expect("join receiver thread");
        }
        // std::thread::sleep(std::time::Duration::from_secs(20));
    }
    dbg!(unsafe { *stats.pstop_report });
    dbg!(unsafe { *stats.ptotal_dynamic_instr_counter });
    dbg!(unsafe { *stats.preported_dynamic_instr_counter });
    println!("done");
}
