#![allow(warnings)]

use anyhow::Result;
use lazy_static::lazy_static;
use libc;
use libloading::{Library, Symbol};
use nvbit_sys::bindings::{CUcontext, CUfunction};
use nvbit_sys::*;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

// todo: set up the channel
// /* Channel used to communicate from GPU to CPU receiving thread */
// #define CHANNEL_SIZE (1l << 20)
// static __managed__ ChannelDev channel_dev;
// static ChannelHost channel_host;

// global control variables for this tool
// uint32_t instr_begin_interval = 0;
// uint32_t instr_end_interval = UINT32_MAX;
// int verbose = 0;
// int enable_compress = 1;
// int print_core_id = 0;
// int exclude_pred_off = 1;
// int active_from_start = 1;
// /* used to select region of interest when active from start is 0 */
// bool active_region = true;

// /* Should we terminate the program once we are done tracing? */
// int terminate_after_limit_number_of_kernels_reached = 0;
// int user_defined_folders = 0;

// the stats we want to output in the end
// /* opcode to id map and reverse map  */
// std::map<std::string, int> opcode_to_id_map;
// std::map<int, std::string> id_to_opcode_map;

// std::string cwd = getcwd(NULL,0);
// std::string traces_location = cwd + "/traces/";
// std::string kernelslist_location = cwd + "/traces/kernelslist";
// std::string stats_location = cwd + "/traces/stats.csv";
// todo: how can we get the inject funcs into our sys crate??

// this must be a c function so that nvbit will call us
#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    // init();
    // println!("{:?}", rust_nvbit_get_related_functions());
    // println!("it works");
}

// todo: use static vars for all those globals
/* std::unordered_set<CUfunction> already_instrumented; */

// todo: make this sync
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
    // static ref ALREADY_INSTRUMENTED: Mutex<HashSet<TestCUfunction>> = Mutex::new(HashSet::new());
    static ref ALREADY_INSTRUMENTED: Mutex<HashSet<CUfunctionKey>> = Mutex::new(HashSet::new());
}

#[derive(Debug, Default, Clone)]
struct InstrumentInstArgs {
    opcode_id: libc::c_int,
    vpc: u32,
    is_mem: bool,
    addr: u64,
    width: i32,
    desReg: i32,
    srcReg1: i32,
    srcReg2: i32,
    srcReg3: i32,
    srcReg4: i32,
    srcReg5: i32,
    srcNum: i32,
    pchannel_dev: u64,
    ptotal_dynamic_instr_counter: u64,
    preported_dynamic_instr_counter: u64,
    pstop_report: u64,
}

impl InstrumentInstArgs {
    pub unsafe fn instrument(&self, instr: *const Instr) {
        bindings::nvbit_add_call_arg_guard_pred_val(instr, false);
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.opcode_id.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(instr, self.vpc, false);
        bindings::nvbit_add_call_arg_const_val32(instr, self.is_mem as u32, false);
        bindings::nvbit_add_call_arg_mref_addr64(
            instr,
            self.addr.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.width.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.desReg.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg1.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg2.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg3.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg4.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcReg5.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val32(
            instr,
            self.srcNum.try_into().unwrap_or_default(),
            false,
        );
        bindings::nvbit_add_call_arg_const_val64(instr, self.pchannel_dev, false);
        bindings::nvbit_add_call_arg_const_val64(instr, self.ptotal_dynamic_instr_counter, false);
        bindings::nvbit_add_call_arg_const_val64(
            instr,
            self.preported_dynamic_instr_counter,
            false,
        );
        bindings::nvbit_add_call_arg_const_val64(instr, self.pstop_report, false);
    }
}

// fn instrument_function_if_needed(ctx: CUcontext , func: CUfunction) {
// fn instrument_function_if_needed(ctx: *mut bindings::CUctx_st, func: *mut bindings::CUfunc_st) {
fn instrument_function_if_needed(ctx: CUcontext, func: CUfunction) {
    // let testctx = TestCUcontext { inner: ctx };
    // let testfunc = TestCUfunction { inner: func };
    let related_functions = unsafe { rust_nvbit_get_related_functions(ctx, func) };
    // let related_functions = nvbit_get_related_functions(ctx, func);
    // std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    // add kernel itself to the related function vector */
    // let mut already_instrumented = std::collections::HashSet::new();
    // std::unordered_set<CUfunction> already_instrumented;

    // todo: perform this in the safe wrapper
    let mut related_functions: Vec<CUfunctionShim> =
        related_functions.into_iter().copied().collect();

    related_functions.push(unsafe { CUfunctionShim::wrap(func) });
    // println!("number of related functions: {:?}", related_functions.len());

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
        for instr in instrs.iter() {
            // continue;
            // iterate on all the static instructions in the function
            // if count < instr_begin_interval || count >= instr_end_interval {
            //     count += 1;
            //     continue;
            // }

            // unsafe { instr.as_mut_ref().printDecoded() };

            // std::map<std::string, int> opcode_to_id_map;
            let mut opcode_to_id_map: HashMap<String, usize> = HashMap::new();

            // if opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end() {

            let opcode: String = {
                let op = unsafe { instr.as_mut_ref().getOpcode() };
                unsafe { ffi::CStr::from_ptr(op).to_string_lossy().to_string() }
            };

            if !opcode_to_id_map.contains_key(&opcode) {
                let opcode_id = opcode_to_id_map.len();
                opcode_to_id_map.insert(opcode.clone(), opcode_id);
                // id_to_opcode_map[opcode_id] = instr->getOpcode();
            }

            // int opcode_id = opcode_to_id_map[instr->getOpcode()];
            let opcode_id = opcode_to_id_map[&opcode];

            // insert call to the instrumentation function with its arguments */
            // *const ::std::os::raw::c_char
            unsafe {
                bindings::nvbit_insert_call(
                    unsafe { instr.as_ptr() },
                    ffi::CString::new("instrument_inst").unwrap().as_ptr(),
                    bindings::ipoint_t::IPOINT_BEFORE,
                );
            }
            let mut inst_args = InstrumentInstArgs::default();

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

            const MAX_SRC: usize = 5;
            // check all operands. For now, we ignore constant, TEX, predicates and
            // unified registers. We only report vector regisers
            let mut src_oprd: [libc::c_int; MAX_SRC] = [-1; MAX_SRC];
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
            // dbg!(&num_operands);
            // dbg!(&unsafe { instr.as_mut_ref().getNumOperands() });
            for i in 1..(MAX_SRC.min(num_operands as usize)) {
                // dbg!(&i);
                // dbg!(i as i32);
                assert!(i < MAX_SRC);
                assert!(i < num_operands as usize);
                let op = unsafe { *instr.as_mut_ref().getOperand(i as i32) };
                match op.type_ {
                    bindings::InstrType_OperandType::MREF => {
                        // mem is found
                        assert!(srcNum < MAX_SRC);
                        src_oprd[srcNum] = unsafe { op.u.mref.ra_num };
                        srcNum += 1;
                        // TODO: handle LDGSTS with two mem refs
                        assert!(mem_oper_idx == -1); // ensure one memory operand per inst
                        mem_oper_idx += 1;
                    }
                    bindings::InstrType_OperandType::REG => {
                        // reg is found
                        assert!(srcNum < MAX_SRC);
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

            inst_args.desReg = dst_oprd;
            // reg info
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

            inst_args.srcNum = srcNum as i32;
            // unsafe {
            //     bindings::nvbit_add_call_arg_const_val32(
            //         unsafe { instr.as_ptr() },
            //         srcNum as u32,
            //         false,
            //     );
            // }

            // add pointer to channel_dev and other counters

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
    println!("name: {:?} (is_exit = {})", event_name, is_exit);

    // if (skip_flag)
    // return;

    // if (first_call == true) {
    //     first_call = false;

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

    // if (active_from_start && !dynamic_kernel_limit_start || dynamic_kernel_limit_start == 1)
    //   active_region = true;
    // else {
    //   if (active_from_start)
    //     active_region = false;
    // }

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
                // if active_from_start
                //     && dynamic_kernel_limit_start
                //     && kernelid == dynamic_kernel_limit_start
                // {
                //     active_region = true;
                // }

                // if terminate_after_limit_number_of_kernels_reached
                //     && dynamic_kernel_limit_end != 0
                //     && kernelid > dynamic_kernel_limit_end
                // {
                //     // exit(0);
                //     panic!("i decided to terminate");
                // }

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
                unsafe {
                    bindings::nvbit_enable_instrumented(ctx, p.f, true, true);
                    // bindings::nvbit_enable_instrumented(ctx.as_mut_ptr(), p.f, true, true);
                }

                // if (active_region) {
                //   nvbit_enable_instrumented(ctx, p->f, true);
                //   stop_report = false;
                // } else {
                //   nvbit_enable_instrumented(ctx, p->f, false);
                //   stop_report = true;
                // }

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
            }
        }
        _ => {
            // do nothing
        }
    }
}

// static *const u64 total_dynamic_instr_counter = 0;
// static &'static u64 total_dynamic_instr_counter; //  = 0;

const CHANNEL_SIZE: u64 = 1u64 << 20;

struct Stats {
    // total_dynamic_instr_counter: *mut bindings::CUdeviceptr,
    pchannel_dev: *mut ChannelDev,
    pchannel_host: *mut ChannelHost,
    ptotal_dynamic_instr_counter: *mut u64,
    preported_dynamic_instr_counter: *mut u64,
    pstop_report: *mut bool,
}

// static __managed__ ChannelDev channel_dev;
// static ChannelHost channel_host;

impl Stats {
    pub fn new() -> Self {
        // let mut devPtr: bindings::CUdeviceptr = std::ptr::null_mut::<u64>() as _;
        // let mut devPtr: *mut u64 = std::ptr::null_mut::<u64>() as _;
        // let total_dynamic_instr_counter: u64 = 0;
        // let pinned: Pin<&mut u64> = Pin::new(&mut total_dynamic_instr_counter);
        // // avoid dropping?
        // std::mem::forget(pinned);
        // // let mut total_dynamic_instr_counter: Box<u64> = Box::new(0);
        // let mut devPtr: *mut u64 = &mut *total_dynamic_instr_counter as _;

        // let pchannel_dev =
        //     unsafe { cuda_malloc_unified::<u64>(1).expect("cuda malloc unified") };
        let pchannel_dev =
            unsafe { cuda_malloc_unified::<ChannelDev>(1).expect("cuda malloc unified") };

        let pchannel_host = 
            unsafe { cuda_malloc_unified::<ChannelHost>(1).expect("cuda malloc unified") };

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
        Self {
            pchannel_dev,
            pchannel_host,
            ptotal_dynamic_instr_counter,
            preported_dynamic_instr_counter,
            pstop_report,
            // total_dynamic_instr_counter: unsafe { &mut devPtr as _ },
        }
    }
}

unsafe impl Send for Stats {}
unsafe impl Sync for Stats {}

lazy_static! {
    static ref stats: Stats = Stats::new();
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

#[no_mangle]
#[inline(never)]
// pub extern "C" fn nvbit_at_ctx_init(ctx: *mut CUctx_st) {
pub extern "C" fn nvbit_at_ctx_init(ctx: CUcontext) {
    // ptr: DevicePointer<T>,
    // if mem::size_of::<T>() == 0 {
    //
    // cudaMallocManaged((void**)&(elm->name), sizeof(char) * (strlen("hello") + 1) );
    println!("nvbit_at_ctx_init");

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
    println!("total_dynamic_instr_counter:");
    // dbg!(stats.total_dynamic_instr_counter);
    dbg!(unsafe { *stats.pstop_report });
    dbg!(unsafe { *stats.ptotal_dynamic_instr_counter });
    dbg!(unsafe { *stats.preported_dynamic_instr_counter });
    println!("nvbit_at_ctx_term");
}

// #[no_mangle]
// #[inline(never)]

// extern "C" {
//     pub fn instrument_inst(
//         pred: libc::c_int,
//         opcode_id: libc::c_int,
//         vpc: u32,
//         is_mem: bool,
//         addr: u64,
//         width: i32,
//         desReg: i32,
//         srcReg1: i32,
//         srcReg2: i32,
//         srcReg3: i32,
//         srcReg4: i32,
//         srcReg5: i32,
//         srcNum: i32,
//         pchannel_dev: u64,
//         ptotal_dynamic_instr_counter: u64,
//         preported_dynamic_instr_counter: u64,
//         pstop_report: u64,
//     );
// }
// {
// panic!("called");
// }

// pub fn say_hi() {
//     // println!("about to die");
//     // unsafe { instrument_inst(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) }
//     // println!("survive");
// }

// #[no_mangle]
// pub unsafe extern "C" fn new_instrument_inst(
//     pred: libc::c_int,
//     opcode_id: libc::c_int,
//     vpc: u32,
//     is_mem: bool,
//     addr: u64,
//     width: i32,
//     desReg: i32,
//     srcReg1: i32,
//     srcReg2: i32,
//     srcReg3: i32,
//     srcReg4: i32,
//     srcReg5: i32,
//     srcNum: i32,
//     pchannel_dev: u64,
//     ptotal_dynamic_instr_counter: u64,
//     preported_dynamic_instr_counter: u64,
//     pstop_report: u64,
// ) {
//     //     // external_symbols::instrument_inst(
//     //     instrument_inst(
//     //         pred,
//     //         opcode_id,
//     //         vpc,
//     //         is_mem,
//     //         addr,
//     //         width,
//     //         desReg,
//     //         srcReg1,
//     //         srcReg2,
//     //         srcReg3,
//     //         srcReg4,
//     //         srcReg5,
//     //         srcNum,
//     //         pchannel_dev,
//     //         ptotal_dynamic_instr_counter,
//     //         preported_dynamic_instr_counter,
//     //         pstop_report,
//     //     );
// }

// #[no_mangle]
// #[inline(never)]
// extern "C" pub fn instrument_inst(
//     pred: libc::c_int,
//     opcode_id: libc::c_int,
//     vpc: u32,
//     is_mem: bool,
//     addr: u64,
//     width: i32,
//     desReg: i32,
//     srcReg1: i32,
//     srcReg2: i32,
//     srcReg3: i32,
//     srcReg4: i32,
//     srcReg5: i32,
//     srcNum: i32,
//     pchannel_dev: u64,
//     ptotal_dynamic_instr_counter: u64,
//     preported_dynamic_instr_counter: u64,
//     pstop_report: u64,
// );

// // // mod external_symbols {
// extern "C" {
//     // #[no_mangle]
//     // pub fn say_hi();

//     #[no_mangle]
//     #[inline(never)]
//     pub fn instrument_inst(
//         pred: libc::c_int,
//         opcode_id: libc::c_int,
//         vpc: u32,
//         is_mem: bool,
//         addr: u64,
//         width: i32,
//         desReg: i32,
//         srcReg1: i32,
//         srcReg2: i32,
//         srcReg3: i32,
//         srcReg4: i32,
//         srcReg5: i32,
//         srcNum: i32,
//         pchannel_dev: u64,
//         ptotal_dynamic_instr_counter: u64,
//         preported_dynamic_instr_counter: u64,
//         pstop_report: u64,
//     );
// }
// // }
