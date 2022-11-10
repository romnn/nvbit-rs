#![allow(warnings)]
use lazy_static::lazy_static;
use libc;
use nvbit_sys::*;
use std::ffi;
use std::sync::Mutex;

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
pub extern "C" fn nvbit_at_init() {
    init();
    println!("{:?}", rust_nvbit_get_related_functions());
    println!("it works");
}

// todo: use static vars for all those globals
/* std::unordered_set<CUfunction> already_instrumented; */

use std::collections::HashSet;

lazy_static! {
    static ref ALREADY_INSTRUMENTED: Mutex<HashSet<TestCUfunction>> = Mutex::new(HashSet::new());
}

// fn instrument_function_if_needed(ctx: CUcontext , func: CUfunction) {
fn instrument_function_if_needed(ctx: *mut bindings::CUctx_st, func: *mut bindings::CUfunc_st) {
    // let testctx = TestCUcontext { inner: ctx };
    // let testfunc = TestCUfunction { inner: func };
    let mut related_functions = unsafe { rust_new_nvbit_get_related_functions(ctx, func) };
    // let related_functions = nvbit_get_related_functions(ctx, func);
    // std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
    // add kernel itself to the related function vector */
    // let mut already_instrumented = std::collections::HashSet::new();
    // std::unordered_set<CUfunction> already_instrumented;

    related_functions.push(TestCUfunction { inner: func });
    println!("number of related functions: {:?}", related_functions.len());

    for f in &related_functions {
        // "recording" function was instrumented,
        // if set insertion failed we have already encountered this function
        let func_name: *const libc::c_char =
            unsafe { bindings::nvbit_get_func_name(ctx, f.inner, false) };

        let func_name = unsafe { ffi::CStr::from_ptr(func_name).to_string_lossy() };

        let func_addr: u64 = unsafe { bindings::nvbit_get_func_addr(f.inner) };

        let mut instrumented_lock = ALREADY_INSTRUMENTED.lock().unwrap();
        if !instrumented_lock.insert(*f) {
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
    }
    /*   for (auto f : related_functions) { */
    /*     // "recording" function was instrumented, */
    /*     // if set insertion failed we have already encountered this function */
    /*     if (!already_instrumented.insert(f).second) { */
    /*       continue; */
    /*     } */

    /*     const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f); */
    /*     if (verbose) { */
    /*       printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f), true); */
    /*     } */

    /*     uint32_t cnt = 0; */
    /*     // iterate on all the static instructions in the function */
    /*     for (auto instr : instrs) { */
    /*       if (cnt < instr_begin_interval || cnt >= instr_end_interval) { */
    /*         cnt++; */
    /*         continue; */
    /*       } */

    /*       if (verbose) { */
    /*         instr->printDecoded(); */
    /*       } */

    /*       if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) { */
    /*         int opcode_id = opcode_to_id_map.size(); */
    /*         opcode_to_id_map[instr->getOpcode()] = opcode_id; */
    /*         id_to_opcode_map[opcode_id] = instr->getOpcode(); */
    /*       } */

    /*       int opcode_id = opcode_to_id_map[instr->getOpcode()]; */

    /*       // insert call to the instrumentation function with its arguments */
    /*       nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE); */

    /*       // pass predicate value */
    /*       nvbit_add_call_arg_guard_pred_val(instr); */

    /*       // send opcode and pc */
    /*       nvbit_add_call_arg_const_val32(instr, opcode_id); */
    /*       nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset()); */

    /*       // check all operands. For now, we ignore constant, TEX, predicates and */
    /*       // unified registers. We only report vector regisers */
    /*       int src_oprd[MAX_SRC]; */
    /*       int srcNum = 0; */
    /*       int dst_oprd = -1; */
    /*       int mem_oper_idx = -1; */

    /*       // find dst reg and handle the special case if the oprd[0] is mem */
    /*       // (e.g. store and RED) */
    /*       if (instr->getNumOperands() > 0 && */
    /*           instr->getOperand(0)->type == InstrType::OperandType::REG) */
    /*         dst_oprd = instr->getOperand(0)->u.reg.num; */
    /*       else if (instr->getNumOperands() > 0 && */
    /*                instr->getOperand(0)->type == InstrType::OperandType::MREF) { */
    /*         src_oprd[0] = instr->getOperand(0)->u.mref.ra_num; */
    /*         mem_oper_idx = 0; */
    /*         srcNum++; */
    /*       } */

    /*       // find src regs and mem */
    /*       for (int i = 1; i < MAX_SRC; i++) { */
    /*         if (i < instr->getNumOperands()) { */
    /*           const InstrType::operand_t *op = instr->getOperand(i); */
    /*           if (op->type == InstrType::OperandType::MREF) { */
    /*             // mem is found */
    /*             assert(srcNum < MAX_SRC); */
    /*             src_oprd[srcNum] = instr->getOperand(i)->u.mref.ra_num; */
    /*             srcNum++; */
    /*             // TO DO: handle LDGSTS with two mem refs */
    /*             assert(mem_oper_idx == -1); // ensure one memory operand per inst */
    /*             mem_oper_idx++; */
    /*           } else if (op->type == InstrType::OperandType::REG) { */
    /*             // reg is found */
    /*             assert(srcNum < MAX_SRC); */
    /*             src_oprd[srcNum] = instr->getOperand(i)->u.reg.num; */
    /*             srcNum++; */
    /*           } */
    /*           // skip anything else (constant and predicates) */
    /*         } */
    /*       } */

    /*       // mem addresses info */
    /*       if (mem_oper_idx >= 0) { */
    /*         nvbit_add_call_arg_const_val32(instr, 1); */
    /*         nvbit_add_call_arg_mref_addr64(instr, 0); */
    /*         nvbit_add_call_arg_const_val32(instr, (int)instr->getSize()); */
    /*       } else { */
    /*         nvbit_add_call_arg_const_val32(instr, 0); */
    /*         nvbit_add_call_arg_const_val64(instr, -1); */
    /*         nvbit_add_call_arg_const_val32(instr, -1); */
    /*       } */

    /*       // reg info */
    /*       nvbit_add_call_arg_const_val32(instr, dst_oprd); */
    /*       for (int i = 0; i < srcNum; i++) { */
    /*         nvbit_add_call_arg_const_val32(instr, src_oprd[i]); */
    /*       } */
    /*       for (int i = srcNum; i < MAX_SRC; i++) { */
    /*         nvbit_add_call_arg_const_val32(instr, -1); */
    /*       } */
    /*       nvbit_add_call_arg_const_val32(instr, srcNum); */

    /*       // add pointer to channel_dev and other counters */
    /*       nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev); */
    /*       nvbit_add_call_arg_const_val64(instr, */
    /*                                      (uint64_t)&total_dynamic_instr_counter); */
    /*       nvbit_add_call_arg_const_val64(instr, */
    /*                                      (uint64_t)&reported_dynamic_instr_counter); */
    /*       nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report); */
    /*       cnt++; */
    /*     } */
    /*   } */
}

#[no_mangle]
pub extern "C" fn nvbit_at_cuda_event(
    ctx: *mut bindings::CUctx_st,
    is_exit: libc::c_int,
    cbid: bindings::nvbit_api_cuda_t,
    event_name: *const libc::c_char,
    params: *mut ffi::c_void,
    pStatus: *mut bindings::CUresult,
) {
    let is_exit = is_exit != 0;
    println!("nvbit_at_cuda_event");
    println!("is exit: {:?}", is_exit);
    let event_name = unsafe { ffi::CStr::from_ptr(event_name).to_string_lossy() };
    println!("name: {:?}", event_name);

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
                dbg!(p);
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
                dbg!(&nregs);

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
                dbg!(&shmem_static_nbytes);

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
                dbg!(&binary_version);

                // int binary_version;
                // CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                //                                  CU_FUNC_ATTRIBUTE_BINARY_VERSION, p->f));
                instrument_function_if_needed(ctx, p.f);

                unsafe {
                    bindings::nvbit_enable_instrumented(ctx, p.f, true, true);
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

#[no_mangle]
pub extern "C" fn nvbit_at_ctx_init(ctx: *mut CUctx_st) {
    println!("nvbit_at_ctx_init");
}

#[no_mangle]
pub extern "C" fn nvbit_at_ctx_term(ctx: *mut CUctx_st) {
    println!("nvbit_at_ctx_term");
}
