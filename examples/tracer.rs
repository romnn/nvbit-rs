#![allow(warnings)]
use nvbit_sys::*;

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
    println!("it works");
}

// todo: use static vars for all those globals 
/* std::unordered_set<CUfunction> already_instrumented; */

fn instrument_function_if_needed(ctx: CUcontext , func: CUfunction) {
    // std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
// add kernel itself to the related function vector */
/*   related_functions.push_back(func); */
// iterate on function */
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

// this must be a c function so that nvbit will call us
// also, the arguments are not quite right yet
#[no_mangle]
pub extern "C" fn nvbit_at_cuda_event(
    ctx: CUcontext,
    is_exit: i32,
    /*nvbit_api_cuda_t cbid,*/
    /* const char *name, void *params, */
    pStatus: CUresult,
) {
}
