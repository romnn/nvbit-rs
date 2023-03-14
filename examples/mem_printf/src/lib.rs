#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_safety_doc)]

use lazy_static::lazy_static;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

#[derive(Debug, Default, Clone)]
struct Args {
    pub opcode_id: std::ffi::c_int,
    pub mref_idx: u64,
}

impl Args {
    pub fn instrument(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        // predicate value
        instr.add_call_arg_guard_pred_val();
        // opcode id
        instr.add_call_arg_const_val32(self.opcode_id.try_into().unwrap_or_default());
        // memory reference 64 bit address
        instr.add_call_arg_mref_addr64(self.mref_idx.try_into().unwrap_or_default());
        // instr.add_call_arg_mref_addr64(self.mref_idx.try_into().unwrap_or_default());
    }
}

extern "C" {
    pub fn noop();
}

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    opcode_to_id_map: RwLock<HashMap<String, usize>>,
    id_to_opcode_map: RwLock<HashMap<usize, String>>,
    start: Instant,
    instr_begin_interval: usize,
    instr_end_interval: usize,
}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        Arc::new(Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            opcode_to_id_map: RwLock::new(HashMap::new()),
            id_to_opcode_map: RwLock::new(HashMap::new()),
            start: Instant::now(),
            instr_begin_interval: 0,
            instr_end_interval: usize::MAX,
        })
    }
}

type Contexts = RwLock<HashMap<nvbit_rs::ContextHandle<'static>, Arc<Instrumentor<'static>>>>;

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
        use nvbit_rs::EventParams;
        let params = EventParams::new(cbid, params);
        if is_exit {
            return;
        }
        if let Some(mut func) = match params {
            Some(EventParams::Launch { func } | EventParams::KernelLaunch { func, .. }) => {
                Some(func)
            }
            _ => None,
        } {
            self.instrument_function_if_needed(&mut func);
            // enable instrumented code to run
            func.enable_instrumented(&mut self.ctx.lock().unwrap(), true, true);
        }
    }

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

        println!("OPCODE {opcode} MAPS TO ID {opcode_id}");
        let mut mref_idx = 0;

        // iterate on the operands
        for operand in instr.operands().collect::<Vec<_>>() {
            if let nvbit_rs::OperandKind::MemRef { .. } = operand.kind() {
                instr.insert_call("instrument_inst", nvbit_rs::InsertionPoint::Before);
                let inst_args = Args {
                    opcode_id: opcode_id.try_into().unwrap(),
                    mref_idx,
                };
                inst_args.instrument(instr);
                mref_idx += 1;
            }
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

    let lock = CONTEXTS.read().unwrap();
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };
    instrumentor.at_cuda_event(is_exit, cbid, &event_name, params, pstatus);
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

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_term");
    let lock = CONTEXTS.read().unwrap();
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };

    unsafe { noop() };
    println!(
        "done after {:?}",
        Instant::now().duration_since(instrumentor.start)
    );
}
