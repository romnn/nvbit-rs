#include "nvbit-sys/nvbit/nvbit_bridge.h"

std::unique_ptr<std::vector<CUfunctionShim>> rust_nvbit_get_related_functions(
    CUcontext ctx, CUfunction func
) {
  std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
  auto vec = std::unique_ptr<std::vector<CUfunctionShim>>(new std::vector<CUfunctionShim>());
  for (auto & element : related) {
    vec->push_back(CUfunctionShim { element });
  }

  return vec;
}

std::unique_ptr<std::vector<InstrShim>> rust_nvbit_get_instrs(
    CUcontext ctx, CUfunction func
) {
  std::vector<Instr*> instructions = nvbit_get_instrs(ctx, func);
  auto vec = std::unique_ptr<std::vector<InstrShim>>(new std::vector<InstrShim>());
  for (auto & instr : instructions ) {
    vec->push_back(InstrShim { instr });
  }
  return vec;
}
