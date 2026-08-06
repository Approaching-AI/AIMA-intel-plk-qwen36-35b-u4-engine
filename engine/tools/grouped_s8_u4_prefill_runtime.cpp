#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <iostream>

int main(int argc, char** argv) {
  return iq36::RunGroupedS8U4PrefillCommandLine(
      argc, argv, std::cout, std::cerr);
}
