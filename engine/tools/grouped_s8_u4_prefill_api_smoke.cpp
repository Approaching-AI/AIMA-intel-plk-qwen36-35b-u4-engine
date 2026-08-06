#include "intel_qwen36/grouped_s8_u4_prefill_runtime.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument("usage: api-smoke TOPK TOPK_STRIDE");
    }
    iq36::GroupedS8U4PrefillConfig config;
    config.topk = argv[1];
    config.topk_stride = std::stoull(argv[2]);
    config.schedule_probe_only = true;
    return iq36::RunGroupedS8U4Prefill(
        config, std::cout, std::cerr);
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 4;
  }
}
