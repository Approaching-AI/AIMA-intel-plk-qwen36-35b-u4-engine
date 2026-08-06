#include "intel_qwen36/resident_harness.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: iq36-load-bundle <model_path> <oracle_bundle_path>\n";
    return 2;
  }

  try {
    iq36::ResidentHarness harness;
    harness.load(argv[1], argv[2]);
    if (!harness.loaded()) {
      std::cerr << "resident harness did not enter loaded state\n";
      return 3;
    }
    const auto& stats = harness.oracle_bundle_stats();
    if (stats.token_topk_rows == 0 ||
        stats.teacher_forced_distribution_rows == 0 ||
        stats.boundary_input_rows == 0 ||
        stats.boundary_output_rows == 0) {
      std::cerr << "resident harness loaded empty oracle references\n";
      return 4;
    }
    const auto result = harness.run_boundary("router_topk");
    if (result.boundary_id != "router_topk") {
      std::cerr << "resident harness boundary smoke returned wrong id\n";
      return 5;
    }
    std::cout << "iq36-load-bundle ok"
              << " token_topk_rows=" << stats.token_topk_rows
              << " teacher_forced_distribution_rows="
              << stats.teacher_forced_distribution_rows
              << " boundary_input_rows=" << stats.boundary_input_rows
              << " boundary_output_rows=" << stats.boundary_output_rows
              << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-load-bundle failed: " << exc.what() << "\n";
    return 1;
  }
}
