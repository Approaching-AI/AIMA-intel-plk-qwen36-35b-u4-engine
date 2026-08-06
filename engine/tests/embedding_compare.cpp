#include "compare_harness.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using iq36::harness::ValueStats;

constexpr const char* kTensorName = "token_embd.weight";
constexpr double kMismatchThreshold = 1e-6;
constexpr double kMaxAbsDiffThreshold = 1e-6;
constexpr double kRmseThreshold = 1e-7;
constexpr double kMinCosine = 0.999999;

}  // namespace

int main(int argc, char** argv) {
  using namespace iq36::harness;
  try {
    require(argc == 4,
            "usage: iq36-embedding-compare <model.gguf> <token_id> <oracle-f32-payload>");
    const std::string model_path = argv[1];
    const auto token_id = parse_u64(argv[2]);
    const std::string oracle_payload_path = argv[3];

    const auto index = iq36::parse_gguf_model_index(model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto* tensor = iq36::find_tensor(index, kTensorName);
    require(tensor != nullptr, "token embedding tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 12 &&
        tensor->dims == std::vector<std::uint64_t>{2048, 248320};

    const auto native = iq36::decode_tensor_row(model_path, index, kTensorName, token_id);
    const auto oracle = iq36::read_f32_vector_file(oracle_payload_path);
    const auto native_stats = stats_from_values(native);
    const auto oracle_stats = stats_from_values(oracle);
    const auto compare = iq36::compare_vectors(native, oracle, kMismatchThreshold);

    const bool passed =
        load_map.ready &&
        tensor_shape_ok &&
        native_stats.count == 2048 &&
        oracle_stats.count == 2048 &&
        native_stats.finite &&
        oracle_stats.finite &&
        native_stats.nonzero &&
        oracle_stats.nonzero &&
        compare.same_size &&
        compare.finite &&
        compare.mismatch_count == 0 &&
        compare.max_abs_diff <= kMaxAbsDiffThreshold &&
        compare.rmse <= kRmseThreshold &&
        compare.cosine >= kMinCosine;

    std::cout << std::setprecision(17);
    std::cout << "{";
    std::cout << "\"comparison\":";
    write_compare_stats(compare);
    std::cout << ",";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"model_path\":\"" << json_escape(model_path) << "\",";
    std::cout << "\"native_vector\":";
    write_value_stats(native_stats);
    std::cout << ",";
    std::cout << "\"oracle_payload_path\":\"" << json_escape(oracle_payload_path) << "\",";
    std::cout << "\"oracle_vector\":";
    write_value_stats(oracle_stats);
    std::cout << ",";
    std::cout << "\"passed\":" << (passed ? "true" : "false") << ",";
    std::cout << "\"schema_version\":\"intel-qwen36-engine-embedding-compare-v0\",";
    std::cout << "\"tensor\":{";
    std::cout << "\"absolute_offset\":" << tensor->absolute_offset << ",";
    std::cout << "\"dims\":";
    write_u64_vector(tensor->dims);
    std::cout << ",";
    std::cout << "\"name\":\"" << json_escape(tensor->name) << "\",";
    std::cout << "\"nbytes\":" << tensor->nbytes << ",";
    std::cout << "\"shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"type_name\":\"" << iq36::ggml_type_name(tensor->type) << "\"";
    std::cout << "},";
    std::cout << "\"thresholds\":{";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine << ",";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold;
    std::cout << "},";
    std::cout << "\"token_id\":" << token_id;
    std::cout << "}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "iq36-embedding-compare: " << exc.what() << "\n";
    return 1;
  }
}
