#include "intel_qwen36/native_carrier_loop.hpp"
#include "intel_qwen36/resident_harness.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace {

template <typename Steps>
void ValidateSteps(const Steps& steps, const char* label) {
  if (steps.empty()) {
    throw std::invalid_argument(std::string(label) + " steps are empty");
  }
  std::array<bool, 40> seen{};
  for (const auto& step : steps) {
    if (step.layer_index < 0 || step.layer_index >= 40) {
      throw std::invalid_argument(
          std::string(label) + " layer index must be in [0, 40)");
    }
    if (seen[static_cast<std::size_t>(step.layer_index)]) {
      throw std::invalid_argument(
          std::string(label) + " layer index is duplicated");
    }
    seen[static_cast<std::size_t>(step.layer_index)] = true;
  }
}

}  // namespace

namespace iq36 {

NativeCarrierLoop::NativeCarrierLoop(NativeCarrierLayerRuntime& runtime)
    : runtime_(runtime) {}

NativeGroupedPrefillSequenceResult NativeCarrierLoop::RunGroupedPrefill(
    const std::vector<NativeGroupedPrefillStep>& steps) {
  ValidateSteps(steps, "native grouped prefill loop");
  NativeGroupedPrefillSequenceResult result;
  result.layer_indices.reserve(steps.size());
  result.runs.reserve(steps.size());
  for (const auto& step : steps) {
    result.layer_indices.push_back(step.layer_index);
    result.runs.push_back(
        runtime_.RunGroupedPrefillLayer(step.layer_index, step.input));
  }
  return result;
}

NativeQ6CarrierSequenceResult NativeCarrierLoop::RunQ6(
    const std::vector<NativeQ6CarrierStep>& steps) {
  ValidateSteps(steps, "native Q6 loop");
  NativeQ6CarrierSequenceResult result;
  result.layer_indices.reserve(steps.size());
  result.runs.reserve(steps.size());
  for (const auto& step : steps) {
    result.layer_indices.push_back(step.layer_index);
    result.runs.push_back(
        runtime_.RunQ6Layer(step.layer_index, step.input, step.repeat));
  }
  return result;
}

bool loop_shape_is_batch_one_only() {
  return model_contract().workstream == "intel-qwen36-35b-a3b-gguf-q4km";
}

}  // namespace iq36
