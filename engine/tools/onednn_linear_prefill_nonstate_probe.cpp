#define IQ36_Q4K_COMPONENT_NO_MAIN
#include "onednn_q4k_bucket_component.cpp"

#include "intel_qwen36/gguf_loader.hpp"

#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>

namespace linear_nonstate {

constexpr std::size_t kTokens = 1024;
constexpr double kCosineMinimum = 0.999;
constexpr double kRelativeL2Maximum = 0.002;

struct Comparison {
  std::size_t count = 0;
  double max_abs = 0.0;
  double relative_l2 = 0.0;
  double cosine = 0.0;
  bool finite = false;
  bool passes = false;
};

struct ProjectionSpec {
  std::string label;
  std::string tensor;
  std::string source_capture;
  std::string reference_capture;
  std::size_t input = 0;
  std::size_t output = 0;
};

struct ProjectionResult {
  std::string label;
  std::string implementation;
  std::size_t input = 0;
  std::size_t output = 0;
  double median_us = 0.0;
  std::vector<double> samples_us;
  Comparison comparison;
};

struct Args {
  std::filesystem::path model;
  std::filesystem::path capture;
  int warmup = 5;
  int repeat = 21;
};

Args Parse(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto Value = [&]() -> std::string {
      if (++index >= argc) Fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--model") args.model = Value();
    else if (option == "--capture") args.capture = Value();
    else if (option == "--warmup") args.warmup = std::stoi(Value());
    else if (option == "--repeat") args.repeat = std::stoi(Value());
    else Fail("unknown option: " + option);
  }
  Require(!args.model.empty() && !args.capture.empty(),
          "model and capture are required");
  Require(args.warmup >= 0 && args.repeat >= 3,
          "warmup/repeat arguments are invalid");
  return args;
}

template <typename Value>
std::vector<Value> ReadVector(const std::filesystem::path& path,
                              std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  Require(static_cast<bool>(input), "could not open " + path.string());
  const auto bytes = input.tellg();
  Require(bytes == static_cast<std::streamoff>(
                       expected_count * sizeof(Value)),
          "payload size mismatch for " + path.string());
  input.seekg(0);
  std::vector<Value> values(expected_count);
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  Require(static_cast<bool>(input), "could not read " + path.string());
  return values;
}

template <typename Value>
std::vector<Value> ReadMemory(const dnnl::memory& memory,
                              std::size_t expected_count,
                              const std::string& label) {
  Require(memory.get_desc().get_size() == expected_count * sizeof(Value),
          label + " oneDNN memory size mismatch");
  void* mapped = memory.map_data();
  Require(mapped != nullptr, label + " oneDNN map failed");
  std::vector<Value> values(expected_count);
  std::memcpy(values.data(), mapped, values.size() * sizeof(Value));
  memory.unmap_data(mapped);
  return values;
}

std::filesystem::path CapturePayload(const std::filesystem::path& capture,
                                     const std::string& tensor) {
  const auto payloads = capture / "payloads";
  const std::string prefix = tensor + "__";
  std::vector<std::filesystem::path> matches;
  for (const auto& entry : std::filesystem::directory_iterator(payloads)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().rfind(prefix, 0) == 0) {
      matches.push_back(entry.path());
    }
  }
  Require(matches.size() == 1, "capture payload count mismatch for " + tensor);
  return matches.front();
}

std::vector<std::uint8_t> ReadTensorBytes(
    const std::filesystem::path& model, const iq36::GgufTensorInfo& tensor) {
  std::ifstream input(model, std::ios::binary);
  Require(static_cast<bool>(input), "could not open model payload");
  input.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
  std::vector<std::uint8_t> bytes(tensor.nbytes);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(input), "could not read model tensor payload");
  return bytes;
}

struct QuantizedSource {
  std::vector<std::int8_t> values;
  std::vector<float> scales;
  std::vector<float> sums32_scaled;
};

QuantizedSource QuantizeSource(const std::vector<float>& input,
                               std::size_t width) {
  Require(width % kQ4BlockValues == 0 &&
              input.size() == kTokens * width,
          "Q8 source shape mismatch");
  const std::size_t blocks = width / kQ4BlockValues;
  QuantizedSource result;
  result.values.resize(input.size(), 0);
  result.scales.resize(kTokens * blocks, 0.0f);
  result.sums32_scaled.resize(kTokens * blocks * 8, 0.0f);
  for (std::size_t token = 0; token < kTokens; ++token) {
    for (std::size_t block = 0; block < blocks; ++block) {
      const float* source = input.data() +
          token * width + block * kQ4BlockValues;
      float max_value = 0.0f;
      float absolute_max = 0.0f;
      for (std::size_t index = 0; index < kQ4BlockValues; ++index) {
        const float absolute = std::abs(source[index]);
        if (absolute > absolute_max) {
          absolute_max = absolute;
          max_value = source[index];
        }
      }
      if (absolute_max == 0.0f) continue;
      const float inverse_scale = -127.0f / max_value;
      const float scale = 1.0f / inverse_scale;
      result.scales[token * blocks + block] = scale;
      std::array<int, 8> sums{};
      for (std::size_t index = 0; index < kQ4BlockValues; ++index) {
        const int value = std::min(
            127, NearestInt(inverse_scale * source[index]));
        result.values[token * width + block * kQ4BlockValues + index] =
            static_cast<std::int8_t>(value);
        sums[index / 32] += value;
      }
      for (std::size_t group = 0; group < 8; ++group) {
        result.sums32_scaled[
            (token * blocks + block) * 8 + group] =
            scale * static_cast<float>(sums[group]);
      }
    }
  }
  return result;
}

std::uint8_t Q4CodeLocal(const std::uint8_t* block,
                         std::size_t within_block) {
  const std::size_t segment = within_block / 64;
  const std::size_t offset = within_block % 32;
  const std::uint8_t packed = block[16 + segment * 32 + offset];
  return static_cast<std::uint8_t>(
      (within_block & 32U) == 0 ? packed & 15U : packed >> 4);
}

void SetU4Local(std::vector<std::uint8_t>& packed,
                std::size_t logical_index, std::uint8_t value) {
  std::uint8_t& byte = packed[logical_index >> 1];
  if ((logical_index & 1U) == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xf0U) | value);
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0fU) | (value << 4));
  }
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

Comparison Compare(const std::vector<float>& observed,
                   const std::vector<float>& reference) {
  Require(observed.size() == reference.size(), "comparison size mismatch");
  Comparison result;
  result.count = observed.size();
  long double dot = 0.0L;
  long double observed_square = 0.0L;
  long double reference_square = 0.0L;
  long double error_square = 0.0L;
  result.finite = true;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    const double lhs = observed[index];
    const double rhs = reference[index];
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) result.finite = false;
    const double error = lhs - rhs;
    result.max_abs = std::max(result.max_abs, std::abs(error));
    dot += static_cast<long double>(lhs) * rhs;
    observed_square += static_cast<long double>(lhs) * lhs;
    reference_square += static_cast<long double>(rhs) * rhs;
    error_square += static_cast<long double>(error) * error;
  }
  result.relative_l2 = std::sqrt(static_cast<double>(
      error_square / std::max(reference_square, 1.0e-30L)));
  result.cosine = static_cast<double>(
      dot / std::sqrt(std::max(observed_square * reference_square, 1.0e-30L)));
  result.passes = result.finite && result.cosine >= kCosineMinimum &&
      result.relative_l2 <= kRelativeL2Maximum;
  return result;
}

class Projection {
 public:
  virtual ~Projection() = default;
  virtual void Execute(dnnl::stream& stream) = 0;
  virtual std::vector<float> ReadOutput() const = 0;
  virtual const std::string& Implementation() const = 0;
  virtual const ProjectionSpec& Spec() const = 0;
};

class AffineQ4Matmul final : public Projection {
 public:
  AffineQ4Matmul(const dnnl::engine& engine,
                 const std::filesystem::path& model,
                 const iq36::GgufModelIndex& index,
                 const ProjectionSpec& spec,
                 const std::vector<float>& source)
      : spec_(spec) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    Require(source.size() == kTokens * spec.input,
            spec.label + " source size mismatch");
    Require(spec.input % kQ4BlockValues == 0 && spec.output % 2 == 0,
            spec.label + " shape is not Q4-packable");
    const auto* tensor = iq36::find_tensor(index, spec.tensor);
    Require(tensor != nullptr, spec.tensor + " is missing");
    if (tensor->type != 12U ||
        tensor->dims != std::vector<std::uint64_t>{spec.input, spec.output}) {
      std::ostringstream message;
      message << spec.tensor << " contract mismatch: type=" << tensor->type
              << " dims=";
      for (const auto value : tensor->dims) message << value << ',';
      Fail(message.str());
    }
    const auto raw = ReadTensorBytes(model, *tensor);
    const std::size_t blocks = spec.input / kQ4BlockValues;
    Require(raw.size() == spec.output * blocks * kQ4BlockBytes,
            spec.tensor + " byte size mismatch");

    const auto source_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.input)},
        dt::s8, tag::ab);
    const auto weights_desc = dnnl::memory::desc(
        {static_cast<int>(spec.input), static_cast<int>(spec.output)},
        dt::u4, tag::ba);
    const auto destination_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.output)},
        dt::f32, tag::ab);
    const std::size_t groups = spec.input / 32;
    const auto scales_desc = dnnl::memory::desc(
        {static_cast<int>(groups), static_cast<int>(spec.output)},
        dt::f32, tag::ab);
    const auto source_scales_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.input / 256)},
        dt::f32, tag::ab);
    const auto min_source_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(groups)},
        dt::f32, tag::ab);
    const auto min_weights_desc = dnnl::memory::desc(
        {static_cast<int>(groups), static_cast<int>(spec.output)},
        dt::f32, tag::ba);

    source_ = dnnl::memory(source_desc, engine);
    weights_ = dnnl::memory(weights_desc, engine);
    destination_ = dnnl::memory(destination_desc, engine);
    scales_ = dnnl::memory(scales_desc, engine);
    source_scales_ = dnnl::memory(source_scales_desc, engine);
    min_source_ = dnnl::memory(min_source_desc, engine);
    min_weights_ = dnnl::memory(min_weights_desc, engine);
    min_destination_ = dnnl::memory(destination_desc, engine);

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_SRC, 3, {1, 256}, dt::f32);
    attributes.set_scales(DNNL_ARG_WEIGHTS, 3, {32, 1}, dt::f32);
    const dnnl::matmul::primitive_desc main_descriptor(
        engine, source_desc, weights_desc, destination_desc, attributes);
    const dnnl::matmul::primitive_desc min_descriptor(
        engine, min_source_desc, min_weights_desc, destination_desc);
    implementation_ = std::string(main_descriptor.impl_info_str()) + "+" +
        min_descriptor.impl_info_str();
    main_ = dnnl::matmul(main_descriptor);
    min_ = dnnl::matmul(min_descriptor);
    main_arguments_ = {
        {DNNL_ARG_SRC, source_}, {DNNL_ARG_WEIGHTS, weights_},
        {DNNL_ARG_DST, destination_},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, source_scales_},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_},
    };
    min_arguments_ = {
        {DNNL_ARG_SRC, min_source_}, {DNNL_ARG_WEIGHTS, min_weights_},
        {DNNL_ARG_DST, min_destination_},
    };

    const auto q8 = QuantizeSource(source, spec.input);

    std::vector<std::uint8_t> packed(spec.input * spec.output / 2, 0);
    std::vector<float> scales(groups * spec.output, 0.0f);
    std::vector<float> minimums(spec.output * groups, 0.0f);
    for (std::size_t output = 0; output < spec.output; ++output) {
      for (std::size_t block = 0; block < blocks; ++block) {
        const std::uint8_t* q4 =
            raw.data() + (output * blocks + block) * kQ4BlockBytes;
        const float d = HalfToFloat(LoadU16(q4));
        const float dmin = HalfToFloat(LoadU16(q4 + 2));
        for (std::size_t group = 0; group < 8; ++group) {
          const std::size_t global_group = block * 8 + group;
          scales[global_group * spec.output + output] =
              d * static_cast<float>(GetScale(group, q4 + 4));
          minimums[output * groups + global_group] =
              dmin * static_cast<float>(GetMinimum(group, q4 + 4));
        }
        for (std::size_t within = 0; within < kQ4BlockValues; ++within) {
          const std::size_t k = block * kQ4BlockValues + within;
          SetU4Local(packed, output * spec.input + k,
                     Q4CodeLocal(q4, within));
        }
      }
    }
    WriteMemory(q8.values, source_);
    WriteMemory(q8.scales, source_scales_);
    WriteMemory(packed, weights_);
    WriteMemory(scales, scales_);
    WriteMemory(q8.sums32_scaled, min_source_);
    WriteMemory(minimums, min_weights_);
  }

  void Execute(dnnl::stream& stream) override {
    main_.execute(stream, main_arguments_);
    min_.execute(stream, min_arguments_);
  }

  std::vector<float> ReadOutput() const override {
    auto main = ReadMemory<float>(
        destination_, kTokens * spec_.output, spec_.label + " main");
    const auto minimum = ReadMemory<float>(
        min_destination_, kTokens * spec_.output, spec_.label + " minimum");
    for (std::size_t index = 0; index < main.size(); ++index) {
      main[index] -= minimum[index];
    }
    return main;
  }

  const std::string& Implementation() const override { return implementation_; }
  const ProjectionSpec& Spec() const override { return spec_; }

 private:
  ProjectionSpec spec_;
  dnnl::memory source_;
  dnnl::memory weights_;
  dnnl::memory destination_;
  dnnl::memory scales_;
  dnnl::memory source_scales_;
  dnnl::memory min_source_;
  dnnl::memory min_weights_;
  dnnl::memory min_destination_;
  dnnl::matmul main_;
  dnnl::matmul min_;
  std::unordered_map<int, dnnl::memory> main_arguments_;
  std::unordered_map<int, dnnl::memory> min_arguments_;
  std::string implementation_;
};

int Q6ValueLocal(const std::uint8_t* block, std::size_t index) {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t quadrant = within / 32;
  const std::size_t lane = within % 32;
  const std::uint8_t high = block[128 + half * 32 + lane];
  int low = 0;
  int high_bits = 0;
  if (quadrant == 0) {
    low = block[half * 64 + lane] & 15;
    high_bits = high & 3;
  } else if (quadrant == 1) {
    low = block[half * 64 + 32 + lane] & 15;
    high_bits = (high >> 2) & 3;
  } else if (quadrant == 2) {
    low = block[half * 64 + lane] >> 4;
    high_bits = (high >> 4) & 3;
  } else {
    low = block[half * 64 + 32 + lane] >> 4;
    high_bits = (high >> 6) & 3;
  }
  return (low | (high_bits << 4)) - 32;
}

class S8Per32Q6Matmul final : public Projection {
 public:
  S8Per32Q6Matmul(const dnnl::engine& engine,
                  const std::filesystem::path& model,
                  const iq36::GgufModelIndex& index,
                  const ProjectionSpec& spec,
                  const std::vector<float>& source)
      : spec_(spec) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    constexpr std::size_t kQ6BlockBytes = 210;
    constexpr std::size_t kQ6ScaleGroup = 32;
    Require(source.size() == kTokens * spec.input,
            spec.label + " source size mismatch");
    Require(spec.input % kQ4BlockValues == 0,
            spec.label + " Q6 input is not block aligned");
    const auto* tensor = iq36::find_tensor(index, spec.tensor);
    Require(tensor != nullptr && tensor->type == 14U &&
                tensor->dims == std::vector<std::uint64_t>{
                    spec.input, spec.output},
            spec.tensor + " Q6 contract mismatch");
    const auto raw = ReadTensorBytes(model, *tensor);
    const std::size_t blocks = spec.input / kQ4BlockValues;
    Require(raw.size() == spec.output * blocks * kQ6BlockBytes,
            spec.tensor + " Q6 byte size mismatch");
    const std::size_t groups = spec.input / kQ6ScaleGroup;

    const auto source_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.input)},
        dt::s8, tag::ab);
    const auto weights_desc = dnnl::memory::desc(
        {static_cast<int>(spec.input), static_cast<int>(spec.output)},
        dt::s8, tag::ba);
    const auto destination_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.output)},
        dt::f32, tag::ab);
    const auto scales_desc = dnnl::memory::desc(
        {static_cast<int>(groups), static_cast<int>(spec.output)},
        dt::f32, tag::ab);
    const auto source_scales_desc = dnnl::memory::desc(
        {static_cast<int>(kTokens), static_cast<int>(spec.input / 256)},
        dt::f32, tag::ab);
    source_ = dnnl::memory(source_desc, engine);
    weights_ = dnnl::memory(weights_desc, engine);
    destination_ = dnnl::memory(destination_desc, engine);
    scales_ = dnnl::memory(scales_desc, engine);
    source_scales_ = dnnl::memory(source_scales_desc, engine);

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_SRC, 3, {1, 256}, dt::f32);
    attributes.set_scales(DNNL_ARG_WEIGHTS, 3, {32, 1}, dt::f32);
    const dnnl::matmul::primitive_desc descriptor(
        engine, source_desc, weights_desc, destination_desc, attributes);
    implementation_ = descriptor.impl_info_str();
    primitive_ = dnnl::matmul(descriptor);
    arguments_ = {
        {DNNL_ARG_SRC, source_}, {DNNL_ARG_WEIGHTS, weights_},
        {DNNL_ARG_DST, destination_},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, source_scales_},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_},
    };

    const auto q8 = QuantizeSource(source, spec.input);
    std::vector<std::int8_t> weights(spec.output * spec.input, 0);
    std::vector<float> scales(groups * spec.output, 0.0f);
    for (std::size_t output = 0; output < spec.output; ++output) {
      for (std::size_t group = 0; group < groups; ++group) {
        std::array<float, 32> reference{};
        float maximum = 0.0f;
        for (std::size_t within = 0; within < 32; ++within) {
          const std::size_t k = group * 32 + within;
          const std::size_t block = k / kQ4BlockValues;
          const std::size_t within_block = k % kQ4BlockValues;
          const std::uint8_t* q6 =
              raw.data() + (output * blocks + block) * kQ6BlockBytes;
          const float d = HalfToFloat(LoadU16(q6 + 208));
          const auto* scale_codes =
              reinterpret_cast<const std::int8_t*>(q6 + 192);
          reference[within] = d *
              static_cast<float>(scale_codes[within_block / 16]) *
              static_cast<float>(Q6ValueLocal(q6, within_block));
          maximum = std::max(maximum, std::abs(reference[within]));
        }
        const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        scales[group * spec.output + output] = scale;
        for (std::size_t within = 0; within < 32; ++within) {
          int code = scale == 0.0f ? 0 : NearestInt(reference[within] / scale);
          code = std::max(-127, std::min(127, code));
          weights[output * spec.input + group * 32 + within] =
              static_cast<std::int8_t>(code);
        }
      }
    }
    WriteMemory(q8.values, source_);
    WriteMemory(q8.scales, source_scales_);
    WriteMemory(weights, weights_);
    WriteMemory(scales, scales_);
  }

  void Execute(dnnl::stream& stream) override {
    primitive_.execute(stream, arguments_);
  }

  std::vector<float> ReadOutput() const override {
    return ReadMemory<float>(destination_, kTokens * spec_.output,
                             spec_.label + " output");
  }

  const std::string& Implementation() const override { return implementation_; }
  const ProjectionSpec& Spec() const override { return spec_; }

 private:
  ProjectionSpec spec_;
  dnnl::memory source_;
  dnnl::memory weights_;
  dnnl::memory destination_;
  dnnl::memory scales_;
  dnnl::memory source_scales_;
  dnnl::matmul primitive_;
  std::unordered_map<int, dnnl::memory> arguments_;
  std::string implementation_;
};

void PrintComparison(const Comparison& value) {
  std::cout << "{\"count\":" << value.count
            << ",\"max_abs\":" << value.max_abs
            << ",\"relative_l2\":" << value.relative_l2
            << ",\"cosine\":" << value.cosine
            << ",\"finite\":" << value.finite
            << ",\"passes\":" << value.passes << '}';
}

void PrintSamples(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

int Main(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  const auto index = iq36::parse_gguf_model_index(args.model.string());
  const std::vector<ProjectionSpec> specs = {
      {"qkv", "blk.0.attn_qkv.weight", "attn_norm-0",
       "linear_attn_qkv_mixed-0", 2048, 8192},
      {"z", "blk.0.attn_gate.weight", "attn_norm-0", "z-0", 2048, 4096},
      {"alpha", "blk.0.ssm_alpha.weight", "attn_norm-0", "alpha-0",
       2048, 32},
      {"beta", "blk.0.ssm_beta.weight", "attn_norm-0", "beta-0",
       2048, 32},
      {"out", "blk.0.ssm_out.weight", "final_output-0",
       "linear_attn_out-0", 4096, 2048},
  };

  dnnl::engine engine(dnnl::engine::kind::gpu, 0);
  dnnl::stream stream(engine);
  std::vector<std::unique_ptr<Projection>> projections;
  for (const auto& spec : specs) {
    const auto source = ReadVector<float>(
        CapturePayload(args.capture, spec.source_capture),
        kTokens * spec.input);
    const auto* tensor = iq36::find_tensor(index, spec.tensor);
    Require(tensor != nullptr, spec.tensor + " is missing");
    if (tensor->type == 12U) {
      projections.push_back(std::make_unique<AffineQ4Matmul>(
          engine, args.model, index, spec, source));
    } else if (tensor->type == 14U) {
      projections.push_back(std::make_unique<S8Per32Q6Matmul>(
          engine, args.model, index, spec, source));
    } else {
      Fail(spec.tensor + " has unsupported quantization type " +
           std::to_string(tensor->type));
    }
  }
  for (int iteration = 0; iteration < args.warmup; ++iteration) {
    for (auto& projection : projections) projection->Execute(stream);
    stream.wait();
  }

  std::vector<std::vector<double>> samples(projections.size());
  for (std::size_t index = 0; index < projections.size(); ++index) {
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      projections[index]->Execute(stream);
      stream.wait();
      const auto end = std::chrono::steady_clock::now();
      samples[index].push_back(std::chrono::duration<double, std::micro>(
          end - begin).count());
    }
  }
  std::vector<double> complete_samples;
  for (int iteration = 0; iteration < args.repeat; ++iteration) {
    const auto complete_begin = std::chrono::steady_clock::now();
    for (auto& projection : projections) projection->Execute(stream);
    stream.wait();
    const auto complete_end = std::chrono::steady_clock::now();
    complete_samples.push_back(std::chrono::duration<double, std::micro>(
        complete_end - complete_begin).count());
  }

  std::vector<ProjectionResult> results;
  bool correctness = true;
  for (std::size_t index = 0; index < projections.size(); ++index) {
    const auto& projection = projections[index];
    const auto reference = ReadVector<float>(
        CapturePayload(args.capture, projection->Spec().reference_capture),
        kTokens * projection->Spec().output);
    ProjectionResult result;
    result.label = projection->Spec().label;
    result.implementation = projection->Implementation();
    result.input = projection->Spec().input;
    result.output = projection->Spec().output;
    result.samples_us = samples[index];
    result.median_us = Median(samples[index]);
    result.comparison = Compare(projection->ReadOutput(), reference);
    correctness = correctness && result.comparison.passes;
    results.push_back(std::move(result));
  }

  const double complete_median_us = Median(complete_samples);
  std::cout << std::boolalpha << std::setprecision(12)
            << "{\"schema_version\":\""
            << "intel-qwen36-onednn-linear-prefill-nonstate-probe-v0\","
            << "\"token_count\":" << kTokens
            << ",\"warmup\":" << args.warmup
            << ",\"repeat\":" << args.repeat
            << ",\"complete_projection_median_us\":"
            << complete_median_us
            << ",\"complete_projection_samples_us\":";
  PrintSamples(complete_samples);
  std::cout << ",\"all_projection_correctness_passed\":" << correctness
            << ",\"projections\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index != 0) std::cout << ',';
    const auto& result = results[index];
    std::cout << "{\"label\":\"" << result.label
              << "\",\"input\":" << result.input
              << ",\"output\":" << result.output
              << ",\"implementation\":\"" << result.implementation
              << "\",\"median_us\":" << result.median_us
              << ",\"samples_us\":";
    PrintSamples(result.samples_us);
    std::cout << ",\"comparison\":";
    PrintComparison(result.comparison);
    std::cout << '}';
  }
  std::cout << "]}" << std::endl;
  return correctness ? 0 : 2;
}

}  // namespace linear_nonstate

int main(int argc, char** argv) {
  try {
    return linear_nonstate::Main(argc, argv);
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN error " << error.status << ": " << error.what()
              << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
