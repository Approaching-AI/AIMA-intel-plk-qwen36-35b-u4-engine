#define IQ36_Q4K_COMPONENT_NO_MAIN
#include "onednn_q4k_bucket_component.cpp"

#include <cstdlib>
#include <filesystem>
#include <functional>

namespace {

constexpr const char* kGroupedQ4KSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float stable_swiglu(float gate, float up) {
  const float sigmoid = gate >= 0.0f
      ? 1.0f / (1.0f + exp(-gate))
      : exp(gate) / (1.0f + exp(gate));
  return gate * sigmoid * up;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void grouped_gather_f16_sums32(
    __global const float *input,
    __global const int *token_map,
    __global half *grouped_input,
    __global float *sums32,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3;
  const uint block = task & 7U;
  if (row >= row_count) return;
  const int token = token_map[row];
  const uint inner = block * 256U + lane;
  const half stored = convert_half_rte(input[(uint)token * 2048U + inner]);
  grouped_input[row * 2048U + inner] = stored;
  __local float values[256];
  values[lane] = convert_float(stored);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    float sum = 0.0f;
    for (uint index = 0U; index < 32U; ++index) {
      sum += values[lane * 32U + index];
    }
    sums32[row * 64U + block * 8U + lane] = sum;
  }
}

int grouped_nearest_int(float value) {
  const float shifted = value + 12582912.0f;
  return (as_int(shifted) & 0x007fffff) - 0x00400000;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void grouped_quantize_tokens_q8_sums32(
    __global const float *input,
    __global char *q8,
    __global float *scales,
    __global float *sums32_scaled,
    __global char *sum_low,
    __global char *sum_high,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3;
  const uint block = task & 7U;
  if (row >= row_count) return;
  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = input[row * 2048U + block * 256U + lane];
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, grouped_nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 2048U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 8U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint item = 0U; item < 32U; ++item) {
      sum += quantized[lane * 32U + item];
    }
    sums32_scaled[row * 64U + block * 8U + lane] = scale * (float)sum;
    const int low = ((sum + 128) & 255) - 128;
    sum_low[row * 64U + block * 8U + lane] = (char)low;
    sum_high[row * 64U + block * 8U + lane] = (char)((sum - low) / 256);
  }
}

__kernel void grouped_gather_quantized_q8(
    __global const int *token_map,
    __global const char *token_q8,
    __global const float *token_scales,
    __global const char *token_sum_low,
    __global const char *token_sum_high,
    __global char *grouped_q8,
    __global float *grouped_scales,
    __global char *grouped_sum_low,
    __global char *grouped_sum_high,
    uint row_count) {
  const uint index = get_global_id(0);
  if (index >= row_count * 2048U) return;
  const uint row = index >> 11;
  const uint inner = index & 2047U;
  const uint token = (uint)token_map[row];
  grouped_q8[index] = token_q8[token * 2048U + inner];
  if (inner < 8U) {
    grouped_scales[row * 8U + inner] = token_scales[token * 8U + inner];
  }
  if (inner < 64U) {
    grouped_sum_low[row * 64U + inner] =
        token_sum_low[token * 64U + inner];
    grouped_sum_high[row * 64U + inner] =
        token_sum_high[token * 64U + inner];
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void grouped_quantize_f16_q8_sums32(
    __global const half *input,
    __global char *q8,
    __global float *scales,
    __global float *sums32_scaled,
    __global char *sum_low,
    __global char *sum_high,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 1;
  const uint block = task & 1U;
  if (row >= row_count) return;
  __local float values[256];
  __local float maxima[256];
  __local uint indices[256];
  __local int quantized[256];
  const float value = convert_float(
      input[row * 512U + block * 256U + lane]);
  values[lane] = value;
  maxima[lane] = fabs(value);
  indices[lane] = lane;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint step = 128U; step > 0U; step >>= 1U) {
    if (lane < step) {
      const float rhs = maxima[lane + step];
      const uint rhs_index = indices[lane + step];
      if (rhs > maxima[lane] ||
          (rhs == maxima[lane] && rhs_index < indices[lane])) {
        maxima[lane] = rhs;
        indices[lane] = rhs_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float max_value = values[indices[0]];
  const float inverse_scale = maxima[0] == 0.0f ? 0.0f : -127.0f / max_value;
  const float scale = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
  const int q = inverse_scale == 0.0f
      ? 0 : min(127, grouped_nearest_int(inverse_scale * value));
  quantized[lane] = q;
  q8[row * 512U + block * 256U + lane] = (char)q;
  if (lane == 0U) scales[row * 2U + block] = scale;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    int sum = 0;
    for (uint item = 0U; item < 32U; ++item) {
      sum += quantized[lane * 32U + item];
    }
    sums32_scaled[row * 16U + block * 8U + lane] = scale * (float)sum;
    const int low = ((sum + 128) & 255) - 128;
    sum_low[row * 16U + block * 8U + lane] = (char)low;
    sum_high[row * 16U + block * 8U + lane] = (char)((sum - low) / 256);
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void grouped_q4k_residual_swiglu_sums32(
    __global const half *gate_main,
    __global const half *up_main,
    __global const float *input_sums32,
    __global const float *gate_min,
    __global const float *up_min,
    __global const int *row_expert,
    __global half *swiglu_output,
    __global float *down_sums32,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 1;
  const uint block = task & 1U;
  if (row >= row_count) return;
  const uint output = block * 256U + lane;
  const uint expert = (uint)row_expert[row];
  float gate = convert_float(gate_main[row * 512U + output]);
  float up = convert_float(up_main[row * 512U + output]);
  for (uint group = 0U; group < 64U; ++group) {
    const float sum = input_sums32[row * 64U + group];
    const uint coefficient = (expert * 64U + group) * 512U + output;
    gate -= sum * gate_min[coefficient];
    up -= sum * up_min[coefficient];
  }
  const half stored = convert_half_rte(stable_swiglu(gate, up));
  swiglu_output[row * 512U + output] = stored;
  __local float values[256];
  values[lane] = convert_float(stored);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lane < 8U) {
    float sum = 0.0f;
    for (uint index = 0U; index < 32U; ++index) {
      sum += values[lane * 32U + index];
    }
    down_sums32[row * 16U + block * 8U + lane] = sum;
  }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void grouped_q4k_down_residual_weight(
    __global const half *down_main,
    __global const float *down_sums32,
    __global const float *down_min,
    __global const int *row_expert,
    __global const float *router_weights,
    __global half *contributions,
    uint row_count) {
  const uint lane = get_local_id(0);
  const uint task = get_group_id(0);
  const uint row = task >> 3;
  const uint block = task & 7U;
  if (row >= row_count) return;
  const uint output = block * 256U + lane;
  const uint expert = (uint)row_expert[row];
  float value = convert_float(down_main[row * 2048U + output]);
  for (uint group = 0U; group < 16U; ++group) {
    const uint coefficient = (expert * 16U + group) * 2048U + output;
    value -= down_sums32[row * 16U + group] * down_min[coefficient];
  }
  contributions[row * 2048U + output] =
      convert_half_rte(value * router_weights[row]);
}

__kernel void grouped_scatter_routed_output(
    __global const half *contributions,
    __global const int *token_rank_to_row,
    __global float *output) {
  const uint index = get_global_id(0);
  if (index >= 1024U * 2048U) return;
  const uint token = index / 2048U;
  const uint hidden = index - token * 2048U;
  float sum = 0.0f;
  for (uint rank = 0U; rank < 8U; ++rank) {
    const int row = token_rank_to_row[token * 8U + rank];
    sum += convert_float(contributions[(uint)row * 2048U + hidden]);
  }
  output[index] = sum;
}
)CLC";

void FillMemoryBytes(const dnnl::memory& memory, std::uint8_t value,
                     int index = 0) {
  void* mapped = memory.map_data(index);
  Require(mapped != nullptr, "oneDNN returned a null mapped pointer");
  std::memset(mapped, value, memory.get_desc().get_size(index));
  memory.unmap_data(mapped, index);
}

void WriteGroupedOffsets(const dnnl::memory& memory,
                         const std::vector<std::int32_t>& offsets) {
  Require(memory.get_desc().get_size(1) ==
              offsets.size() * sizeof(std::int32_t),
          "grouped offset buffer size mismatch");
  std::int32_t* mapped = memory.map_data<std::int32_t>(1);
  Require(mapped != nullptr, "oneDNN returned null grouped offsets");
  std::copy(offsets.begin(), offsets.end(), mapped);
  memory.unmap_data(mapped, 1);
}

struct GroupedMetadata {
  std::vector<std::int32_t> offsets;
  std::vector<std::int32_t> row_expert;
  std::vector<std::int32_t> token_map;
  std::vector<std::int32_t> inverse_map;
  std::array<std::size_t, kExpertCount> counts{};
  std::size_t active_experts = 0;
  std::size_t max_group = 0;
};

GroupedMetadata MakeGroupedMetadata(const std::vector<std::uint8_t>& topk,
                                    std::size_t stride,
                                    const BucketPlan& plan,
                                    bool enforce_locked_layer27 = true) {
  GroupedMetadata metadata;
  for (std::size_t token = 0; token < kTokenCount; ++token) {
    for (std::size_t rank = 0; rank < kSelectedExperts; ++rank) {
      std::int32_t expert = -1;
      std::memcpy(&expert,
                  topk.data() + token * stride + rank * sizeof(expert),
                  sizeof(expert));
      Require(expert >= 0 && expert < static_cast<std::int32_t>(kExpertCount),
              "grouped top-k expert out of range");
      ++metadata.counts[static_cast<std::size_t>(expert)];
    }
  }
  metadata.offsets.reserve(kExpertCount);
  metadata.row_expert.reserve(kAssignments);
  std::int32_t cumulative = 0;
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    const std::size_t count = metadata.counts[expert];
    metadata.active_experts += count != 0;
    metadata.max_group = std::max(metadata.max_group, count);
    for (std::size_t index = 0; index < count; ++index) {
      metadata.row_expert.push_back(static_cast<std::int32_t>(expert));
    }
    cumulative += static_cast<std::int32_t>(count);
    metadata.offsets.push_back(cumulative);
  }
  Require(cumulative == static_cast<std::int32_t>(kAssignments),
          "grouped assignment count mismatch");
  if (enforce_locked_layer27) {
    Require(metadata.active_experts == 222 && metadata.max_group == 361,
            "locked grouped expert shape changed");
  }
  metadata.token_map.reserve(kAssignments);
  metadata.inverse_map.assign(kAssignments, -1);
  for (std::size_t row = 0; row < kAssignments; ++row) {
    const std::size_t token = plan.bucket_token[row];
    const std::size_t rank = plan.bucket_rank[row];
    std::int32_t expert = -1;
    std::memcpy(&expert,
                topk.data() + token * stride + rank * sizeof(expert),
                sizeof(expert));
    Require(expert == metadata.row_expert[row],
            "bucket and grouped assignment order differ");
    metadata.token_map.push_back(static_cast<std::int32_t>(token));
    metadata.inverse_map[token * kSelectedExperts + rank] =
        static_cast<std::int32_t>(row);
  }
  Require(std::none_of(metadata.inverse_map.begin(), metadata.inverse_map.end(),
                       [](std::int32_t value) { return value < 0; }),
          "grouped inverse map is incomplete");
  return metadata;
}

std::uint8_t Q4Code(const std::uint8_t* block, std::size_t within_block) {
  const std::size_t segment = within_block / 64;
  const std::size_t offset = within_block % 32;
  const std::uint8_t packed = block[16 + segment * 32 + offset];
  return static_cast<std::uint8_t>(
      (within_block & 32U) == 0 ? packed & 15U : packed >> 4);
}

void SetU4(std::vector<std::uint8_t>& packed, std::size_t logical_index,
           std::uint8_t value) {
  std::uint8_t& byte = packed[logical_index >> 1];
  if ((logical_index & 1U) == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xf0U) | value);
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0fU) | (value << 4));
  }
}

std::uint8_t GetU4(const std::vector<std::uint8_t>& packed,
                   std::size_t logical_index) {
  const std::uint8_t byte = packed[logical_index >> 1];
  return static_cast<std::uint8_t>(
      (logical_index & 1U) == 0 ? byte & 15U : byte >> 4);
}

struct RepackStats {
  std::uint64_t resident_code_count = 0;
  std::uint64_t active_code_count = 0;
  std::uint64_t mismatch_count = 0;
};

class GroupedLinear {
 public:
  GroupedLinear(const dnnl::engine& engine, const dnnl::memory& input,
                const std::vector<std::int32_t>& offsets,
                const std::array<std::size_t, kExpertCount>& counts,
                const std::vector<std::uint8_t>& q4, std::size_t input_width,
                std::size_t output_width, std::size_t source_output_start,
                std::size_t source_rows_per_expert,
                std::size_t source_blocks_per_row, std::size_t max_group)
      : k_(input_width), n_(output_width), source_(input) {
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const bool generate_s8 = std::getenv("IQ36_GENERATE_S8_GROUPED") != nullptr;
    const bool f32_down_output = generate_s8 &&
        std::getenv("IQ36_GROUPED_F32_DOWN_OUTPUT") != nullptr &&
        output_width == kDownOutputSize;
    const auto weights_desc = dnnl::memory::desc(
        {static_cast<int>(kExpertCount), static_cast<int>(k_),
         static_cast<int>(n_)}, dt::u4, tag::acb);
    const auto output_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kAssignments), static_cast<int>(n_)},
        f32_down_output ? dt::f32 : dt::f16, 0,
        kExpertCount);
    const std::size_t groups = k_ / 32;
    const auto coefficient_desc = dnnl::memory::desc(
        {static_cast<int>(kExpertCount), static_cast<int>(groups),
         static_cast<int>(n_)}, dt::f32, tag::abc);
    const auto zero_point_desc = dnnl::memory::desc(
        {static_cast<int>(kExpertCount), static_cast<int>(groups),
         static_cast<int>(n_)}, dt::u4, tag::abc);
    weights_ = dnnl::memory(weights_desc, engine);
    destination_ = dnnl::memory(output_desc, engine);
    scales_ = dnnl::memory(coefficient_desc, engine);
    mins_ = dnnl::memory(coefficient_desc, engine);
    zero_points_ = dnnl::memory(zero_point_desc, engine);
    compact_min_codes_ = dnnl::memory(
        dnnl::memory::desc(
            {static_cast<int>(kExpertCount * n_ * source_blocks_per_row * 8)},
            dt::u8, tag::a),
        engine);
    compact_dmins_ = dnnl::memory(
        dnnl::memory::desc(
            {static_cast<int>(kExpertCount * n_ * source_blocks_per_row)},
            dt::f32, tag::a),
        engine);
    WriteGroupedOffsets(destination_, offsets);
    FillMemoryBytes(destination_, 0, 0);
    FillMemoryBytes(zero_points_, 0);

    dnnl::primitive_attr attributes;
    attributes.set_scales(DNNL_ARG_WEIGHTS, 7, {32, 1}, dt::f32);
    if (generate_s8) {
      const auto source_scale_desc = dnnl::memory::desc(
          {static_cast<int>(kAssignments), static_cast<int>(k_ / 256)},
          dt::f32, tag::ab);
      source_scales_ = dnnl::memory(source_scale_desc, engine);
      std::vector<float> source_scales(kAssignments * (k_ / 256), 1.0f);
      WriteMemory(source_scales, source_scales_);
      attributes.set_scales(DNNL_ARG_SRC, 3, {1, 256}, dt::f32);
    } else {
      attributes.set_zero_points(DNNL_ARG_WEIGHTS, 7, {32, 1}, dt::u4);
    }
    attributes.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    descriptor_ = std::make_unique<dnnl::matmul::primitive_desc>(
        engine, source_.get_desc(), weights_desc, output_desc, attributes);
    implementation_ = descriptor_->impl_info_str();
    primitive_ = std::make_unique<dnnl::matmul>(*descriptor_);
    arguments_ = {
        {DNNL_ARG_SRC, source_},
        {DNNL_ARG_WEIGHTS, weights_},
        {DNNL_ARG_DST, destination_},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_},
    };
    if (generate_s8) {
      arguments_.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                         source_scales_);
    } else {
      arguments_.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                         zero_points_);
    }
    max_group_hint_ = dnnl::memory(
        dnnl::memory::desc::host_scalar(dt::s32),
        static_cast<std::int32_t>(max_group));
    arguments_.emplace(DNNL_ARG_HINT_MAX_GROUP_SIZE, max_group_hint_);

    const std::size_t logical_count = kExpertCount * k_ * n_;
    std::vector<std::uint8_t> packed(logical_count / 2, 0);
    std::vector<float> scales(kExpertCount * groups * n_, 0.0f);
    std::vector<float> mins(kExpertCount * groups * n_, 0.0f);
    std::vector<std::uint8_t> zero_points(
        kExpertCount * groups * n_ / 2, 0);
    std::vector<std::uint8_t> compact_min_codes(
        kExpertCount * n_ * source_blocks_per_row * 8, 0);
    scale_codes_.resize(
        kExpertCount * n_ * source_blocks_per_row * 8, 0);
    std::vector<float> compact_dmins(
        kExpertCount * n_ * source_blocks_per_row, 0.0f);
    block_ds_.resize(
        kExpertCount * n_ * source_blocks_per_row, 0.0f);
    for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
      const bool active = counts[expert] != 0;
      for (std::size_t output = 0; output < n_; ++output) {
        const std::size_t source_output = source_output_start + output;
        for (std::size_t block = 0; block < source_blocks_per_row; ++block) {
          const std::uint8_t* q4_block = Q4BlockLayout(
              q4, expert, source_output, block, source_rows_per_expert,
              source_blocks_per_row);
          const float d = HalfToFloat(LoadU16(q4_block));
          const float dmin = HalfToFloat(LoadU16(q4_block + 2));
          const std::size_t compact_block =
              (expert * n_ + output) * source_blocks_per_row + block;
          compact_dmins[compact_block] = dmin;
          block_ds_[compact_block] = d;
          for (std::size_t group = 0; group < 8; ++group) {
            const std::size_t coefficient =
                (expert * groups + block * 8 + group) * n_ + output;
            const float scale =
                d * static_cast<float>(GetScale(group, q4_block + 4));
            const float minimum =
                dmin * static_cast<float>(GetMinimum(group, q4_block + 4));
            scales[coefficient] = scale;
            mins[coefficient] = minimum;
            compact_min_codes[compact_block * 8 + group] =
                GetMinimum(group, q4_block + 4);
            scale_codes_[compact_block * 8 + group] =
                GetScale(group, q4_block + 4);
          }
          for (std::size_t within = 0; within < kQ4BlockValues; ++within) {
            const std::uint8_t code = Q4Code(q4_block, within);
            const std::size_t k = block * kQ4BlockValues + within;
            const std::size_t logical = (expert * n_ + output) * k_ + k;
            SetU4(packed, logical, code);
            ++repack_.resident_code_count;
            repack_.active_code_count += active;
            repack_.mismatch_count += GetU4(packed, logical) != code;
          }
        }
      }
    }
    WriteMemory(packed, weights_);
    WriteMemory(scales, scales_);
    WriteMemory(mins, mins_);
    WriteMemory(zero_points, zero_points_);
    WriteMemory(compact_min_codes, compact_min_codes_);
    WriteMemory(compact_dmins, compact_dmins_);
  }

  void Execute(dnnl::stream& stream) {
    primitive_->execute(stream, arguments_);
  }

  const dnnl::memory& Destination() const { return destination_; }
  const dnnl::memory& Minimums() const { return mins_; }
  const dnnl::memory& Weights() const { return weights_; }
  const dnnl::memory& Scales() const { return scales_; }
  const dnnl::memory& ZeroPoints() const { return zero_points_; }
  const dnnl::memory& CompactMinCodes() const { return compact_min_codes_; }
  const dnnl::memory& CompactDmins() const { return compact_dmins_; }
  const std::vector<std::uint8_t>& ScaleCodes() const {
    return scale_codes_;
  }
  const std::vector<float>& BlockDs() const { return block_ds_; }
  const std::string& Implementation() const { return implementation_; }
  const RepackStats& Repack() const { return repack_; }
  std::size_t InputWidth() const { return k_; }
  std::size_t OutputWidth() const { return n_; }

  std::uint64_t ResidentBytes() const {
    return weights_.get_desc().get_size() + scales_.get_desc().get_size() +
           mins_.get_desc().get_size() +
           zero_points_.get_desc().get_size() +
           compact_min_codes_.get_desc().get_size() +
           compact_dmins_.get_desc().get_size();
  }

 private:
  std::size_t k_ = 0;
  std::size_t n_ = 0;
  dnnl::memory source_;
  dnnl::memory weights_;
  dnnl::memory destination_;
  dnnl::memory scales_;
  dnnl::memory mins_;
  dnnl::memory zero_points_;
  dnnl::memory compact_min_codes_;
  dnnl::memory compact_dmins_;
  std::vector<std::uint8_t> scale_codes_;
  std::vector<float> block_ds_;
  dnnl::memory source_scales_;
  dnnl::memory max_group_hint_;
  std::unique_ptr<dnnl::matmul::primitive_desc> descriptor_;
  std::unique_ptr<dnnl::matmul> primitive_;
  std::unordered_map<int, dnnl::memory> arguments_;
  std::string implementation_;
  RepackStats repack_;
};

cl_program LoadGroupedBinary(cl_context context, cl_device_id device,
                             const std::string& path) {
  const std::vector<std::uint8_t> binary = ReadBytes(path);
  const std::size_t size = binary.size();
  const unsigned char* data = binary.data();
  cl_int binary_status = CL_SUCCESS;
  cl_int status = CL_SUCCESS;
  cl_program program = clCreateProgramWithBinary(
      context, 1, &device, &size, &data, &binary_status, &status);
  CheckCl(status, "clCreateProgramWithBinary grouped microkernel");
  CheckCl(binary_status, "grouped microkernel binary status");
  status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
  if (status != CL_SUCCESS) {
    const std::string log = ProgramBuildLog(program, device);
    clReleaseProgram(program);
    Fail("grouped microkernel binary build failed: " + log);
  }
  return program;
}

cl_mem CreateCopiedBuffer(cl_context context, const void* data,
                          std::size_t bytes, const std::string& label) {
  cl_int status = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
      const_cast<void*>(data), &status);
  CheckCl(status, "clCreateBuffer " + label);
  return buffer;
}

void WriteBinaryFile(const std::filesystem::path& path, const void* data,
                     std::size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  Require(static_cast<bool>(output), "could not create output: " + path.string());
  output.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
  Require(static_cast<bool>(output), "could not write output: " + path.string());
}

template <typename Value>
void WriteBinaryVector(const std::filesystem::path& path,
                       const std::vector<Value>& values) {
  WriteBinaryFile(path, values.data(), values.size() * sizeof(Value));
}

std::vector<std::uint8_t> ReadClBytes(cl_command_queue queue, cl_mem buffer,
                                      std::size_t bytes,
                                      const std::string& label) {
  std::vector<std::uint8_t> result(bytes);
  CheckCl(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bytes, result.data(),
                              0, nullptr, nullptr),
          "clEnqueueReadBuffer " + label);
  return result;
}

template <typename Value>
std::vector<Value> ReadDnnlMemory(const dnnl::memory& memory,
                                  std::size_t count,
                                  const std::string& label) {
  Require(memory.get_desc().get_size() == count * sizeof(Value),
          label + " byte size mismatch");
  const Value* mapped = memory.map_data<Value>();
  Require(mapped != nullptr, "could not map " + label);
  std::vector<Value> result(mapped, mapped + count);
  memory.unmap_data(const_cast<Value*>(mapped));
  return result;
}

struct InterleavedGateUp {
  cl_mem weights = nullptr;
  cl_mem scales = nullptr;
  cl_mem compact_min_codes = nullptr;
  cl_mem compact_dmins = nullptr;
  cl_mem scale_codes = nullptr;
  cl_mem block_ds = nullptr;
};

InterleavedGateUp MakeInterleavedGateUp(cl_context context,
                                        const GroupedLinear& gate,
                                        const GroupedLinear& up) {
  constexpr std::size_t kOutput = 512;
  constexpr std::size_t kInterleavedOutput = 1024;
  constexpr std::size_t kGroups = kHiddenSize / 32;
  constexpr std::size_t kRowBytes = kHiddenSize / 2;
  const auto gate_weights = ReadDnnlMemory<std::uint8_t>(
      gate.Weights(), kExpertCount * kOutput * kRowBytes, "gate weights");
  const auto up_weights = ReadDnnlMemory<std::uint8_t>(
      up.Weights(), kExpertCount * kOutput * kRowBytes, "up weights");
  std::vector<std::uint8_t> weights(
      kExpertCount * kInterleavedOutput * kRowBytes);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    for (std::size_t output = 0; output < kOutput; ++output) {
      const std::size_t source = (expert * kOutput + output) * kRowBytes;
      const std::size_t destination =
          (expert * kInterleavedOutput + output * 2) * kRowBytes;
      std::memcpy(weights.data() + destination,
                  gate_weights.data() + source, kRowBytes);
      std::memcpy(weights.data() + destination + kRowBytes,
                  up_weights.data() + source, kRowBytes);
    }
  }

  const std::size_t coefficient_count = kExpertCount * kGroups * kOutput;
  const auto gate_scales = ReadDnnlMemory<float>(
      gate.Scales(), coefficient_count, "gate scales");
  const auto up_scales = ReadDnnlMemory<float>(
      up.Scales(), coefficient_count, "up scales");
  std::vector<float> scales(coefficient_count * 2);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    for (std::size_t group = 0; group < kGroups; ++group) {
      for (std::size_t output = 0; output < kOutput; ++output) {
        const std::size_t source =
            (expert * kGroups + group) * kOutput + output;
        const std::size_t destination =
            (expert * kGroups + group) * kInterleavedOutput + output * 2;
        scales[destination] = gate_scales[source];
        scales[destination + 1] = up_scales[source];
      }
    }
  }
  InterleavedGateUp result;
  result.weights = CreateCopiedBuffer(
      context, weights.data(), weights.size(), "interleaved gate/up weights");
  result.scales = CreateCopiedBuffer(context, scales.data(),
      scales.size() * sizeof(float), "interleaved gate/up scales");
  constexpr std::size_t kBlocks = kHiddenSize / kQ4BlockValues;
  const std::size_t compact_code_count =
      kExpertCount * kOutput * kBlocks * 8;
  const std::size_t compact_dmin_count = kExpertCount * kOutput * kBlocks;
  const auto gate_min_codes = ReadDnnlMemory<std::uint8_t>(
      gate.CompactMinCodes(), compact_code_count, "gate compact min codes");
  const auto up_min_codes = ReadDnnlMemory<std::uint8_t>(
      up.CompactMinCodes(), compact_code_count, "up compact min codes");
  const auto gate_dmins = ReadDnnlMemory<float>(
      gate.CompactDmins(), compact_dmin_count, "gate compact dmins");
  const auto up_dmins = ReadDnnlMemory<float>(
      up.CompactDmins(), compact_dmin_count, "up compact dmins");
  std::vector<std::uint8_t> compact_min_codes(compact_code_count * 2);
  std::vector<float> compact_dmins(compact_dmin_count * 2);
  std::vector<std::uint8_t> scale_codes(compact_code_count * 2);
  std::vector<float> block_ds(compact_dmin_count * 2);
  for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
    for (std::size_t output = 0; output < kOutput; ++output) {
      for (std::size_t block = 0; block < kBlocks; ++block) {
        const std::size_t source =
            (expert * kOutput + output) * kBlocks + block;
        const std::size_t destination =
            (expert * kInterleavedOutput + output * 2) * kBlocks + block;
        compact_dmins[destination] = gate_dmins[source];
        compact_dmins[destination + kBlocks] = up_dmins[source];
        block_ds[destination] = gate.BlockDs()[source];
        block_ds[destination + kBlocks] = up.BlockDs()[source];
        std::memcpy(compact_min_codes.data() + destination * 8,
                    gate_min_codes.data() + source * 8, 8);
        std::memcpy(compact_min_codes.data() +
                        (destination + kBlocks) * 8,
                    up_min_codes.data() + source * 8, 8);
        std::memcpy(scale_codes.data() + destination * 8,
                    gate.ScaleCodes().data() + source * 8, 8);
        std::memcpy(scale_codes.data() +
                        (destination + kBlocks) * 8,
                    up.ScaleCodes().data() + source * 8, 8);
      }
    }
  }
  result.compact_min_codes = CreateCopiedBuffer(context,
      compact_min_codes.data(), compact_min_codes.size(),
      "interleaved gate/up compact min codes");
  result.compact_dmins = CreateCopiedBuffer(context, compact_dmins.data(),
      compact_dmins.size() * sizeof(float),
      "interleaved gate/up compact dmins");
  result.scale_codes = CreateCopiedBuffer(
      context, scale_codes.data(), scale_codes.size(),
      "interleaved gate/up scale codes");
  result.block_ds = CreateCopiedBuffer(
      context, block_ds.data(), block_ds.size() * sizeof(float),
      "interleaved gate/up block ds");
  return result;
}

void ReleaseInterleavedGateUp(InterleavedGateUp* buffers) {
  for (cl_mem memory : {buffers->weights, buffers->scales,
                        buffers->compact_min_codes,
                        buffers->compact_dmins, buffers->scale_codes,
                        buffers->block_ds}) {
    if (memory != nullptr) clReleaseMemObject(memory);
  }
  *buffers = {};
}

void SetKernelMemory(cl_kernel kernel, cl_uint index, cl_mem memory,
                     const char* label) {
  CheckCl(clSetKernelArg(kernel, index, sizeof(memory), &memory), label);
}

void SetKernelI64(cl_kernel kernel, cl_uint index, std::int64_t value,
                  const char* label) {
  CheckCl(clSetKernelArg(kernel, index, sizeof(value), &value), label);
}

void EnqueueIq36S8GateUp(cl_command_queue queue, cl_kernel kernel,
                         const InterleavedGateUp& gateup, cl_mem source,
                         cl_mem source_scales, cl_mem sum_low,
                         cl_mem sum_high,
                         cl_mem destination, cl_mem offsets, cl_mem dummy,
                         std::size_t max_group) {
  constexpr std::int64_t k = kHiddenSize;
  constexpr std::int64_t n = kGateUpSize;
  constexpr std::int64_t ldsrc = kHiddenSize;
  constexpr std::int64_t lddst = kIntermediateSize;
  constexpr std::int64_t ldsrcq = 8;
  constexpr std::int64_t ldweiq = n;
  const std::array<std::int64_t, 4> strides = {k * n, 1, k, 0};
  SetKernelMemory(kernel, 0, source, "set S8 gate/up source");
  SetKernelI64(kernel, 1, ldsrc, "set S8 gate/up ldsrc");
  SetKernelMemory(kernel, 2, gateup.weights, "set S8 gate/up weights");
  CheckCl(clSetKernelArg(kernel, 3, sizeof(strides), strides.data()),
          "set S8 gate/up strides");
  SetKernelMemory(kernel, 4, destination, "set S8 gate/up destination");
  SetKernelI64(kernel, 5, lddst, "set S8 gate/up lddst");
  SetKernelMemory(kernel, 6, offsets, "set S8 gate/up offsets");
  SetKernelMemory(kernel, 7, dummy, "set S8 gate/up unused weights");
  SetKernelMemory(kernel, 8, source_scales, "set S8 gate/up source scales");
  SetKernelMemory(kernel, 9, sum_low, "set S8 gate/up low sums");
  SetKernelI64(kernel, 10, ldsrcq, "set S8 gate/up ldsrcq");
  SetKernelMemory(kernel, 11, gateup.scales, "set S8 gate/up weight scales");
  SetKernelMemory(kernel, 12, gateup.compact_min_codes,
                  "set S8 gate/up compact min codes");
  SetKernelI64(kernel, 13, ldweiq, "set S8 gate/up ldweiq");
  SetKernelI64(kernel, 14, n, "set S8 gate/up n");
  SetKernelI64(kernel, 15, k, "set S8 gate/up k");
  SetKernelMemory(kernel, 16, sum_high, "set S8 gate/up high sums");
  SetKernelMemory(kernel, 17, gateup.compact_dmins,
                  "set S8 gate/up compact dmins");
  SetKernelMemory(kernel, 18, dummy, "set S8 gate/up unused pointer 0");
  SetKernelMemory(kernel, 19, dummy, "set S8 gate/up unused pointer 1");
  constexpr std::array<std::size_t, 3> local = {32, 4, 1};
  const std::array<std::size_t, 3> global = {
      local[0] * ((static_cast<std::size_t>(n) + 63) / 64),
      local[1] * ((max_group + 31) / 32), kExpertCount};
  CheckCl(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global.data(),
                                 local.data(), 0, nullptr, nullptr),
          "enqueue S8 gate/up");
}

void EnqueueIq36S8Down(cl_command_queue queue, cl_kernel kernel,
                       const GroupedLinear& down, cl_mem source,
                       cl_mem source_scales, cl_mem sum_low, cl_mem sum_high,
                       cl_mem destination, cl_mem offsets,
                       cl_mem router_weights, cl_mem dummy,
                       std::size_t max_group) {
  const auto Mem = [](const dnnl::memory& memory) {
    return dnnl::ocl_interop::get_mem_object(memory);
  };
  constexpr std::int64_t k = kDownInputSize;
  constexpr std::int64_t n = kDownOutputSize;
  constexpr std::int64_t ldsrc = kDownInputSize;
  constexpr std::int64_t lddst = kDownOutputSize;
  constexpr std::int64_t ldsrcq = 2;
  constexpr std::int64_t ldweiq = n;
  const std::array<std::int64_t, 4> strides = {k * n, 1, k, 0};
  SetKernelMemory(kernel, 0, source, "set S8 down source");
  SetKernelI64(kernel, 1, ldsrc, "set S8 down ldsrc");
  SetKernelMemory(kernel, 2, Mem(down.Weights()), "set S8 down weights");
  CheckCl(clSetKernelArg(kernel, 3, sizeof(strides), strides.data()),
          "set S8 down strides");
  SetKernelMemory(kernel, 4, destination, "set S8 down destination");
  SetKernelI64(kernel, 5, lddst, "set S8 down lddst");
  SetKernelMemory(kernel, 6, offsets, "set S8 down offsets");
  SetKernelMemory(kernel, 7, dummy, "set S8 down unused offsets");
  SetKernelMemory(kernel, 8, source_scales, "set S8 down source scales");
  SetKernelMemory(kernel, 9, sum_low, "set S8 down low sums");
  SetKernelI64(kernel, 10, ldsrcq, "set S8 down ldsrcq");
  SetKernelMemory(kernel, 11, Mem(down.Scales()),
                  "set S8 down weight scales");
  SetKernelMemory(kernel, 12, Mem(down.CompactMinCodes()),
                  "set S8 down compact min codes");
  SetKernelI64(kernel, 13, ldweiq, "set S8 down ldweiq");
  SetKernelI64(kernel, 14, n, "set S8 down n");
  SetKernelI64(kernel, 15, k, "set S8 down k");
  SetKernelMemory(kernel, 16, router_weights,
                  "set S8 down router weights");
  SetKernelMemory(kernel, 17, sum_high, "set S8 down high sums");
  SetKernelMemory(kernel, 18, Mem(down.CompactDmins()),
                  "set S8 down compact dmins");
  SetKernelMemory(kernel, 19, dummy, "set S8 down unused pointer");
  constexpr std::array<std::size_t, 3> local = {32, 4, 1};
  const std::array<std::size_t, 3> global = {
      local[0] * ((static_cast<std::size_t>(n) + 63) / 64),
      local[1] * ((max_group + 31) / 32), kExpertCount};
  CheckCl(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global.data(),
                                 local.data(), 0, nullptr, nullptr),
          "enqueue S8 down");
}

cl_program BuildGroupedProgram(cl_context context, cl_device_id device,
                               std::string* build_log) {
  const char* source = kGroupedQ4KSource;
  const std::size_t length = std::strlen(source);
  cl_int status = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, &length, &status);
  CheckCl(status, "clCreateProgramWithSource grouped Q4_K");
  status = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
  *build_log = ProgramBuildLog(program, device);
  if (status != CL_SUCCESS) {
    clReleaseProgram(program);
    Fail("grouped Q4_K program build failed: " + *build_log);
  }
  return program;
}

double TimedStage(const std::function<void()>& enqueue,
                  cl_command_queue queue) {
  const auto begin = std::chrono::steady_clock::now();
  enqueue();
  CheckCl(clFinish(queue), "clFinish grouped stage");
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - begin)
      .count();
}

std::vector<float> ReadHalfMemory(const dnnl::memory& memory,
                                  std::size_t count) {
  const std::uint16_t* mapped = memory.map_data<std::uint16_t>();
  Require(mapped != nullptr, "could not map F16 grouped output");
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = HalfToFloat(mapped[index]);
  }
  memory.unmap_data(const_cast<std::uint16_t*>(mapped));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    Require(RoutedMode(args), "grouped Q4_K gate requires routed mode");
    const std::size_t topk_stride = args.prepack_only
        ? kSelectedExperts * sizeof(std::int32_t) : args.topk_stride;
    const auto input = args.prepack_only
        ? std::vector<float>(kTokenCount * kHiddenSize, 0.0f)
        : ReadVector<float>(args.input, kTokenCount * kHiddenSize);
    auto topk = args.prepack_only
        ? std::vector<std::uint8_t>(kTokenCount * topk_stride)
        : ReadBytes(args.topk);
    if (args.prepack_only) {
      for (std::size_t token = 0; token < kTokenCount; ++token) {
        for (std::size_t rank = 0; rank < kSelectedExperts; ++rank) {
          const std::int32_t expert = static_cast<std::int32_t>(
              (token * kSelectedExperts + rank) % kExpertCount);
          std::memcpy(topk.data() + token * topk_stride +
                          rank * sizeof(expert),
                      &expert, sizeof(expert));
        }
      }
    }
    const auto oracle = args.prepack_only
        ? std::vector<float>()
        : ReadVector<float>(args.oracle, kAssignments * kIntermediateSize);
    const auto gate_up_q4 = ReadModelSlice(
        args.model, args.weight_offset, args.weight_bytes);
    const auto down_q4 = ReadModelSlice(
        args.model, args.down_weight_offset, args.down_weight_bytes);
    const auto router_weights = args.prepack_only
        ? std::vector<float>(kAssignments, 0.0f)
        : ReadVector<float>(args.router_weights, kAssignments);
    const auto down_oracle = args.prepack_only
        ? std::vector<float>()
        : ReadVector<float>(args.down_oracle,
                            kAssignments * kDownOutputSize);
    const auto moe_oracle = args.prepack_only
        ? std::vector<float>()
        : ReadVector<float>(args.moe_oracle, kTokenCount * kHiddenSize);
    const BucketPlan plan = BuildPlan(topk, topk_stride, !args.prepack_only);
    const GroupedMetadata metadata =
        MakeGroupedMetadata(topk, topk_stride, plan, !args.prepack_only);

    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    const cl_context context = dnnl::ocl_interop::get_context(engine);
    const cl_device_id device = dnnl::ocl_interop::get_device(engine);
    const cl_command_queue queue = dnnl::ocl_interop::get_command_queue(stream);
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    const bool generate_s8 = std::getenv("IQ36_GENERATE_S8_GROUPED") != nullptr;
    const char* fused_kind = std::getenv("IQ36_GROUPED_FUSED_KIND");
    const bool generate_down_first =
        generate_s8 && fused_kind != nullptr &&
        std::string(fused_kind) == "down";
    const auto grouped_input_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kAssignments), static_cast<int>(kHiddenSize)},
        generate_s8 ? dt::s8 : dt::f16, 0, kExpertCount);
    dnnl::memory grouped_input(grouped_input_desc, engine);
    WriteGroupedOffsets(grouped_input, metadata.offsets);
    FillMemoryBytes(grouped_input, 0, 0);
    const auto swiglu_desc = dnnl::memory::desc::grouped(
        {static_cast<int>(kAssignments), static_cast<int>(kIntermediateSize)},
        dt::f16, 0, kExpertCount);
    dnnl::memory swiglu(swiglu_desc, engine);
    WriteGroupedOffsets(swiglu, metadata.offsets);
    FillMemoryBytes(swiglu, 0, 0);
    dnnl::memory down_descriptor_source = swiglu;
    if (generate_s8) {
      const auto down_source_desc = dnnl::memory::desc::grouped(
          {static_cast<int>(kAssignments),
           static_cast<int>(kIntermediateSize)},
          dt::s8, 0, kExpertCount);
      down_descriptor_source = dnnl::memory(down_source_desc, engine);
      WriteGroupedOffsets(down_descriptor_source, metadata.offsets);
      FillMemoryBytes(down_descriptor_source, 0, 0);
    }
    std::unique_ptr<GroupedLinear> gate_holder;
    std::unique_ptr<GroupedLinear> up_holder;
    std::unique_ptr<GroupedLinear> down_holder;
    const auto make_gate = [&]() {
      return std::make_unique<GroupedLinear>(
          engine, grouped_input, metadata.offsets, metadata.counts,
          gate_up_q4, kHiddenSize, kIntermediateSize, 0, kRowsPerExpert,
          kBlocksPerRow, metadata.max_group);
    };
    const auto make_up = [&]() {
      return std::make_unique<GroupedLinear>(
          engine, grouped_input, metadata.offsets, metadata.counts,
          gate_up_q4, kHiddenSize, kIntermediateSize, kIntermediateSize,
          kRowsPerExpert, kBlocksPerRow, metadata.max_group);
    };
    const auto make_down = [&]() {
      return std::make_unique<GroupedLinear>(
          engine, down_descriptor_source, metadata.offsets, metadata.counts,
          down_q4, kDownInputSize, kDownOutputSize, 0, kDownOutputSize,
          kDownBlocksPerRow, metadata.max_group);
    };
    if (generate_down_first) {
      down_holder = make_down();
      gate_holder = make_gate();
      up_holder = make_up();
    } else {
      gate_holder = make_gate();
      up_holder = make_up();
      down_holder = make_down();
    }
    GroupedLinear& gate = *gate_holder;
    GroupedLinear& up = *up_holder;
    GroupedLinear& down = *down_holder;

    cl_program fused_gateup_program = nullptr;
    cl_program fused_down_program = nullptr;
    cl_kernel fused_gateup_kernel = nullptr;
    cl_kernel fused_down_kernel = nullptr;
    InterleavedGateUp interleaved_gateup;
    cl_mem grouped_binary_offsets = nullptr;
    cl_mem grouped_binary_dummy = nullptr;
    const bool native_binary_mode = !args.grouped_gateup_binary.empty();
    Require(args.dump_prepacked_dir.empty() || native_binary_mode,
            "prepacked dump requires grouped S8 fused mode");
    if (native_binary_mode) {
      fused_gateup_program = LoadGroupedBinary(
          context, device, args.grouped_gateup_binary);
      fused_down_program = LoadGroupedBinary(
          context, device, args.grouped_down_binary);
      cl_int status = CL_SUCCESS;
      fused_gateup_kernel = clCreateKernel(
          fused_gateup_program, "grouped_micro_gemm", &status);
      CheckCl(status, "clCreateKernel fused gate/up");
      fused_down_kernel = clCreateKernel(
          fused_down_program, "grouped_micro_gemm", &status);
      CheckCl(status, "clCreateKernel fused down");
      interleaved_gateup = MakeInterleavedGateUp(context, gate, up);
    }
    if (native_binary_mode) {
      grouped_binary_offsets = CreateCopiedBuffer(
          context, metadata.offsets.data(),
          metadata.offsets.size() * sizeof(metadata.offsets.front()),
          "grouped binary offsets");
      const std::uint32_t zero = 0;
      grouped_binary_dummy = CreateCopiedBuffer(
          context, &zero, sizeof(zero), "grouped binary dummy");
    }
    const auto ReleaseNativeResources = [&]() {
      if (grouped_binary_dummy != nullptr) {
        clReleaseMemObject(grouped_binary_dummy);
        grouped_binary_dummy = nullptr;
      }
      if (grouped_binary_offsets != nullptr) {
        clReleaseMemObject(grouped_binary_offsets);
        grouped_binary_offsets = nullptr;
      }
      if (fused_down_kernel != nullptr) {
        clReleaseKernel(fused_down_kernel);
        fused_down_kernel = nullptr;
      }
      if (fused_gateup_kernel != nullptr) {
        clReleaseKernel(fused_gateup_kernel);
        fused_gateup_kernel = nullptr;
      }
      if (fused_down_program != nullptr) {
        clReleaseProgram(fused_down_program);
        fused_down_program = nullptr;
      }
      if (fused_gateup_program != nullptr) {
        clReleaseProgram(fused_gateup_program);
        fused_gateup_program = nullptr;
      }
      ReleaseInterleavedGateUp(&interleaved_gateup);
    };

    dnnl::memory input_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), static_cast<int>(kHiddenSize)},
            dt::f32, tag::ab),
        engine);
    dnnl::memory token_map_memory(
        dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::s32, tag::a),
        engine);
    dnnl::memory row_expert_memory(
        dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::s32, tag::a),
        engine);
    dnnl::memory inverse_map_memory(
        dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::s32, tag::a),
        engine);
    dnnl::memory compact_weight_memory(
        dnnl::memory::desc({static_cast<int>(kAssignments)}, dt::f32, tag::a),
        engine);
    dnnl::memory input_sums_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 64}, dt::f32, tag::ab),
        engine);
    dnnl::memory grouped_q8_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), static_cast<int>(kHiddenSize)},
            dt::s8, tag::ab),
        engine);
    dnnl::memory token_q8_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), static_cast<int>(kHiddenSize)},
            dt::s8, tag::ab),
        engine);
    dnnl::memory token_scale_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), 8}, dt::f32, tag::ab),
        engine);
    dnnl::memory token_sums_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), 64}, dt::f32, tag::ab),
        engine);
    dnnl::memory token_sum_low_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), 64}, dt::s8, tag::ab),
        engine);
    dnnl::memory token_sum_high_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), 64}, dt::s8, tag::ab),
        engine);
    dnnl::memory input_scale_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 8}, dt::f32, tag::ab),
        engine);
    dnnl::memory input_sum_low_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 64}, dt::s8, tag::ab),
        engine);
    dnnl::memory input_sum_high_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 64}, dt::s8, tag::ab),
        engine);
    dnnl::memory down_sums_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 16}, dt::f32, tag::ab),
        engine);
    dnnl::memory down_q8_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments),
             static_cast<int>(kIntermediateSize)},
            dt::s8, tag::ab),
        engine);
    dnnl::memory down_scale_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 2}, dt::f32, tag::ab),
        engine);
    dnnl::memory down_sum_low_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 16}, dt::s8, tag::ab),
        engine);
    dnnl::memory down_sum_high_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), 16}, dt::s8, tag::ab),
        engine);
    dnnl::memory contributions_memory(
        dnnl::memory::desc(
            {static_cast<int>(kAssignments), static_cast<int>(kHiddenSize)},
            dt::f16, tag::ab),
        engine);
    dnnl::memory output_memory(
        dnnl::memory::desc(
            {static_cast<int>(kTokenCount), static_cast<int>(kHiddenSize)},
            dt::f32, tag::ab),
        engine);
    WriteMemory(input, input_memory);
    WriteMemory(metadata.token_map, token_map_memory);
    WriteMemory(metadata.row_expert, row_expert_memory);
    WriteMemory(metadata.inverse_map, inverse_map_memory);
    std::vector<float> compact_weights(kAssignments);
    for (std::size_t row = 0; row < kAssignments; ++row) {
      compact_weights[row] = router_weights[
          plan.bucket_token[row] * kSelectedExperts + plan.bucket_rank[row]];
    }
    WriteMemory(compact_weights, compact_weight_memory);
    FillMemoryBytes(input_sums_memory, 0);
    FillMemoryBytes(grouped_q8_memory, 0);
    FillMemoryBytes(token_q8_memory, 0);
    FillMemoryBytes(token_scale_memory, 0);
    FillMemoryBytes(token_sums_memory, 0);
    FillMemoryBytes(token_sum_low_memory, 0);
    FillMemoryBytes(token_sum_high_memory, 0);
    FillMemoryBytes(input_scale_memory, 0);
    FillMemoryBytes(input_sum_low_memory, 0);
    FillMemoryBytes(input_sum_high_memory, 0);
    FillMemoryBytes(down_sums_memory, 0);
    FillMemoryBytes(down_q8_memory, 0);
    FillMemoryBytes(down_scale_memory, 0);
    FillMemoryBytes(down_sum_low_memory, 0);
    FillMemoryBytes(down_sum_high_memory, 0);
    FillMemoryBytes(contributions_memory, 0);
    FillMemoryBytes(output_memory, 0);

    if (!args.dump_prepacked_dir.empty()) {
      const std::filesystem::path dump_dir(args.dump_prepacked_dir);
      std::filesystem::create_directories(dump_dir);
      constexpr std::size_t gateup_weight_bytes =
          kExpertCount * kGateUpSize * kHiddenSize / 2;
      constexpr std::size_t gateup_scale_count =
          kExpertCount * (kHiddenSize / 32) * kGateUpSize;
      constexpr std::size_t gateup_min_code_count =
          kExpertCount * kGateUpSize * (kHiddenSize / 256) * 8;
      constexpr std::size_t gateup_dmin_count =
          kExpertCount * kGateUpSize * (kHiddenSize / 256);
      WriteBinaryVector(dump_dir / "gateup-weights.bin",
                        ReadClBytes(queue, interleaved_gateup.weights,
                                    gateup_weight_bytes, "gate/up weights"));
      WriteBinaryVector(dump_dir / "gateup-scales.bin",
                        ReadClBytes(queue, interleaved_gateup.scales,
                                    gateup_scale_count * sizeof(float),
                                    "gate/up scales"));
      WriteBinaryVector(dump_dir / "gateup-min-codes.bin",
                        ReadClBytes(queue,
                                    interleaved_gateup.compact_min_codes,
                                    gateup_min_code_count,
                                    "gate/up compact min codes"));
      WriteBinaryVector(dump_dir / "gateup-dmins.bin",
                        ReadClBytes(queue, interleaved_gateup.compact_dmins,
                                    gateup_dmin_count * sizeof(float),
                                    "gate/up compact dmins"));
      WriteBinaryVector(dump_dir / "gateup-scale-codes.bin",
                        ReadClBytes(queue, interleaved_gateup.scale_codes,
                                    gateup_min_code_count,
                                    "gate/up scale codes"));
      WriteBinaryVector(dump_dir / "gateup-block-ds.bin",
                        ReadClBytes(queue, interleaved_gateup.block_ds,
                                    gateup_dmin_count * sizeof(float),
                                    "gate/up block ds"));

      constexpr std::size_t down_weight_bytes =
          kExpertCount * kDownOutputSize * kDownInputSize / 2;
      constexpr std::size_t down_scale_count =
          kExpertCount * (kDownInputSize / 32) * kDownOutputSize;
      constexpr std::size_t down_min_code_count =
          kExpertCount * kDownOutputSize * (kDownInputSize / 256) * 8;
      constexpr std::size_t down_dmin_count =
          kExpertCount * kDownOutputSize * (kDownInputSize / 256);
      WriteBinaryVector(dump_dir / "down-weights.bin",
          ReadDnnlMemory<std::uint8_t>(down.Weights(), down_weight_bytes,
                                       "down weights"));
      WriteBinaryVector(dump_dir / "down-scales.bin",
          ReadDnnlMemory<float>(down.Scales(), down_scale_count,
                                "down scales"));
      WriteBinaryVector(dump_dir / "down-min-codes.bin",
          ReadDnnlMemory<std::uint8_t>(down.CompactMinCodes(),
                                       down_min_code_count,
                                       "down compact min codes"));
      WriteBinaryVector(dump_dir / "down-dmins.bin",
          ReadDnnlMemory<float>(down.CompactDmins(), down_dmin_count,
                                "down compact dmins"));
      WriteBinaryVector(dump_dir / "down-scale-codes.bin",
                        down.ScaleCodes());
      WriteBinaryVector(dump_dir / "down-block-ds.bin", down.BlockDs());
      const std::string manifest =
          "{\"schema_version\":\"iq36-grouped-s8-u4-static-prepack-v2\","
          "\"experts\":256,\"gateup_output_width\":1024,"
          "\"down_output_width\":2048,"
          "\"router_schedule\":\"dynamic_runtime_input\"}\n";
      WriteBinaryFile(dump_dir / "manifest.json", manifest.data(),
                      manifest.size());
    }
    if (args.prepack_only) {
      char device_name[256] = {};
      CheckCl(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, nullptr),
              "clGetDeviceInfo prepack-only device name");
      std::cout << std::boolalpha << "{"
                << "\"active_experts\":" << metadata.active_experts << ","
                << "\"device_name\":\"" << JsonEscape(device_name) << "\","
                << "\"max_group_size\":" << metadata.max_group << ","
                << "\"native_binary_loaded\":" << native_binary_mode << ","
                << "\"prepack_only\":true,"
                << "\"resident_weight_bytes\":541065216}"
                << std::endl;
      ReleaseNativeResources();
      return 0;
    }

    std::string build_log;
    cl_program program = BuildGroupedProgram(context, device, &build_log);
    cl_kernel gather_kernel =
        CreateKernel(program, "grouped_gather_f16_sums32");
    cl_kernel q8_gather_kernel =
        CreateKernel(program, "grouped_quantize_tokens_q8_sums32");
    cl_kernel q8_group_gather_kernel =
        CreateKernel(program, "grouped_gather_quantized_q8");
    cl_kernel q8_down_quantize_kernel =
        CreateKernel(program, "grouped_quantize_f16_q8_sums32");
    cl_kernel swiglu_kernel =
        CreateKernel(program, "grouped_q4k_residual_swiglu_sums32");
    cl_kernel finalize_kernel =
        CreateKernel(program, "grouped_q4k_down_residual_weight");
    cl_kernel scatter_kernel =
        CreateKernel(program, "grouped_scatter_routed_output");
    const auto Mem = [](const dnnl::memory& memory) {
      return dnnl::ocl_interop::get_mem_object(memory);
    };
    const auto ExecuteGate = [&]() {
      if (native_binary_mode) {
        EnqueueIq36S8GateUp(queue, fused_gateup_kernel, interleaved_gateup,
                            Mem(grouped_q8_memory), Mem(input_scale_memory),
                            Mem(input_sum_low_memory),
                            Mem(input_sum_high_memory), Mem(swiglu),
                            grouped_binary_offsets, grouped_binary_dummy,
                            metadata.max_group);
      } else {
        gate.Execute(stream);
      }
    };
    const auto ExecuteUp = [&]() {
      if (native_binary_mode) {
        return;
      } else {
        up.Execute(stream);
      }
    };
    const auto ExecuteDown = [&]() {
      if (native_binary_mode) {
        EnqueueIq36S8Down(queue, fused_down_kernel, down,
                          Mem(down_q8_memory), Mem(down_scale_memory),
                          Mem(down_sum_low_memory), Mem(down_sum_high_memory),
                          Mem(contributions_memory), grouped_binary_offsets,
                          Mem(compact_weight_memory), grouped_binary_dummy,
                          metadata.max_group);
      } else {
        down.Execute(stream);
      }
    };
    const cl_uint rows = static_cast<cl_uint>(kAssignments);
    std::array<cl_mem, 4> gather_args = {
        Mem(input_memory), Mem(token_map_memory), Mem(grouped_input),
        Mem(input_sums_memory)};
    for (cl_uint index = 0; index < gather_args.size(); ++index) {
      CheckCl(clSetKernelArg(gather_kernel, index, sizeof(cl_mem),
                             &gather_args[index]),
              "clSetKernelArg grouped gather");
    }
    CheckCl(clSetKernelArg(gather_kernel, 4, sizeof(rows), &rows),
            "clSetKernelArg grouped gather rows");
    std::array<cl_mem, 6> q8_gather_args = {
        Mem(input_memory), Mem(token_q8_memory), Mem(token_scale_memory),
        Mem(token_sums_memory), Mem(token_sum_low_memory),
        Mem(token_sum_high_memory)};
    for (cl_uint index = 0; index < q8_gather_args.size(); ++index) {
      CheckCl(clSetKernelArg(q8_gather_kernel, index, sizeof(cl_mem),
                             &q8_gather_args[index]),
              "clSetKernelArg grouped Q8 gather");
    }
    const cl_uint token_rows = static_cast<cl_uint>(kTokenCount);
    CheckCl(clSetKernelArg(q8_gather_kernel, 6, sizeof(token_rows),
                           &token_rows),
            "clSetKernelArg token Q8 rows");
    std::array<cl_mem, 9> q8_group_gather_args = {
        Mem(token_map_memory), Mem(token_q8_memory), Mem(token_scale_memory),
        Mem(token_sum_low_memory), Mem(token_sum_high_memory),
        Mem(grouped_q8_memory), Mem(input_scale_memory),
        Mem(input_sum_low_memory), Mem(input_sum_high_memory)};
    for (cl_uint index = 0; index < q8_group_gather_args.size(); ++index) {
      CheckCl(clSetKernelArg(q8_group_gather_kernel, index, sizeof(cl_mem),
                             &q8_group_gather_args[index]),
              "clSetKernelArg grouped quantized Q8 gather");
    }
    CheckCl(clSetKernelArg(q8_group_gather_kernel, 9, sizeof(rows), &rows),
            "clSetKernelArg grouped quantized Q8 rows");
    std::array<cl_mem, 8> swiglu_args = {
        Mem(gate.Destination()), Mem(up.Destination()), Mem(input_sums_memory),
        Mem(gate.Minimums()), Mem(up.Minimums()), Mem(row_expert_memory),
        Mem(swiglu), Mem(down_sums_memory)};
    for (cl_uint index = 0; index < swiglu_args.size(); ++index) {
      CheckCl(clSetKernelArg(swiglu_kernel, index, sizeof(cl_mem),
                             &swiglu_args[index]),
              "clSetKernelArg grouped swiglu");
    }
    CheckCl(clSetKernelArg(swiglu_kernel, 8, sizeof(rows), &rows),
            "clSetKernelArg grouped swiglu rows");
    std::array<cl_mem, 6> q8_down_args = {
        Mem(swiglu), Mem(down_q8_memory), Mem(down_scale_memory),
        Mem(down_sums_memory), Mem(down_sum_low_memory),
        Mem(down_sum_high_memory)};
    for (cl_uint index = 0; index < q8_down_args.size(); ++index) {
      CheckCl(clSetKernelArg(q8_down_quantize_kernel, index, sizeof(cl_mem),
                             &q8_down_args[index]),
              "clSetKernelArg grouped down Q8 quantize");
    }
    CheckCl(clSetKernelArg(q8_down_quantize_kernel, 6, sizeof(rows), &rows),
            "clSetKernelArg grouped down Q8 rows");
    std::array<cl_mem, 6> finalize_args = {
        Mem(down.Destination()), Mem(down_sums_memory), Mem(down.Minimums()),
        Mem(row_expert_memory), Mem(compact_weight_memory),
        Mem(contributions_memory)};
    for (cl_uint index = 0; index < finalize_args.size(); ++index) {
      CheckCl(clSetKernelArg(finalize_kernel, index, sizeof(cl_mem),
                             &finalize_args[index]),
              "clSetKernelArg grouped finalize");
    }
    CheckCl(clSetKernelArg(finalize_kernel, 6, sizeof(rows), &rows),
            "clSetKernelArg grouped finalize rows");
    std::array<cl_mem, 3> scatter_args = {
        Mem(contributions_memory), Mem(inverse_map_memory), Mem(output_memory)};
    for (cl_uint index = 0; index < scatter_args.size(); ++index) {
      CheckCl(clSetKernelArg(scatter_kernel, index, sizeof(cl_mem),
                             &scatter_args[index]),
              "clSetKernelArg grouped scatter");
    }

    constexpr std::size_t local = 256;
    const std::size_t gather_global = kAssignments * 8 * local;
    const std::size_t token_quant_global = kTokenCount * 8 * local;
    const std::size_t quantized_gather_global = kAssignments * kHiddenSize;
    const std::size_t swiglu_global = kAssignments * 2 * local;
    const std::size_t finalize_global = kAssignments * 8 * local;
    const std::size_t scatter_global = kTokenCount * kHiddenSize;
    const auto EnqueueGather = [&]() {
      if (native_binary_mode) {
        CheckCl(clEnqueueNDRangeKernel(queue, q8_gather_kernel, 1, nullptr,
                                       &token_quant_global, &local, 0, nullptr,
                                       nullptr),
                "clEnqueueNDRangeKernel token Q8 quantize");
        CheckCl(clEnqueueNDRangeKernel(queue, q8_group_gather_kernel, 1,
                                       nullptr, &quantized_gather_global,
                                       &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel grouped quantized Q8 gather");
      } else {
        CheckCl(clEnqueueNDRangeKernel(queue, gather_kernel, 1, nullptr,
                                       &gather_global, &local, 0, nullptr,
                                       nullptr),
                "clEnqueueNDRangeKernel grouped gather");
      }
    };
    const auto EnqueueSwiGlu = [&]() {
      if (native_binary_mode) {
        CheckCl(clEnqueueNDRangeKernel(queue, q8_down_quantize_kernel, 1,
                                       nullptr, &swiglu_global, &local, 0,
                                       nullptr, nullptr),
                "clEnqueueNDRangeKernel grouped down Q8 quantize");
      } else {
        CheckCl(clEnqueueNDRangeKernel(queue, swiglu_kernel, 1, nullptr,
                                       &swiglu_global, &local, 0, nullptr,
                                       nullptr),
                "clEnqueueNDRangeKernel grouped swiglu");
      }
    };
    const auto EnqueueFinalize = [&]() {
      if (!native_binary_mode) {
        CheckCl(clEnqueueNDRangeKernel(queue, finalize_kernel, 1, nullptr,
                                       &finalize_global, &local, 0, nullptr,
                                       nullptr),
                "clEnqueueNDRangeKernel grouped finalize");
      }
    };
    const auto EnqueueScatter = [&]() {
      CheckCl(clEnqueueNDRangeKernel(queue, scatter_kernel, 1, nullptr,
                                     &scatter_global, &local, 0, nullptr,
                                     nullptr),
              "clEnqueueNDRangeKernel grouped scatter");
    };
    const auto Execute = [&]() {
      EnqueueGather();
      ExecuteGate();
      ExecuteUp();
      EnqueueSwiGlu();
      ExecuteDown();
      EnqueueFinalize();
      EnqueueScatter();
      CheckCl(clFinish(queue), "clFinish grouped Q4_K routed MoE");
    };

    for (int iteration = 0; iteration < args.warmup; ++iteration) Execute();
    std::array<double, 7> stage_us{};
    stage_us[0] = TimedStage(EnqueueGather, queue);
    stage_us[1] = TimedStage(ExecuteGate, queue);
    stage_us[2] = TimedStage(ExecuteUp, queue);
    stage_us[3] = TimedStage(EnqueueSwiGlu, queue);
    stage_us[4] = TimedStage(ExecuteDown, queue);
    stage_us[5] = TimedStage(EnqueueFinalize, queue);
    stage_us[6] = TimedStage(EnqueueScatter, queue);
    std::vector<double> samples_us;
    samples_us.reserve(args.repeat);
    for (int iteration = 0; iteration < args.repeat; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      Execute();
      samples_us.push_back(std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - begin)
                               .count());
    }

    const std::vector<float> swiglu_output =
        ReadHalfMemory(swiglu, kAssignments * kIntermediateSize);
    const CompareStats compare = Compare(swiglu_output, oracle, plan);
    const std::vector<float> weighted_output =
        ReadHalfMemory(contributions_memory, kAssignments * kHiddenSize);
    std::vector<float> weighted_oracle(kAssignments * kHiddenSize, 0.0f);
    for (std::size_t row = 0; row < kAssignments; ++row) {
      const std::size_t source =
          plan.bucket_token[row] * kSelectedExperts + plan.bucket_rank[row];
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        weighted_oracle[row * kHiddenSize + hidden] =
            down_oracle[source * kHiddenSize + hidden] * router_weights[source];
      }
    }
    const CompareStats weighted_compare =
        CompareFlat(weighted_output, weighted_oracle);
    const float* output_map = output_memory.map_data<float>();
    Require(output_map != nullptr, "could not map grouped routed output");
    std::vector<float> routed_output(
        output_map, output_map + kTokenCount * kHiddenSize);
    output_memory.unmap_data(const_cast<float*>(output_map));
    const CompareStats moe_compare = CompareFlat(routed_output, moe_oracle);

    std::vector<double> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    const double minimum_us = sorted.front();
    const double median_us = sorted[sorted.size() / 2];
    const double mean_us =
        std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
        samples_us.size();
    const bool implementations_pass =
        gate.Implementation().find("grouped_gemm:micro") != std::string::npos &&
        up.Implementation().find("grouped_gemm:micro") != std::string::npos &&
        down.Implementation().find("grouped_gemm:micro") != std::string::npos;
    const bool correctness_pass = ComparePass(compare) &&
        ComparePass(weighted_compare) && ComparePass(moe_compare);
    const bool performance_pass = minimum_us <= args.kernel_cap_us;
    const std::uint64_t active_codes = gate.Repack().active_code_count +
        up.Repack().active_code_count + down.Repack().active_code_count;
    const std::uint64_t resident_codes = gate.Repack().resident_code_count +
        up.Repack().resident_code_count + down.Repack().resident_code_count;
    const std::uint64_t repack_mismatches = gate.Repack().mismatch_count +
        up.Repack().mismatch_count + down.Repack().mismatch_count;
    const std::uint64_t resident_bytes = gate.ResidentBytes() +
        up.ResidentBytes() + down.ResidentBytes();
    const bool repack_pass =
        active_codes == 698351616ULL && resident_codes == 805306368ULL &&
        repack_mismatches == 0;
    const dnnl_version_t* version = dnnl_version();
    char device_name[256] = {};
    char driver_version[256] = {};
    CheckCl(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                            device_name, nullptr),
            "clGetDeviceInfo grouped device name");
    CheckCl(clGetDeviceInfo(device, CL_DRIVER_VERSION,
                            sizeof(driver_version), driver_version, nullptr),
            "clGetDeviceInfo grouped driver version");

    std::cout << std::boolalpha << std::setprecision(12) << "{";
    std::cout << "\"active_experts\":" << metadata.active_experts << ",";
    std::cout << "\"active_q4_code_count\":" << active_codes << ",";
    std::cout << "\"assignment_count\":" << kAssignments << ",";
    PrintCompare("compare", compare);
    std::cout << "\"correctness_pass\":" << correctness_pass << ",";
    std::cout << "\"device_name\":\"" << JsonEscape(device_name) << "\",";
    std::cout << "\"driver_version\":\"" << JsonEscape(driver_version)
              << "\",";
    std::cout << "\"exact_floating_residual\":true,";
    std::cout << "\"implementations\":[\""
              << JsonEscape(gate.Implementation()) << "\",\""
              << JsonEscape(up.Implementation()) << "\",\""
              << JsonEscape(down.Implementation()) << "\"],";
    std::cout << "\"implementations_pass\":" << implementations_pass << ",";
    std::cout << "\"kernel_cap_us\":" << args.kernel_cap_us << ",";
    std::cout << "\"native_binary_loaded\":" << native_binary_mode << ",";
    std::cout << "\"max_group_size\":" << metadata.max_group << ",";
    std::cout << "\"mean_us\":" << mean_us << ",";
    std::cout << "\"median_us\":" << median_us << ",";
    std::cout << "\"minimum_us\":" << minimum_us << ",";
    std::cout << "\"mode\":\"grouped_sparse_q4k_residual_routed_moe\",";
    PrintCompare("moe_compare", moe_compare);
    std::cout << "\"onednn_version\":{\"hash\":\""
              << JsonEscape(version->hash == nullptr ? "" : version->hash)
              << "\",\"major\":" << version->major
              << ",\"minor\":" << version->minor
              << ",\"patch\":" << version->patch << "},";
    std::cout << "\"performance_pass\":" << performance_pass << ",";
    std::cout << "\"repack_mismatch_count\":" << repack_mismatches << ",";
    std::cout << "\"repack_pass\":" << repack_pass << ",";
    std::cout << "\"resident_q4_code_count\":" << resident_codes << ",";
    std::cout << "\"resident_weight_bytes\":" << resident_bytes << ",";
    std::cout << "\"samples_us\":[";
    for (std::size_t index = 0; index < samples_us.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << samples_us[index];
    }
    std::cout << "],\"source_type\":\""
              << (native_binary_mode ? "s8" : "f16") << "\",";
    std::cout << "\"stage_us\":{\"gather\":" << stage_us[0]
              << ",\"gate\":" << stage_us[1]
              << ",\"up\":" << stage_us[2]
              << ",\"residual_swiglu\":" << stage_us[3]
              << ",\"down\":" << stage_us[4]
              << ",\"residual_weight\":" << stage_us[5]
              << ",\"scatter\":" << stage_us[6] << "},";
    PrintCompare("weighted_down_compare", weighted_compare);
    std::cout << "\"weight_group_size\":32}" << std::endl;

    clReleaseKernel(scatter_kernel);
    clReleaseKernel(finalize_kernel);
    clReleaseKernel(swiglu_kernel);
    clReleaseKernel(q8_down_quantize_kernel);
    clReleaseKernel(q8_group_gather_kernel);
    clReleaseKernel(q8_gather_kernel);
    clReleaseKernel(gather_kernel);
    clReleaseProgram(program);
    ReleaseNativeResources();
    return implementations_pass && repack_pass && correctness_pass &&
            performance_pass
        ? 0
        : 2;
  } catch (const dnnl::error& error) {
    std::cerr << "oneDNN status " << error.status << ": " << error.what()
              << '\n';
    return 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
