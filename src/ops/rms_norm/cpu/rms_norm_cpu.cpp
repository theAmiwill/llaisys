#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch_size, size_t hidden_size, float eps) {
    for (size_t b = 0; b < batch_size; b++) {
        const T *input_row = in + b * hidden_size;
        T *output_row = out + b * hidden_size;
        
        // Compute sum of squares
        float sum_squares = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(input_row[i]);
            } else {
                val = input_row[i];
            }
            sum_squares += val * val;
        }
        
        // Compute RMS normalization factor
        float mean_square = sum_squares / hidden_size;
        float rms_norm_factor = 1.0f / std::sqrt(mean_square + eps);
        
        // Apply normalization and weight
        for (size_t i = 0; i < hidden_size; i++) {
            float input_val, weight_val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                input_val = llaisys::utils::cast<float>(input_row[i]);
                weight_val = llaisys::utils::cast<float>(weight[i]);
            } else {
                input_val = input_row[i];
                weight_val = weight[i];
            }
            
            float normalized = input_val * rms_norm_factor;
            float result = normalized * weight_val;
            
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                output_row[i] = llaisys::utils::cast<T>(result);
            } else {
                output_row[i] = result;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
              llaisysDataType_t type, size_t batch_size, size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight), batch_size, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu