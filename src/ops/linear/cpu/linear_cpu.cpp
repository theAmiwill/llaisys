#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features) {
    // Y = X * W^T + b
    // X: [batch_size, in_features]
    // W: [out_features, in_features] 
    // Y: [batch_size, out_features]
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            
            // Compute dot product of input row with weight row
            for (size_t i = 0; i < in_features; i++) {
                float x_val, w_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                } else {
                    x_val = in[b * in_features + i];
                    w_val = weight[o * in_features + i];
                }
                sum += x_val * w_val;
            }
            
            // Add bias if provided
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[o]);
                } else {
                    sum += bias[o];
                }
            }
            
            // Store result
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                out[b * out_features + o] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                      reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                      reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias),
                      batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                      reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias),
                      batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu