#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, 
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // RoPE implementation
    // Input shape: [seq_len, n_heads, head_dim]
    // pos_ids shape: [seq_len]
    
    size_t half_dim = head_dim / 2;
    
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t d = 0; d < half_dim; d++) {
                // Calculate frequency
                float freq = pos / std::pow(theta, 2.0f * d / head_dim);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                
                // Get input values
                size_t idx_a = s * n_heads * head_dim + h * head_dim + d;
                size_t idx_b = s * n_heads * head_dim + h * head_dim + d + half_dim;
                
                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[idx_a]);
                    b_val = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a_val = in[idx_a];
                    b_val = in[idx_b];
                }
                
                // Apply RoPE rotation
                float new_a = a_val * cos_val - b_val * sin_val;
                float new_b = b_val * cos_val + a_val * sin_val;
                
                // Store results
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(new_a);
                    out[idx_b] = llaisys::utils::cast<T>(new_b);
                } else {
                    out[idx_a] = new_a;
                    out[idx_b] = new_b;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu