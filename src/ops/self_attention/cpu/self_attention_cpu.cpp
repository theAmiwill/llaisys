#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstddef>
#include <type_traits>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seq_len, size_t kv_len, size_t n_heads, 
                     size_t n_kv_heads, size_t head_dim, float scale) {
    // Q: [seq_len, n_heads, head_dim]
    // K: [kv_len, n_kv_heads, head_dim] 
    // V: [kv_len, n_kv_heads, head_dim]
    // Output: [seq_len, n_heads, head_dim]
    
    size_t head_group_size = n_heads / n_kv_heads;
    
    for (size_t q_pos = 0; q_pos < seq_len; q_pos++) {
        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_head = h / head_group_size;
            
            // Step 1: Compute attention scores A = Q * K^T * scale
            std::vector<float> scores(kv_len);
            
            for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = q_pos * n_heads * head_dim + h * head_dim + d;
                    size_t k_idx = k_pos * n_kv_heads * head_dim + kv_head * head_dim + d;
                    
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = q[q_idx];
                        k_val = k[k_idx];
                    }
                    score += q_val * k_val;
                }
                scores[k_pos] = score * scale;
            }
            
            // Step 2: Apply causal mask and compute softmax
            // For causal attention with KV cache: position i can attend to positions 0 to min(i + (kv_len - seq_len), kv_len - 1)
            size_t max_attend_pos = q_pos + (kv_len - seq_len);
            if (max_attend_pos >= kv_len) {
                max_attend_pos = kv_len - 1;
            }
            
            // Apply causal mask by setting invalid positions to -inf
            for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                if (k_pos > max_attend_pos) {
                    scores[k_pos] = -std::numeric_limits<float>::infinity();
                }
            }
            
            // Find max for numerical stability
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                if (scores[k_pos] != -std::numeric_limits<float>::infinity()) {
                    max_score = std::max(max_score, scores[k_pos]);
                }
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                if (scores[k_pos] != -std::numeric_limits<float>::infinity()) {
                    scores[k_pos] = std::exp(scores[k_pos] - max_score);
                    sum_exp += scores[k_pos];
                } else {
                    scores[k_pos] = 0.0f;
                }
            }
            
            // Normalize
            if (sum_exp > 0.0f) {
                for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                    scores[k_pos] /= sum_exp;
                }
            }
            
            // Step 3: Compute weighted sum of values Y = softmax(A) * V
            for (size_t d = 0; d < head_dim; d++) {
                float result = 0.0f;
                for (size_t k_pos = 0; k_pos < kv_len; k_pos++) {
                    size_t v_idx = k_pos * n_kv_heads * head_dim + kv_head * head_dim + d;
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = v[v_idx];
                    }
                    result += scores[k_pos] * v_val;
                }
                
                size_t out_idx = q_pos * n_heads * head_dim + h * head_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(result);
                } else {
                    attn_val[out_idx] = result;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seq_len, size_t kv_len, size_t n_heads, 
                    size_t n_kv_heads, size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), 
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), 
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), 
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu