#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val, up_val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            gate_val = llaisys::utils::cast<float>(gate[i]);
            up_val = llaisys::utils::cast<float>(up[i]);
        } else {
            gate_val = gate[i];
            up_val = up[i];
        }
        
        // SwiGLU: out = up * (gate / (1 + exp(-gate)))
        // This is equivalent to: out = up * sigmoid(gate)
        float sigmoid_gate = gate_val / (1.0f + std::exp(-gate_val));
        float result = up_val * sigmoid_gate;
        
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            out[i] = result;
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), 
                      reinterpret_cast<const float *>(gate),
                      reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), 
                      reinterpret_cast<const llaisys::bf16_t *>(gate),
                      reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), 
                      reinterpret_cast<const llaisys::fp16_t *>(gate),
                      reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu