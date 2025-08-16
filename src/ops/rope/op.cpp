#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: all tensors must be contiguous.");
    
    ASSERT(out->dtype() == in->dtype(), "RoPE: out and in must have same dtype.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64 type.");
    
    ASSERT(in->shape().size() == 3 && out->shape().size() == 3 && pos_ids->shape().size() == 1,
           "RoPE: in and out must be 3D, pos_ids must be 1D.");
    
    ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == out->shape()[1] && in->shape()[2] == out->shape()[2],
           "RoPE: in and out must have same shape.");
    
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must match sequence length.");
    ASSERT(in->shape()[2] % 2 == 0, "RoPE: head dimension must be even.");
    
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                        out->dtype(), seq_len, n_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
