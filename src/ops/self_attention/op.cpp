#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "Self Attention: all tensors must be contiguous.");
    
    ASSERT(attn_val->dtype() == q->dtype() && q->dtype() == k->dtype() && k->dtype() == v->dtype(), 
           "Self Attention: all tensors must have same dtype.");
    
    ASSERT(q->shape().size() == 3 && k->shape().size() == 3 && v->shape().size() == 3 && attn_val->shape().size() == 3,
           "Self Attention: all tensors must be 3D.");
    
    size_t seq_len = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t kv_len = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    
    ASSERT(k->shape()[2] == head_dim && v->shape()[2] == head_dim, 
           "Self Attention: head dimensions must match.");
    ASSERT(v->shape()[0] == kv_len && v->shape()[1] == n_kv_heads,
           "Self Attention: k and v must have same sequence length and head count.");
    ASSERT(attn_val->shape()[0] == seq_len && attn_val->shape()[1] == n_heads && attn_val->shape()[2] == head_dim,
           "Self Attention: output shape mismatch.");
    ASSERT(n_heads % n_kv_heads == 0, "Self Attention: n_heads must be divisible by n_kv_heads.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), seq_len, kv_len, n_heads, n_kv_heads, head_dim, scale);
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
