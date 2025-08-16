#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64 type.");
    ASSERT(out->dtype() == weight->dtype(), "Embedding: out and weight must have same dtype.");
    ASSERT(index->shape().size() == 1, "Embedding: index must be 1D tensor.");
    ASSERT(weight->shape().size() == 2, "Embedding: weight must be 2D tensor.");
    ASSERT(out->shape().size() == 2, "Embedding: out must be 2D tensor.");
    
    size_t batch_size = index->shape()[0];
    size_t embed_dim = weight->shape()[1];
    
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == embed_dim, 
           "Embedding: output shape mismatch.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), batch_size, embed_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), batch_size, embed_dim);
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
