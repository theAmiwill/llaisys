#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: out, in, weight tensors must be contiguous.");
    if (bias) ASSERT(bias->isContiguous(), "Linear: bias tensor must be contiguous.");
    
    ASSERT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(), 
           "Linear: out, in, weight must have same dtype.");
    if (bias) ASSERT(bias->dtype() == out->dtype(), "Linear: bias must have same dtype as out.");
    
    ASSERT(in->shape().size() == 2 && weight->shape().size() == 2 && out->shape().size() == 2,
           "Linear: all tensors must be 2D.");
    if (bias) ASSERT(bias->shape().size() == 1, "Linear: bias must be 1D.");
    
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    ASSERT(weight->shape()[1] == in_features, "Linear: weight shape mismatch.");
    ASSERT(out->shape()[0] == batch_size && out->shape()[1] == out_features, 
           "Linear: output shape mismatch.");
    if (bias) ASSERT(bias->shape()[0] == out_features, "Linear: bias shape mismatch.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr, out->dtype(), 
                          batch_size, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr, out->dtype(), 
                          batch_size, in_features, out_features);
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
