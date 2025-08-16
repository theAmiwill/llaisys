#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMS Norm: all tensors must be contiguous.");
    
    ASSERT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(), 
           "RMS Norm: all tensors must have same dtype.");
    
    ASSERT(in->shape().size() == 2 && out->shape().size() == 2 && weight->shape().size() == 1,
           "RMS Norm: in and out must be 2D, weight must be 1D.");
    
    ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == out->shape()[1],
           "RMS Norm: in and out must have same shape.");
    
    ASSERT(weight->shape()[0] == in->shape()[1],
           "RMS Norm: weight size must match input hidden dimension.");
    
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                            out->dtype(), batch_size, hidden_size, eps);
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
