#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    ASSERT(out->isContiguous(), "Rearrange: output tensor must be contiguous.");
    
    // 获取张量信息
    size_t ndim = in->shape().size();
    size_t element_size;
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        element_size = sizeof(float);
        break;
    case LLAISYS_DTYPE_F16:
        element_size = sizeof(uint16_t);
        break;
    case LLAISYS_DTYPE_BF16:
        element_size = sizeof(uint16_t);
        break;
    case LLAISYS_DTYPE_I64:
        element_size = sizeof(int64_t);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
    
    // 转换shape和strides为size_t类型
    std::vector<size_t> out_shape_vec(out->shape().begin(), out->shape().end());
    std::vector<size_t> out_strides_vec(out->strides().begin(), out->strides().end());
    std::vector<size_t> in_shape_vec(in->shape().begin(), in->shape().end());
    std::vector<size_t> in_strides_vec(in->strides().begin(), in->strides().end());
    
    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(),
                             out_shape_vec.data(), out_strides_vec.data(),
                             in_shape_vec.data(), in_strides_vec.data(),
                             ndim, element_size);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(),
                             out_shape_vec.data(), out_strides_vec.data(),
                             in_shape_vec.data(), in_strides_vec.data(),
                             ndim, element_size);
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
