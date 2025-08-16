#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops::cpu {

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, 
               const size_t *out_shape, const size_t *out_strides,
               const size_t *in_shape, const size_t *in_strides,
               size_t ndim, size_t element_size) {
    
    // 计算总元素数
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= out_shape[i];
    }
    
    // 对每个元素进行重排
    for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
        // 将线性索引转换为多维索引
        std::vector<size_t> indices(ndim);
        size_t temp_idx = linear_idx;
        for (int d = ndim - 1; d >= 0; d--) {
            indices[d] = temp_idx % out_shape[d];
            temp_idx /= out_shape[d];
        }
        
        // 计算输出偏移
        size_t out_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            out_offset += indices[d] * out_strides[d] * element_size;
        }
        
        // 计算输入偏移
        size_t in_offset = 0;
        for (size_t d = 0; d < ndim; d++) {
            in_offset += indices[d] * in_strides[d] * element_size;
        }
        
        // 复制数据
        std::memcpy(out + out_offset, in + in_offset, element_size);
    }
}

} // namespace llaisys::ops::cpu
