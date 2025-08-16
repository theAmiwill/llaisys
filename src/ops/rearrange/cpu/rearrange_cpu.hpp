#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, 
               const size_t *out_shape, const size_t *out_strides,
               const size_t *in_shape, const size_t *in_strides,
               size_t ndim, size_t element_size);
}