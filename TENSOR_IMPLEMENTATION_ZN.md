# LLAISYS 作业 #1：张量实现代码中文解释

本文档旨在逐行解释在 `src/tensor/tensor.cpp` 文件中为完成作业 #1 所实现的各个函数。

## 任务 1.1: `load()`

此函数负责将主机（CPU）内存中的数据加载到张量所处的内存空间中，这个空间可能位于 CPU 或其他设备（如 GPU）上。

```cpp
void Tensor::load(const void *src_) {
  // 步骤1: 获取当前设备的运行时上下文
  // 这确保我们使用正确的设备上下文进行内存操作
  core::context().setDevice(this->deviceType(), this->deviceId());

  // 步骤2: 计算需要复制的数据大小
  // numel() 返回张量中元素的总数
  // elementSize() 返回每个元素的字节大小
  size_t copy_size = this->numel() * this->elementSize();

  // 步骤3: 确定内存复制的类型
  // 根据张量所在的设备类型选择合适的内存复制方向
  llaisysMemcpyKind_t copy_kind;
  if (this->deviceType() == LLAISYS_DEVICE_CPU) {
    // 如果张量在CPU上，则是主机到主机的复制
    copy_kind = LLAISYS_MEMCPY_H2H;
  } else {
    // 如果张量在其他设备（如GPU）上，则是主机到设备的复制
    copy_kind = LLAISYS_MEMCPY_H2D;
  }

  // 步骤4: 执行同步内存复制
  // src_: 源数据指针（主机内存）
  // this->data(): 目标数据指针（张量的内存位置）
  // copy_size: 复制的字节数
  // copy_kind: 内存复制的类型
  core::context().runtime().api()->memcpy_sync(
      this->data(), // 目标地址：张量的数据指针
      src_,         // 源地址：传入的主机数据指针
      copy_size,    // 复制大小：总字节数
      copy_kind     // 复制类型：H2H或H2D
  );
}
```

## 任务 1.2: `isContiguous()`

此函数用于检查张量在内存中是否是连续的。一个张量是连续的，意味着它的元素在内存中是按顺序紧密排列的，没有任何间隙。

```cpp
bool Tensor::isContiguous() const {
  // 步骤1: 处理0维张量
  // 0维张量（标量）总是连续的
  if (this->ndim() == 0) {
    return true;
  }

  // 步骤2: 初始化预期的步长
  // 从最内层的维度开始，步长应为1
  ptrdiff_t stride = 1;

  // 步骤3: 从后向前遍历所有维度
  // C/C++中的多维数组是按行主序存储的，所以我们从最后一个维度开始检查
  for (long i = this->ndim() - 1; i >= 0; i--) {
    // 步骤3.1: 处理维度大小为0的情况
    // 如果任何一个维度的大小为0，那么这个张量可以被认为是连续的
    if (this->shape()[i] == 0) {
        return true;
    }
    
    // 步骤3.2: 检查当前维度的步长
    // 如果实际步长与预期步长不符，则张量不连续
    if (this->strides()[i] != stride) {
      return false;
    }

    // 步骤3.3: 更新下一个维度的预期步长
    // 预期步长应乘以当前维度的形状大小
    stride *= this->shape()[i];
  }

  // 步骤4: 所有维度检查通过
  // 如果循环完成而没有返回false，说明张量是连续的
  return true;
}
```

## 任务 1.3: `view()`

此函数创建一个新的张量，该张量是原始张量数据的一个新“视图”，具有不同的形状，但共享相同的底层存储。这是一种高效的操作，因为它不涉及数据复制。

```cpp
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
  // 步骤1: 检查张量是否连续
  // `view` 操作通常只支持连续张量，以确保新的步长计算是有效的
  if (!this->isContiguous()) {
    EXCEPTION("view is only supported for contiguous tensors");
  }

  // 步骤2: 验证新旧形状的元素总数是否一致
  size_t new_numel = 1;
  for (auto s : shape) {
    new_numel *= s;
  }
  if (new_numel != this->numel()) {
    EXCEPTION("shape mismatch");
  }

  // 步骤3: 为新形状计算新的步长
  // 这与创建连续张量时的步长计算逻辑相同
  size_t ndim_ = shape.size();
  std::vector<ptrdiff_t> strides(ndim_);
  ptrdiff_t stride = 1;
  for (size_t i = 1; i <= ndim_; i++) {
    strides[ndim_ - i] = stride;
    stride *= shape[ndim_ - i];
  }

  // 步骤4: 创建并返回新的张量对象
  // 新张量共享原始张量的存储（_storage）和偏移量（_offset）
  TensorMeta new_meta{this->dtype(), shape, strides};
  return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

## 任务 1.4: `permute()`

此函数通过重新排列维度来创建一个新的张量视图。例如，可以将一个形状为 `(2, 3, 4)` 的张量转置为 `(4, 3, 2)`。此操作也不涉及数据复制。

```cpp
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
  // 步骤1: 检查维度顺序向量的大小是否正确
  if (order.size() != this->ndim()) {
    EXCEPTION("permute order must have the same size as tensor ndim");
  }

  // 步骤2: 根据 `order` 计算新的形状和步长
  std::vector<size_t> new_shape(this->ndim());
  std::vector<ptrdiff_t> new_strides(this->ndim());
  for (size_t i = 0; i < this->ndim(); i++) {
    // 新形状的第 i 维是原形状的第 order[i] 维
    new_shape[i] = this->shape()[order[i]];
    // 新步长的第 i 维是原步长的第 order[i] 维
    new_strides[i] = this->strides()[order[i]];
  }

  // 步骤3: 创建并返回新的张量对象
  // 新张量共享原始张量的存储和偏移量
  TensorMeta new_meta{this->dtype(), new_shape, new_strides};
  return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```

## 任务 1.5: `slice()`

此函数沿着指定的维度对张量进行切片，返回一个新的张量视图，该视图表示原始张量的一个子集。这同样是一个不涉及数据复制的元数据操作。

```cpp
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
  // 步骤1: 验证输入参数的有效性
  if (dim >= this->ndim()) {
    EXCEPTION("dimension out of range");
  }
  if (start >= end || end > this->shape()[dim]) {
    EXCEPTION("slice indices out of range");
  }

  // 步骤2: 计算新张量的形状
  // 只有被切片的维度大小会改变
  std::vector<size_t> new_shape = this->shape();
  new_shape[dim] = end - start;

  // 步骤3: 计算新张量在共享存储中的偏移量
  // 偏移量需要增加 `start` 个元素在 `dim` 维度上所占的字节数
  size_t new_offset = _offset + start * this->strides()[dim] * this->elementSize();

  // 步骤4: 创建并返回新的张量对象
  // 新张量共享存储，但具有新的形状、步长和偏移量
  TensorMeta new_meta{this->dtype(), new_shape, this->strides()};
  return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}
```
