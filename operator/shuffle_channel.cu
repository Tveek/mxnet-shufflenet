#include "./shuffle_channel-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

#define SHUFFLECHANNEL_CUDA_CHECK(condition) \
    /* Code block avoids redefinition of cudaError_t error */ \
    do { \
      cudaError_t error = condition; \
      CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    }while (0)
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace mshadow {
namespace cuda {
template <typename DType>
__global__ void ShuffleChannelKernel(const int count, const int feature_map_size, DType *output, 
        const DType *input, int group_row, int group_column, int len) {
    CUDA_KERNEL_LOOP(index, count) {
        const int n = index / group_row / group_column / len;
        const int i = (index / group_column / len) % group_row;
        const int j = index / len % group_column;
        const int k = index - (n * feature_map_size + (i * group_column + j) * len);
        DType* p_o = output + n * feature_map_size + (j * group_row + i) * len;
        p_o[k] = input[index];
    }
}


template <typename DType>
inline void ShuffleChannelKernelForward(const Tensor<gpu, 4, DType> &out,
                                  const Tensor<gpu, 4, DType> &data,
                                  const uint32_t group) {
    const DType *bottom_data = data.dptr_;
    DType *top_data = out.dptr_;

    const int num = data.size(0);
    const int feature_map_size = data.size(1) * data.size(2) * data.size(3);
    const int sp_sz = data.size(2) * data.size(3);
    const int chs = data.size(1);

    int group_row = group;
    int group_column = int(chs / group_row);
    CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";

    int count = num * group_column * group_row * sp_sz;
    const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "ShuffleChannelForward");
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    ShuffleChannelKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, feature_map_size, top_data, 
            bottom_data, group_row, group_column, sp_sz);
    SHUFFLECHANNEL_CUDA_CHECK(cudaPeekAtLastError());
}

template <typename DType>
inline void ShuffleChannelKernelBackward(const Tensor<gpu, 4, DType> &in_grad,
                                   const Tensor<gpu, 4, DType> &out_grad,
                                   const Tensor<gpu, 4, DType> &data,
                                   const uint32_t group) {
    const DType* top_diff = out_grad.dptr_;
    DType* bottom_diff = in_grad.dptr_;

    const int num = data.size(0);
    const int feature_map_size = data.size(1) * data.size(2) * data.size(3);
    const int sp_sz = data.size(2) * data.size(3);
    const int chs = data.size(1);

    int group_row = int(chs / group);
    int group_column = group;
    int count = num * group_column * group_row * sp_sz;
    const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "ShuffleChannelBackward");
    cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
    ShuffleChannelKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, feature_map_size, 
            bottom_diff, top_diff, group_row, group_column, sp_sz);
    SHUFFLECHANNEL_CUDA_CHECK(cudaPeekAtLastError());

}
} // namespace cuda


template <typename DType>
inline void ShuffleChannelForward(const Tensor<gpu, 4, DType> &out,
                                  const Tensor<gpu, 4, DType> &data,
                                  const uint32_t group) {
    cuda::ShuffleChannelKernelForward(out, data, group);
}

template <typename DType>
inline void ShuffleChannelBackward(const Tensor<gpu, 4, DType> &in_grad,
                                   const Tensor<gpu, 4, DType> &out_grad, 
                                   const Tensor<gpu, 4, DType> &data,
                                   const uint32_t group) {
    cuda::ShuffleChannelKernelBackward(in_grad, out_grad, data, group);
}

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ShuffleChannelParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ShuffleChannelOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet


