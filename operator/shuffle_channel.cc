#include "./shuffle_channel-inl.h"

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include <cstring>


using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {

template <typename DType>
inline void Resize_cpu(DType *output, const DType *input, int group_row, int group_column, int len) {
  for (int i = 0; i < group_row; ++i) {
    for (int j = 0; j < group_column; ++j) {
      const DType* p_i = input + (i * group_column + j ) * len;
      DType* p_o = output + (j * group_row + i ) * len;

      std::memcpy(p_o, p_i, sizeof(DType)*len);
    }
  }

}

template <typename DType>
inline void ShuffleChannelForward(const Tensor<cpu, 4, DType> &out,
                                  const Tensor<cpu, 4, DType> &data,
                                  const uint32_t group) {
  const DType *bottom_data = data.dptr_;
  DType *top_data = out.dptr_;

  const int num = data.size(0);
  const int feature_map_size = data.size(1) * data.size(2) * data.size(3);
  const int sp_sz = data.size(3) * data.size(2);
  const int chs = data.size(1);

  int group_row = group;
  int group_column = int(chs / group_row);
  CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";
  int count = num * group_column * group_row;

  for (int n = 0; n < num; ++n) {
    Resize_cpu(top_data + n*feature_map_size, bottom_data + n*feature_map_size, group_row, group_column, sp_sz);
  }
}


template <typename DType>
inline void ShuffleChannelBackward(const Tensor<cpu, 4, DType> &in_grad,
                                   const Tensor<cpu, 4, DType> &out_grad,
                                   const Tensor<cpu, 4, DType> &data,
                                   const uint32_t group) {
  const DType* top_diff = out_grad.dptr_;
  DType* bottom_diff = in_grad.dptr_;

  const int num = data.size(0);
  const int feature_map_size = data.size(1) * data.size(2) * data.size(3);
  const int sp_sz = data.size(2) * data.size(3);
  const int chs = data.size(1);

  int group_row = int(chs / group);
  int group_column = group;

  for (int n = 0; n < num; ++n) {
    Resize_cpu(bottom_diff + n*feature_map_size, top_diff + n*feature_map_size, group_row, group_column, sp_sz);
  }

}


} // namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ShuffleChannelParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ShuffleChannelOp<cpu, DType>(param);
  });
  return op;
}

Operator *ShuffleChannelProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           	   std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ShuffleChannelParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_ShuffleChannel, ShuffleChannelProp)
.describe("shuffle channel.")
.add_argument("data", "NDArray-or-Symbol", "The input array to the channelshuffle operator, a 4D Feature maps")
.add_arguments(ShuffleChannelParam::__FIELDS__());

} // namespace op
} // namespace mxnet

