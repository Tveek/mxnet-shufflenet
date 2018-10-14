#ifndef MXNET_OPERATOR_SHUFFLE_CHANNEL_INL_H_
#define MXNET_OPERATOR_SHUFFLE_CHANNEL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"


namespace mxnet {
namespace op {

namespace shuffle_channel {
enum ShuffleChannelOpInputs {kData};
enum ShuffleChannelOpOutput {kOut};
} // shuffle_channel

struct ShuffleChannelParam : public dmlc::Parameter<ShuffleChannelParam> {
  uint32_t group;
  DMLC_DECLARE_PARAMETER(ShuffleChannelParam) {
    DMLC_DECLARE_FIELD(group).describe("The number of group");
  }

};

template<typename xpu, typename DType>
class ShuffleChannelOp : public Operator {
  public:
    explicit ShuffleChannelOp(ShuffleChannelParam p) {
      this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {

      using namespace mshadow;
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 1);

      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 4, DType> data = in_data[shuffle_channel::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = out_data[shuffle_channel::kOut].get<xpu, 4, DType>(s);

      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      out = -FLT_MAX;
      ShuffleChannelForward(out, data, param_.group);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      CHECK_EQ(in_grad.size(), 1);
      CHECK_EQ(out_grad.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 4, DType> grad_out = out_grad[shuffle_channel::kOut].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> data = in_data[shuffle_channel::kData].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> grad_in = in_grad[shuffle_channel::kData].get<xpu, 4, DType>(s);

      CHECK_EQ(grad_out.CheckContiguous(), true);
      CHECK_EQ(grad_in.CheckContiguous(), true);

      grad_in = 0.0f;
      ShuffleChannelBackward(grad_in, grad_out, data, param_.group);
    }

  private:
    ShuffleChannelParam param_;

};

template<typename xpu>
Operator* CreateOp(ShuffleChannelParam param, int dtype);

#if DMLC_USE_CXX11
class ShuffleChannelProp : public OperatorProperty {
  public:
    std::vector<std::string> ListArguments() const override {
      return {"data"};
    }

    std::vector<std::string> ListOutputs() const override {
      return {"output"};
    }

    int NumOutputs() const override {
      return 1;
    }

    int NumVisibleOutputs() const override {
      return 1;
    }

    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
      param_.Init(kwargs);
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      using namespace mshadow;
      CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
      TShape dshape = in_shape->at(shuffle_channel::kData);

      out_shape->clear();
      out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
      return true;
    }

    bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      CHECK_EQ(in_type->size(), 1U);
      int dtype = (*in_type)[0];
      CHECK_NE(dtype, -1) << "Input must have specified type";

      out_type->clear();
      out_type->push_back(dtype);
      return true;
    }

    OperatorProperty* Copy() const override {
      ShuffleChannelProp* shuffle_channel_sym = new ShuffleChannelProp();
      shuffle_channel_sym->param_ = this->param_;
      return shuffle_channel_sym;
    }

  std::string TypeString() const override {
    return "_contrib_ShuffleChannel";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[shuffle_channel::kOut], in_data[shuffle_channel::kData]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

  private:
    ShuffleChannelParam param_;

};


#endif // DMLC_USE_CXX11


} // namespace op
} // namespace mxnet



#endif
