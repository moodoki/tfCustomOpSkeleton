#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("MyAdd")
    .Attr("T: {float, double}")
    .Input("aa: T")
    .Input("bb: T")
    .Output("added: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename T>
class MyAddOp : public OpKernel {
 public:
  explicit MyAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor0 = context->input(0);
    const Tensor& input_tensor1 = context->input(1);
    auto input0 = input_tensor0.flat<T>();
    auto input1 = input_tensor1.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor0.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<T>();

    // Set all but the first element of the output tensor to 0.
    const int N = input0.size();

    #pragma omp simd
    for (int i = 0; i < N; i++) {
      output_flat(i) = input0(i) + input1(i);
    }

  }
};

REGISTER_KERNEL_BUILDER(
        Name("MyAdd")
        .Device(DEVICE_CPU) 
        .TypeConstraint<float>("T"),
        MyAddOp<float>);

REGISTER_KERNEL_BUILDER(
        Name("MyAdd")
        .Device(DEVICE_CPU) 
        .TypeConstraint<double>("T"),
        MyAddOp<double>);

//Status MyAddGrad(const Scope& scope, const Operation& op,
//                 const std::vector<Output>& grad_inputs,
//                 std::Vector<Output>* grad_outputs){
//    //y = x0 + x1
//    //dy/dx0 = x1
//    //dy/dx1 = x0
//    grad_outputs->push_back(Mul(scope, grad_inputs[0], x0))
//    grad_outputs->push_back(Mul(scope, grad_inputs[0], x1))
//
//    return scope.status();
//}
//REGISTER_GRADIENT_OP("MyAdd", MyAddGrad);
