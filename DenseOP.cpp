#include "DenseOP.h"
#include "MathFunctions.h"
#include "Tools.h"

namespace MiniDL {
    DenseOP::DenseOP() {

    };
    DenseOP::~DenseOP() {

    };

    //setting
    void DenseOP::setup(const int input_dim, const int output_dim, const bool with_bias) {
        // weight shape: (input_dim, output_dim)
        weights.reshape(Shape({ input_dim, output_dim }));
        if (get_phase() == Phase::Train) {
            weights_grad.reshape(weights.get_shape());
        }

        // bias shape: (output_dim)
        if (with_bias) {
            bias.reshape(Shape({ output_dim }));
            if (get_phase() == Phase::Train) {
                bias_grad.reshape(bias.get_shape());
            }
        }
        else {
            bias.clear();
        }
        //init weights and bias
        uniform_filler(weights.get_data(), -1.0f, 1.0f);
        constant_filler(bias.get_data(), 0.0f);

        //init grads
        if (get_phase() == Phase::Train) {
            constant_filler(weights_grad.get_data(), 0.0f);
            constant_filler(bias_grad.get_data(), 0.0f);
        }
    };

    const Data& DenseOP::get_weights() const {
        return weights;
    }
    const Data& DenseOP::get_bias() const {
        return bias;
    }

    //get all weights of op
    std::vector<Data*> DenseOP::get_all_weights(){
        std::vector<Data*> all_weights;
        all_weights.push_back(&weights);
        all_weights.push_back(&bias);
        return all_weights;
    }
    //get all grads of op
    std::vector<Data*> DenseOP::get_all_grads(){
        std::vector<Data*> all_grads;
        all_grads.push_back(&weights_grad);
        all_grads.push_back(&bias_grad);
        return all_grads;
    }

    //forward of op
    // y = w * x + b
    void DenseOP::forward(const std::vector<Data*>& inputs,
        std::vector<Data*>& outputs) {
        do_assert(inputs.size() == outputs.size(), "size of inputs is not equals with outputs");

        for (size_t i = 0; i < inputs.size(); i++) {
            const Data& input = *inputs[i];
            Data& output = *outputs[i];

            const Shape& input_shape = input.get_shape();
            const Shape& output_shape = output.get_shape();
            const Shape& weights_shape = weights.get_shape();
            const Shape& bias_shape = bias.get_shape();

            const int batch_size = input_shape.get_dim(0);
            const int input_dim = input_shape.get_dim(1);
            const int output_dim = output_shape.get_dim(1);

            //shape of input: N * input_dim
            //shape of output: N * output_dim
            //shape of weights: input_dim * output_dim
            //shape of bias: output_dim
            do_assert(input_shape.get_dims() == output_shape.get_dims() &&
                input_shape.get_dim(0) == output_shape.get_dim(0) &&
                input_dim == weights_shape.get_dim(0) &&
                output_dim == weights_shape.get_dim(1) &&
                (bias.is_empty() ? true : output_dim == bias_shape.get_dim(0)),
                "dim of input or output is invalidate");

            // y = w * x + b
            const DataType& input_mems = input.get_data();
            DataType& output_mems = output.get_data();
            const DataType& weights_mems = weights.get_data();
            const DataType& bias_mems = bias.get_data();
            for (size_t i = 0; i < batch_size; i++) {
                // output_dim
                // [y0, y1 ... yL] ... [y0, y1 ... yL]
                for (size_t j = 0; j < output_dim; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < input_dim; k++) {
                        sum += weights_mems[k * output_dim + j] * input_mems[i * input_dim + k];
                    }
                    if (!bias.is_empty()) {
                        sum += bias_mems[j];
                    }
                    output_mems[i * output_dim + j] = sum;
                }
            }

        }


    };
    //backward of op
    //forward: y = w * x + b
    void DenseOP::backward(const std::vector<Data*>& inputs,
        std::vector<Data*>& prev_diffs,
        const std::vector<Data*>& next_diffs,
        const std::vector<Data*>& outputs) {
        // forward y = w * x + b
        // backward dy/dx = w, dy/dw = x, dy/db = 1
        do_assert(inputs.size() == outputs.size(), "size of inputs must be equals with outputs");
        do_assert(prev_diffs.size() == inputs.size(), "size of prev_diffs must be equals with inputs");
        do_assert(next_diffs.size() == outputs.size(), "size of next_diffs must be equals with outputs");
        do_assert(inputs.size() > 0, "size of inputs must bigger than zero");

        // 
        for (size_t i = 0; i < prev_diffs.size(); i++) {
            Data& input = *inputs[i];
            Data& prev_diff = *prev_diffs[i];
            const Data& next_diff = *next_diffs[i];
            const Data& output = *outputs[i];

            const Shape& input_shape = input.get_shape();
            const Shape& output_shape = output.get_shape();
            const Shape& weights_shape = weights.get_shape();
            const Shape& bias_shape = bias.get_shape();
            const Shape& weights_grad_shape = weights_grad.get_shape();
            const Shape& bias_grad_shape = bias_grad.get_shape();
            const Shape& prev_diff_shape = prev_diff.get_shape();
            const Shape& next_diff_shape = next_diff.get_shape();

            const int batch_size = input_shape.get_dim(0);
            const int input_dim = input_shape.get_dim(1);
            const int diff_dim = input_dim;
            const int output_dim = output_shape.get_dim(1);

            //shape of input: N * input_dim
            //shape of output: N * output_dim
            //shape of weights: input_dim * output_dim
            //shape of bias: output_dim
            do_assert(input_shape.get_dims() == output_shape.get_dims() &&
                input_shape.get_dim(0) == output_shape.get_dim(0) &&
                input_dim == weights_shape.get_dim(0) &&
                output_dim == weights_shape.get_dim(1) &&
                weights_grad_shape == weights_shape &&
                bias_grad_shape == bias_shape &&
                prev_diff_shape == input_shape &&
                next_diff_shape == output_shape &&
                (bias.is_empty() ? true : output_dim == bias_shape.get_dim(0)),
                "dim of input or output is invalidate");

            const DataType& input_mems = input.get_data();
            DataType& prev_diff_mems = prev_diff.get_data();
            const DataType& next_diff_mems = next_diff.get_data();
            const DataType& output_mems = output.get_data();
            const DataType& weights_mems = weights.get_data();
            const DataType& bias_mems = bias.get_data();
            DataType& weights_grad_mems = weights_grad.get_data();
            DataType& bias_grad_mems = bias_grad.get_data();

            //dy/dx = w --> prev_diff
            prev_diff.fill(0.0f); // 每次计算前，梯度归0
            for (size_t i = 0; i < batch_size; i++) {
                // const float* x_frame = &input_mems[0] + i * input_dim;
                for (size_t j = 0; j < input_dim; j++) {
                    for (size_t k = 0; k < output_dim; k++) {
                        const size_t x_idx = i * input_dim + j;
                        const float x = input_mems[x_idx];
                        const size_t y_idx = i * output_dim + k;
                        const float y = output_mems[y_idx];
                        const float w = weights_mems[j * output_dim + k];

                        const float next_diff_val = next_diff_mems[y_idx];
                        prev_diff_mems[x_idx] += w * next_diff_val;  //每个input上的梯度累加
                    }
                }
            }

            //dy/dw = x --> weights_grad
            weights_grad.fill(0.0f); // 每次计算前，梯度归0
            for (size_t i = 0; i < batch_size; i++) {
                for (size_t j = 0; j < input_dim; j++) {
                    for (size_t k = 0; k < output_dim; k++) {
                        const float x = input_mems[i * input_dim + j];
                        const size_t y_idx = i * output_dim + k;
                        const float y = output_mems[y_idx];
                        const float next_diff_val = next_diff_mems[y_idx];

                        const size_t w_idx = j * output_dim + k;
                        const float w = weights_mems[w_idx];
                        weights_grad_mems[w_idx] += x * next_diff_val;  //每个input上的梯度累加
                    }
                }
            }
            for (size_t i = 0; i < weights_grad.get_shape().get_total(); i++) {
                weights_grad_mems[i] /= batch_size;  //梯度求平均
            }

            bias_grad.fill(0.0f); // 每次计算前，梯度归0
            if (!bias.is_empty()){
                //dy/db = 1 --> bias_grad
                for (size_t i = 0; i < batch_size; i++) {
                    for (size_t k = 0; k < output_dim; k++) {
                        const size_t y_idx = i * output_dim + k;
                        const float next_diff_val = next_diff_mems[y_idx];
                        bias_grad_mems[k] += 1.0f * next_diff_val;  //每个input上的梯度累加
                    }
                }
                for (size_t i = 0; i < bias_grad.get_shape().get_total(); i++) {
                    bias_grad_mems[i] /= batch_size;  //梯度求平均
                }
            }
        };

    }
}// namespace MiniDL
