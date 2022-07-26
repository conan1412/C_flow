#pragma once
#include "Operator.h"

namespace MiniDL {
    // full connection
    // y = w * x + b
    // x: inputs
    // y: outputs
    // w: weights
    // b: bias

    class DenseOP : public Operator {
    public:
        DenseOP();
        virtual ~DenseOP();
    public:
        //setting
        void setup(const int input_dim, const int output_dim, const bool with_bias);
        const Data& get_weights() const;
        const Data& get_bias() const;

        //get all weights of op
        virtual std::vector<Data*> get_all_weights();
        //get all grads of op
        virtual std::vector<Data*> get_all_grads();
        //forward of op
        virtual void forward(const std::vector<Data*>& inputs,
            std::vector<Data*>& outputs);
        //backward of op
        virtual void backward(const std::vector<Data*>& inputs,
            std::vector<Data*>& prev_diffs,
            const std::vector<Data*>& next_diffs,
            const std::vector<Data*>& outputs);
    private:
        //inner paramters
        Data weights;
        Data weights_grad;
        Data bias;
        Data bias_grad;
    };


} // namespace MiniDL
