#include "Network.h"
#include "Tools.h"
#include <iostream>

namespace MiniDL {
    Network& Network::add_op(Operator* op) {
        do_assert(op != nullptr, "op can't be NULL");
        ops.push_back(op);
        return *this;
    }

    //set LossFunctor
    void Network::set_loss_functor(LossFunctor* loss_functor) {
        do_assert(loss_functor != nullptr, "loss_functor can't be NULL");
        this->loss_functor = loss_functor;
    }

    //forward
    void Network::forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts) {
        //TODO
        // std::vector<Operator*> ops = this->ops;
        for (size_t i = 0; i < this->ops.size(); i++) {
            this->ops[i]->forward(inputs, predicts); // ERROR predicts是变化的
        }
    }

    //backward
    void Network::backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruths) {
        do_assert(loss_functor != nullptr, "loss_functor can't be NULL");
        //TODO
        const Data& input = *inputs[0];
        const Data& groundtruth = *groundtruths[0];

        MiniDL::Data prev_diff(MiniDL::Shape(input.get_shape()));
        prev_diff.fill(0.0f);
        MiniDL::Data next_diff(MiniDL::Shape(groundtruth.get_shape()));
        next_diff.fill(1.0f);
        std::vector<MiniDL::Data*> prev_diffs{ &prev_diff };
        std::vector<MiniDL::Data*> next_diffs{ &next_diff };

        for (size_t i = 0; i < this->ops.size(); i++) {
            this->ops[i]->backward(inputs, prev_diffs, next_diffs, groundtruths);
        }
    }

    //calculate loss
    float Network::get_loss(const std::vector<Data*>& groundtruths, const std::vector<Data*>& predicts) {
        do_assert(loss_functor != nullptr, "loss can't be NULL");
        return loss_functor->get_loss(groundtruths, predicts);
    }

    // update weights
    void Network::update_weights() {
        //TODO
    }

    //save model
    bool Network::save_model(const std::string& model_path) const {
        //TODO
        return false;
    }

    //load model
    bool Network::load_model(const std::string& model_path) {
        //TODO
        return false;
    }

} // namespace MiniDL