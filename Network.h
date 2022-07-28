#pragma once
#include "Configure.h"
#include "Data.h"
#include "Operator.h"
#include "LossFunctor.h"

namespace MiniDL {
    // 1. 描述网络的拓扑结构
    // 2. 控制数据流向
    // 3. 模型加载保存
    class Network {
    public:
        // add op
        Network& add_op(Operator* op);

        //set LossFunctor
        void set_loss_functor(LossFunctor* loss_functor);

        //forward
        void forward(const std::vector<Data*>& inputs, std::vector<Data*>& predicts);

        //backward
        void backward(const std::vector<Data*>& inputs, const std::vector<Data*>& groundtruths);

        //calculate loss
        float get_loss(const std::vector<Data*>& groundtruths, const std::vector<Data*>& predicts);

        // update weights
        void update_weights();

        //save model
        bool save_model(const std::string& model_path) const;

        //load model
        bool load_model(const std::string& model_path);
    private:
        //all op
        std::vector<Operator*> ops;
        // loss_functor
        LossFunctor* loss_functor = nullptr;
    };

} // namespace MiniDL