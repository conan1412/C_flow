#include <algorithm>
#include <random>
#include "MathFunctions.h"

namespace MiniDL {
    //constant initialization
    void constant_filler(DataType& data, const float val) {
        std::fill(data.begin(), data.end(), val);
    }

    //gaussian initialization
    void gaussian_filler(DataType& data, const float mean, const float std) {
        std::default_random_engine generator(std::random_device{}());
        std::normal_distribution<float> distribution(mean, std);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = distribution(generator);
        }
    }

    //uniform initialization
    void uniform_filler(DataType& data, const float min_val, const float max_val) {
        std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<float> distribution(min_val, max_val);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = distribution(generator);
        }
    }
}