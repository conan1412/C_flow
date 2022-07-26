#pragma once
#include "Configure.h"

namespace MiniDL {
    //constant initialization
    void constant_filler(DataType& data, const float val);

    //gaussian initialization
    void gaussian_filler(DataType& data, const float mean, const float std);

    //uniform initialization
    void uniform_filler(DataType& data, const float min_val, const float max_val); 
}