#pragma once
#include "Data.h"

namespace MiniDL {
    //custom assert
    void do_assert(const bool condition, const char* fmt, ...);

    //log shape
    void log_shape(const Shape& shape);

    //log mems
    void log_data(const Data& mems);
}; // namespace MiniDL


