#pragma once
#include "Configure.h"
#include "Shape.h"

namespace MiniDL{
    class Data{
    public:
        Data();
        Data(const Shape& new_shape);
        ~Data();
    public:
        void reshape(const Shape& new_shape);
        Shape get_shape() const;
        const DataType& get_data() const;
        DataType& get_data();
        void fill(const float val);
    private:
        //shape of data
        Shape shape;
        // data container
        DataType data;
    };
    
    
}
