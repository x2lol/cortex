// dense_layer.hpp
#pragma once
#include "matrix.hpp"
#include "activation.hpp"

namespace network {
    struct DenseLayer {
        math::Matrix<float> W;
        math::Matrix<float> b;
        activation::Type act;

        math::Matrix<float> input;
        math::Matrix<float> output;

        DenseLayer(size_t in_size, size_t out_size, activation::Type act);

        math::Matrix<float> forward(const math::Matrix<float>& x);
        math::Matrix<float> backward(const math::Matrix<float>& grad_output, float lr);
    };
} // namespace network