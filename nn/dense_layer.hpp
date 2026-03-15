// dense_layer.hpp
#pragma once

#include "activation.hpp"
#include "../math/matrix.hpp"

namespace network {
    struct DenseLayer {
        math::Matrix<float> W, Z, b;
        activation::Type act;

        math::Matrix<float> input;
        math::Matrix<float> output;

        DenseLayer(size_t in_size, size_t out_size, activation::Type act);

        math::Matrix<float> forward(const math::Matrix<float>& x);
        math::Matrix<float> backward(const math::Matrix<float>& grad_output, float learning_rate);
    };
} // namespace network