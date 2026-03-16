#pragma once

#include <cortex/nn/activation.hpp>
#include <cortex/math/matrix.hpp>

namespace cortex {
    enum class InitType {
        Xavier,
        Zero
    };

    struct DenseLayer {
        math::Matrix<float> W, Z, b;
        activation::Type act;

        math::Matrix<float> input;
        math::Matrix<float> output;

        DenseLayer(size_t in_size, size_t out_size, activation::Type a, InitType init = InitType::Zero);
        
        math::Matrix<float> forward(const math::Matrix<float>& x);
        math::Matrix<float> backward(const math::Matrix<float>& grad_output, float learning_rate);
    };
} // namespace cortex