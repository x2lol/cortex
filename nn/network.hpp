#pragma once
#include <vector>
#include "dense_layer.hpp"

namespace network {
    struct Network {
        std::vector<DenseLayer> layers;

        network::math::Matrix<float> forward(const network::math::Matrix<float>& x);
        void backward(const network::math::Matrix<float>& grad_output, float lr);
    };
} // nemaspace network