#pragma once
#include "loss.hpp"

namespace network {
    void train(Network& net, const math::Matrix<float>& X, const network::math::Matrix<float>& Y, size_t epochs, float learning_rate);
}