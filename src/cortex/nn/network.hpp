#pragma once
#include <vector>
#include <cortex/nn/dense_layer.hpp>

namespace cortex {
    struct Network {
        std::vector<DenseLayer> layers;

        math::Matrix<float> forward(const math::Matrix<float>& x);
        void backward(const math::Matrix<float>& grad_output, float learning_rate);
        void serialize(std::vector<float>& weights, std::vector<float>& biases const);
        void initialize(const std::vector<float>& weights, const std::vector<float>& biases); 
        static Network Network::load(const std::string& path);
    };
} // nemaspace cortex