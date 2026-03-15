#include <format>
#include <iostream>

#include "../nn/network.hpp"
#include "trainer.hpp"

namespace network {
    void train(Network& net, const math::Matrix<float>& X, const network::math::Matrix<float>& Y, size_t epochs, float learning_rate) {
        for(size_t e = 0; e < epochs; ++e){
            math::Matrix<float> y_pred = net.forward(X);
            float loss = cross_entropy(y_pred, Y);
            
            math::Matrix<float> grad = cross_entropy_grad(y_pred, Y);
            net.backward(grad, learning_rate);
        }
    }
}