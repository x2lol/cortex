#include "network.hpp"

namespace network {
    math::Matrix<float> Network::forward(const math::Matrix<float>& input){
        math::Matrix<float> activations = input;
        
        for(DenseLayer& layer: layers) activations = layer.forward(activations);
        
        return activations;
    }

    void Network::backward(const math::Matrix<float>& grad_output, float lr){
        math::Matrix<float> grad_input = grad_output;
        
        for(auto it = layers.rbegin(); it != layers.rend(); ++it){
            grad_input = it->backward(grad_input, lr);
        }
    }
}