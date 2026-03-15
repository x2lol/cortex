#include "network.hpp"

namespace network {
    math::Matrix<float> Network::forward(const math::Matrix<float>& input){
        math::Matrix<float> activations = input;
        
        for(DenseLayer& layer: layers) activations = layer.forward(activations);
        
        return activations;
    }

    void Network::backward(const math::Matrix<float>& grad_output, float learning_rate){
        math::Matrix<float> grad_input = grad_output;
        
        for(auto it = layers.rbegin(); it != layers.rend(); ++it){
            grad_input = it->backward(grad_input, learning_rate);
        }
    }

    void Network::serialize(std::vector<float>& weights, std::vector<float>& biases) {
        for (DenseLayer& layer : layers) {

            for (size_t i = 0; i < layer.W.size(); ++i) {
                weights.push_back(layer.W[i]);
            }

            for (size_t i = 0; i < layer.b.size(); ++i) {
                biases.push_back(layer.b[i]);
            }
        }
    }
}