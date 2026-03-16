#include <cortex/nn/network.hpp>
#include <cstring>

namespace cortex {
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

    void Network::serialize(std::vector<float>& weights, std::vector<float>& biases) const {
        for (const DenseLayer& layer : layers) {

            for (size_t i = 0; i < layer.W.size(); ++i) {
                weights.push_back(layer.W[i]);
            }

            for (size_t i = 0; i < layer.b.size(); ++i) {
                biases.push_back(layer.b[i]);
            }
        }
    }

    void Network::initialize(const std::vector<float>& weights, const std::vector<float>& biases) {

        size_t expected_w = 0;
        size_t expected_b = 0;

        for(const DenseLayer& layer : layers) {
            expected_w += layer.W.size();
            expected_b += layer.b.size();
        }

        if(weights.size() != expected_w) throw math::DimensionMismatch("Weight vector size mismatch");

        if(biases.size() != expected_b) throw math::DimensionMismatch("Bias vector size mismatch");

        size_t wi = 0;
        size_t bi = 0;

        for(DenseLayer& layer : layers) {

            size_t w_size = layer.W.size();
            size_t b_size = layer.b.size();

            std::memcpy(layer.W.data(), weights.data() + wi, w_size * sizeof(float));
            std::memcpy(layer.b.data(), biases.data() + bi, b_size * sizeof(float));

            wi += w_size;
            bi += b_size;
        }
    }
} // namspace cortex