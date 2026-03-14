#include "dense_layer.hpp"
#include <random>

namespace network {

DenseLayer::DenseLayer(size_t in_size, size_t out_size, activation::Type a) : W(out_size, in_size), b(out_size, 1), act(a) {
    // Xavier init
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f / std::sqrt(in_size), 1.0f / std::sqrt(in_size));

    for(size_t i = 0; i < W.size(); ++i) W[i] = dist(gen);
    for(size_t i = 0; i < b.size(); ++i) b[i] = 0;
}

    math::Matrix<float> DenseLayer::forward(const math::Matrix<float>& x) {
    input = x; // cache for backprop
    math::Matrix<float> z = W * x + b; // matrix + matrix broadcast
    output = activation::apply_activation(z, act);
    return output;
}

    math::Matrix<float> DenseLayer::backward(const math::Matrix<float>& dA, float lr) {
    // derivative of activation
    math::Matrix<float> dZ = activation::apply_activation_deriv(output, act).hadamard(dA);

    math::Matrix<float> dW = dZ * input.transpose();
    // db = sum along columns
    math::Matrix<float> db(dZ.rows(),1);
    for(size_t i = 0; i < dZ.rows(); ++i){
        float sum = 0;
        for(size_t j = 0; j < dZ.cols(); ++j) sum += dZ(i,j);
        
        db(i,0)=sum;
    }
    // update
    W = W - dW * lr;
    b = b - db * lr;
    // return dA for previous layer
    return W.transpose() * dZ;
}

}
