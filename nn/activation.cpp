#include <cmath>
#include <algorithm>

#include "activation.hpp"
#include "../exceptions/exceptions.hpp"

namespace network::activation {

template<typename T>
math::Matrix<T> apply_activation(const math::Matrix<T>& x, Type type) {
    switch(type) {
        case Type::ReLU:
            return network::math::apply(x, [](T v){ return std::max(T(0), v); });
        case Type::Sigmoid:
            return network::math::apply(x, [](T v){ return T(1) / (T(1) + std::exp(-v)); });
        case Type::Softmax: {
            // row-wise softmax
            Matrix<T> y(x.rows(), x.cols());
            for (size_t i = 0; i < x.rows(); ++i){
                T max_val = x(i,0);
                for(size_t j = 1; j < x.cols(); ++j) if(x(i,j) > max_val) max_val = x(i,j);

                T sum = 0;
                for(size_t j = 0; j < x.cols(); ++j){
                    y(i,j) = std::exp(x(i,j) - max_val);
                    sum += y(i,j);
                }
                for(size_t j = 0; j < x.cols(); ++j) y(i,j) /= sum;
            }
            return y;
        }
        default:
            throw UnsupportedActivationType(type);
    }
}

template<typename T>
math::Matrix<T> apply_activation_deriv(const math::Matrix<T>& x, Type type) {
    switch(type) {
        case Type::ReLU:
            return network::math::apply(x, [](T v){ return v > 0 ? 1 : 0; });
        case Type::Sigmoid: {
            Matrix<T> sig = apply_activation(x, Type::Sigmoid);
            return network::math::apply(sig, [](T v){ return v * (1 - v); });
        }
        default:
            throw UnsupportedActivationType(type);
    }
}

template math::Matrix<float> apply_activation(const math::Matrix<float>&, Type);
template math::Matrix<float> apply_activation_deriv(const math::Matrix<float>&, Type);

} // namespace nn::activation