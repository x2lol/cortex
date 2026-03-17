#include <cmath>
#include <algorithm>

#include <cortex/math/matrix.hpp>
#include <cortex/exceptions/exceptions.hpp>

namespace cortex::activation {

enum class Type { ReLU, Sigmoid, Softmax };

inline std::string to_string(const Type& type) {
    switch (type)
    {
    case Type::ReLU:
        return "ReLU";
        break;
    case Type::Sigmoid:
        return "Sigmoid";
        break;
    case Type::Softmax:
        return "Softmax";
        break;
    }

    return "Unsupported";
}

template<typename T>
math::Matrix<T> apply_activation(const math::Matrix<T>& x, const Type& type) {
    switch(type) {
        case Type::ReLU:
            return math::apply(x, [](T v){ return std::max(T(0), v); });
        case Type::Sigmoid:
            return math::apply(x, [](T v){ return T(1) / (T(1) + std::exp(-v)); });
        case Type::Softmax: {
            math::Matrix<T> y(x.rows(), x.cols());

            for (size_t j = 0; j < x.cols(); ++j) {
                
                T max_val = x(0, j);
                for (size_t i = 1; i < x.rows(); ++i)
                    if (x(i, j) > max_val)
                        max_val = x(i, j);

                T sum = 0;
                for (size_t i = 0; i < x.rows(); ++i) {
                    y(i, j) = std::exp(x(i, j) - max_val);
                    sum += y(i, j);
                }

                for (size_t i = 0; i < x.rows(); ++i)
                    y(i, j) /= sum;
            }

            return y;
        }
        default:
            throw UnsupportedActivationType(to_string(type));
    }
}

template<typename T>
math::Matrix<T> apply_activation_deriv(const math::Matrix<T>& x, const Type& type) {
    switch(type) {
        case Type::ReLU:
            return math::apply(x, [](T v){ return v > 0 ? 1 : 0; });
        case Type::Sigmoid: {
            math::Matrix<T> sig = apply_activation(x, Type::Sigmoid);
            return math::apply(sig, [](T v){ return v * (1 - v); });
        }
        default:
            throw UnsupportedActivationType(to_string(type));
    }
}

} // namespace cortex::activation

