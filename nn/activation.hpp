#pragma once

#include "../math/matrix.hpp"
#include "matrix.hpp"

namespace network::activation {

enum class Type { ReLU, Sigmoid, Softmax };

template<typename T>
math::Matrix<T> apply_activation(const math::Matrix<T>& x, Type type);

template<typename T>
math::Matrix<T> apply_activation_deriv(const math::Matrix<T>& x, Type type);

} // namespace nn::activation