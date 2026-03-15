#pragma once
#include <cmath>
#include "../math/matrix.hpp"

namespace network {
    inline float cross_entropy(const math::Matrix<float>& y_pred, const math::Matrix<float>& y_true) {
        float loss = 0;
        for(size_t i = 0; i < y_pred.rows(); ++i) {
            for(size_t j = 0; j < y_pred.cols(); ++j) {
                loss -= y_true(i,j) * std::log(std::max(y_pred(i,j), 1e-8f));
            }
        }
        return loss / y_pred.cols(); // average over batch
    }

    inline math::Matrix<float> cross_entropy_grad(const math::Matrix<float>& y_pred, const math::Matrix<float>& y_true) {
        math::Matrix<float> grad = y_pred - y_true;
        return grad / y_pred.cols();
    }
}