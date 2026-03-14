#pragma once

#include "matrix_expr.hpp"

namespace network::math {
    template<typename Expr, typename T>
    class TransposeView : public MatrixExpr<TransposeView<Expr,T>,T> {
    private:
        const Expr& expr_;

    public:
        TransposeView(const Expr& e) : expr_(e) {}

        size_t rows() const { return expr_.cols(); }
        size_t cols() const { return expr_.rows(); }
        size_t size() const { return expr_.size(); }

        decltype(auto) operator()(size_t r, size_t c) const {
            return expr_(c,r);
        }

        const Expr& transpose() const {
            return expr_;
        }
    };
} // namespace network::math