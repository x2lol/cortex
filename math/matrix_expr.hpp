#pragma once

#include <format>

namespace network::math {
    template<typename Derived, typename T>
    class MatrixExpr {
    public:
        const Derived& derived() const {
            return static_cast<const Derived&>(*this);
        }

        size_t rows() const {
            return derived().rows();
        }

        size_t cols() const {
            return derived().cols();
        }

        size_t size() const {
            return derived().size();
        }

        T operator()(size_t r, size_t c) const {
            return derived()(r,c);
        }

        bool is_vector() const {
            const auto& d = derived();
            return (d.cols() == 1 || d.rows() == 1);
        }

        template<typename RHS>
        T dot(const MatrixExpr<RHS,T>& rhs) const {

            const auto& A = derived();
            const auto& B = rhs.derived();

            if (!A.is_vector() || !B.is_vector() || A.size() != B.size()) {
                throw DimensionMismatch(std::format("dot product is not defined for {}x{} and {}x{}", A.rows(), A.cols(), B.rows(), B.cols()));
            }

            T res{};

            for (size_t i = 0; i < A.rows(); ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    res += A(i,j) * B(i,j);
                }
            }

            return res;
        }
    };
} // namespace network::math