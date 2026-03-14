#pragma once
#include <memory>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <format>

#include "matrix_expr.hpp"
#include "transpose_view.hpp"
#include "exceptions.hpp"

namespace network::math {

template<typename T>
class Matrix : public MatrixExpr<Matrix<T>, T> {
private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<T[]> data;

public:
    Matrix() = delete;
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data(std::make_unique<T[]>(rows_ * cols_)) {}

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data(std::make_unique<T[]>(other.size())) {
        std::copy(
            other.data.get(), 
            other.data.get() + other.size(), 
            data.get()
        );
    }

    Matrix(Matrix&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data(std::move(other.data)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = std::make_unique<T[]>(other.size());

        std::copy(
            other.data.get(),
            other.data.get() + other.size(),
            data.get()
        );

        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) return *this;
        
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = std::move(other.data);
        
        other.rows_ = 0;
        other.cols_ = 0;

        return *this;
    }

    T& operator()(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw OutOfBounds(std::format("cannot access entry at ({}, {}) on a {}x{} matrix", r, c, rows_, cols_));
        }
        
        return data[r*cols_ + c];
    }

    const T& operator()(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw OutOfBounds(std::format("cannot access entry at ({}, {}) on a {}x{} matrix", r, c, rows_, cols_));
        }
        
        return data[r*cols_ + c];
    }

    T& operator[](size_t i) {
        if (i >= this->size()) {
            throw OutOfBounds(std::format("cannot access elementy at {} in a matrix of size {}", i, this->size()));
        }
        
        return data[i];
    }


    const T& operator[](size_t i) const {
        if (i >= this->size()) {
            throw OutOfBounds(std::format("cannot access elementy at {} in a matrix of size {}", i, this->size()));
        }
        
        return data[i];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    template<typename Expr>
    Matrix(const MatrixExpr<Expr, T>& expr) {
        const Expr& e = expr.derived();
        rows_ = e.rows();
        cols_ = e.cols();
        data = std::make_unique<T[]>(rows_ * cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i,j) = e(i,j);
            }
        }
    }

    template<typename Expr>
    Matrix& operator=(const MatrixExpr<Expr,T>& expr) {
        Matrix tmp(expr);
        *this = std::move(tmp);
        return *this;
    }

    auto transpose() const {
        return TransposeView<Matrix<T>, T>(*this);
    }

    Matrix operator*(const T& scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < size(); ++i) res.data[i] = data[i] * scalar;
        return res;
    }

    Matrix operator/(const T& scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < size(); ++i) res.data[i] = data[i] / scalar;
        return res;
    }

    Matrix operator+(const T& scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < size(); ++i) res.data[i] = data[i] + scalar;
        return res;
    }

    Matrix operator-(const T& scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < size(); ++i) res.data[i] = data[i] - scalar;
        return res;
    }

    friend Matrix operator*(const T& scalar, const Matrix& m) {
        return m * scalar;
    }

    template<typename LHS, typename RHS>
    friend Matrix operator+(const MatrixExpr<LHS, T>& lhs, const MatrixExpr<RHS, T>& rhs) {
        const LHS& A = lhs.derived();
        const RHS& B = rhs.derived();

        if (A.rows() != B.rows() || A.cols() != B.cols()) {
            throw DimensionMismatch(std::format("cannot add {}x{} matrix to {}x{} matrix", B.rows(), B.cols(), A.rows(), A.cols()));
        }

        Matrix<T> res(A.rows(), A.cols());
        
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                res(i,j) = A(i,j) + B(i,j);
            }
        }

        return res;
    }

    template<typename LHS, typename RHS>
    friend Matrix operator-(const MatrixExpr<LHS, T>& lhs, const MatrixExpr<RHS, T>& rhs) {
        const LHS& A = lhs.derived();
        const RHS& B = rhs.derived();

        if (A.rows() != B.rows() || A.cols() != B.cols()) {
            throw DimensionMismatch(std::format("cannot substract {}x{} matrix from {}x{} matrix", B.rows(), B.cols(), A.rows(), A.cols()));
        }

        Matrix<T> res(A.rows(), A.cols());

        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                res(i,j) = A(i,j) - B(i,j);
            }
        }

        return res;
    }

    // cache optmized more specific '*' overloading 
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw DimensionMismatch(
                std::format("cannot multiply {}x{} matrix by {}x{} matrix", rows_, cols_, other.rows_, other.cols_));
        }

        Matrix res(rows_, other.cols_);
        std::fill(res.data.get(), res.data.get() + res.size(), T{});

        for (size_t i = 0; i < rows_; ++i) {
            T* res_row = res.data.get() + i * other.cols_;
            const T* a_row = data.get() + i * cols_;

            for (size_t k = 0; k < cols_; ++k) {
                const T* b_row = other.data.get() + k * other.cols_;
                T a = a_row[k];

                for (size_t j = 0; j < other.cols_; ++j) {
                    res_row[j] += a * b_row[j];
                }
            }
        }

        return res;
    }  

    template<typename LHS, typename RHS>
    friend Matrix operator*(const MatrixExpr<LHS, T>& lhs, const MatrixExpr<RHS, T>& rhs) {
        const LHS& A = lhs.derived();
        const RHS& B = rhs.derived();

        if (A.cols() != B.rows()) {
            throw DimensionMismatch(std::format("cannot multiply {}x{} matrix by {}x{} matrix", A.rows(), A.cols(), B.rows(), B.cols()));
        }

        Matrix<T> res(A.rows(), B.cols());
        std::fill(res.data.get(), res.data.get() + res.size(), T{});

        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t k = 0; k < A.cols(); ++k) {
                T a = A(i,k);
                for (size_t j = 0; j < B.cols(); ++j) {
                    res(i,j) += a * B(k,j);
                }
            }
        }
        return res;
    }


    template<typename RHS>
    Matrix& operator+=(const MatrixExpr<RHS, T>& rhs) {
        const RHS& B = rhs.derived();
        
        if (rows_ != B.rows() || cols_ != B.cols()) {
            throw DimensionMismatch(std::format("cannot add {}x{} matrix to {}x{} matrix", B.rows(), B.cols(), rows_, cols_));
        }

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i,j) += B(i,j);
            }
        }

        return *this;
    }

    template<typename RHS>
    Matrix& operator-=(const MatrixExpr<RHS, T>& rhs) {
        const RHS& B = rhs.derived();
        
        if (rows_ != B.rows() || cols_ != B.cols()) {
            throw DimensionMismatch(std::format("cannot substract {}x{} matrix from {}x{} matrix", B.rows(), B.cols(), rows_, cols_));
        }

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i,j) -= B(i,j);
            }
        }

        return *this;
    }

    template<typename RHS>
    Matrix hadamard(const MatrixExpr<RHS, T>& rhs) const {
        const RHS& B = rhs.derived();

        if (rows_ != B.rows() || cols_ != B.cols()){
            throw DimensionMismatch(std::format("cannot elementwise multiply {}x{} matrix by {}x{} matrix", B.rows(), B.cols(), rows_, cols_));
        }
        Matrix<T> res(rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                res(i,j) = (*this)(i,j) * B(i,j);
            }
        }
        return res;
    }
};

template<typename Expr, typename Func, typename T>
Matrix<T> apply(const MatrixExpr<Expr,T>& e, Func f) {
    Matrix<T> res(e.rows(), e.cols());

    for (size_t i = 0; i < e.rows(); ++i) {
        for (size_t j = 0; j < e.cols(); ++j) {
            res(i,j) = f(e(i,j));
        }
    }

    return res;
}

} // namespace network::math


/* TODO:

1. Define apply() as a member function
2. Have a proper wrapper for Vector 
3. Extend operator overloading for all possible scenarios e.g a*=b

3/13/2026

*/