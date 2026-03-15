#pragma once
#include <memory>
#include <cstddef>
#include <functional>
#include <algorithm>

#include "../exceptions/exceptions.hpp"

namespace network::math {

template<typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<T[]> data;

public:
    Matrix() : rows_(0), cols_(0), data(std::make_unique<T[]>(0)) {}

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

    Matrix transpose() const {
        Matrix res(cols_, rows_);

        T* dst = res.data.get();
        const T* src = data.get();

        for (size_t i = 0; i < rows_; ++i) {
            const T* src_row = src + i * cols_;

            for (size_t j = 0; j < cols_; ++j) {
                dst[j * rows_ + i] = src_row[j];
            }
        }

        return res;
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

    Matrix operator+(const Matrix& other) const {
        
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw DimensionMismatch(std::format("cannot add {}x{} matrix to {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }

        Matrix<T> res(rows_, cols_);
        

        for (size_t i = 0; i < this->size(); ++i) {
            res.data[i] = data[i] + other.data[i];
        }

        return res;
    }

    Matrix operator-(const Matrix& other) const {

        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw DimensionMismatch(std::format("cannot substract {}x{} matrix from {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }

        Matrix<T> res(rows_, cols_);

        for (size_t i = 0; i < this->size(); ++i) {
            res.data[i] = data[i] - other.data[i];
        }

        return res;
    }

    Matrix operator*(const Matrix<T>& other) const {

        if (cols_ != other.rows()) {
            throw DimensionMismatch(std::format("cannot multiply {}x{} matrix by {}x{} matrix", rows_, cols_, other.rows(), other.cols()));
        }
        
        Matrix<T> res(rows_, other.cols());
        std::fill(res.data.get(), res.data.get() + res.size(), T{});

        for (size_t i = 0; i < rows_; ++i) {
            T* res_row = res.data.get() + i * other.cols();
            const T* a_row = data.get() + i * cols_;

            for (size_t k = 0; k < cols_; ++k) {
                const T* b_row = other.data.get() + k * other.cols();
                T a = a_row[k];

                for (size_t j = 0; j < other.cols(); ++j) {
                    res_row[j] += a * b_row[j];
                }
            }
        }

        return res;
    }

    Matrix& operator+=(const Matrix<T>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw DimensionMismatch(std::format("cannot add {}x{} matrix to {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }

        for (size_t i = 0; i < this->size(); ++i) {
            data[i] += other.data[i];
        }

        return *this;
    }

    Matrix& operator-=(const Matrix<T>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw DimensionMismatch(std::format("cannot substract {}x{} matrix from {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }
        
        for (size_t i = 0; i < this->size(); ++i) {
            data[i] -= other.data[i];
        }

        return *this;
    }

    Matrix hadamard(const Matrix<T>& other) const {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw DimensionMismatch(std::format("cannot elementwise multiply {}x{} matrix by {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }
        Matrix<T> res(rows_, cols_);

        for (size_t i = 0; i < this->size(); ++i) {
            res.data[i] = data[i] * other.data[i];
        }

        return res;
    }


    Matrix add_colwise(const Matrix& other) const {

        if (other.cols() != 1 || other.rows() != rows_) {
            throw DimensionMismatch(std::format("cannot columnwise broadcast {}x{} matrix to {}x{} matrix", other.rows(), other.cols(), rows_, cols_));
        }

        Matrix res(*this);


        for (size_t i = 0; i < rows_; ++i) {
            T bias = other.data[i];
            const T* a_row = data.get() + i * cols_;
            T* r_row = res.data.get() + i * cols_;

            for (size_t j = 0; j < cols_; ++j) {
                r_row[j] = a_row[j] + bias;
            }
        }

        return res;
    }

    
    bool is_vector() const {
        return (cols_ == 1 || rows_ == 1);
    }

    T dot(const Matrix<T>& other) const {

        if (!this->is_vector() || !other.is_vector() || this->size() != other.size()) {
            throw DimensionMismatch( std::format("dot product is not defined for {}x{} and {}x{}", rows_, cols_, other.rows(), other.cols()));
        }

        const T* a = data.get();
        const T* b = other.data.get();

        size_t n = size();

        T res = T{};

        for (size_t i = 0; i < n; ++i)
            res += a[i] * b[i];

        return res;
    }

};

template<typename T, typename Func>
Matrix<T> apply(const Matrix<T>& e, Func f) {
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
4. Expose data()

3/13/2026

*/