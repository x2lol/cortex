#pragma once;

#include <string>
#include <format>
#include <stdexcept>

namespace network::math {
    class LinearAlgebraException : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class DimensionMismatch : public LinearAlgebraException{
    public:
        DimensionMismatch(const std::string& msg) : LinearAlgebraException("Dimension mismatch: " + msg) {}
    };

    class OutOfBounds : public LinearAlgebraException {
    public:
        OutOfBounds(const std::string& msg) : LinearAlgebraException ("Out of bounds: " + msg) {}
    };
} // namespace network::math