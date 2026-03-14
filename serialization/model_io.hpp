#pragma once
#include <vector>
#include <string>

namespace network::serialization {

class ModelIO {
public:
    static void save(const std::string& path, const std::vector<float>& weights, const std::vector<float>& biases);

    static void load(const std::string& path, std::vector<float>& weights, std::vector<float>& biases);
};

} // namespace serialization