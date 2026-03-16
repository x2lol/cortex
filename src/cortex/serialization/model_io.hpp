#pragma once

#include <string>
#include <cortex/nn/network.hpp>

namespace cortex::serialization {

struct ModelHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
};

struct LayerHeader {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t activation;
};

class ModelIO {
public:

    static void save(const std::string& path, const cortex::Network& net);

    static cortex::Network load(const std::string& path);

};

}