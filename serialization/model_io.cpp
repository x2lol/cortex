#include <vector>
#include <string>
#include <fstream>

#include "model_io.hpp"
#include "../exceptions/exceptions.hpp"

namespace network::serialization {

    void ModelIO::save( const std::string& path, const std::vector<float>& weights, const std::vector<float>& biases) {
        std::ofstream file(path, std::ios::binary);

        if (!file) {
            throw loader::FileOpenException(path);
        }

        const size_t w = weights.size();
        const size_t b = biases.size();
        
        file.write(reinterpret_cast<const char*>(&w), sizeof(w));
        file.write(reinterpret_cast<const char*>(weights.data()), w * sizeof(float));

        file.write(reinterpret_cast<const char*>(&b), sizeof(b));
        file.write(reinterpret_cast<const char*>(biases.data()), b * sizeof(float));
        
    }

    void ModelIO::load(const std::string& path, std::vector<float>& weights, std::vector<float>& biases) {
        std::ifstream file(path, std::ios::binary);
        size_t w, b;

        if(!file.read(reinterpret_cast<char*>(&w), sizeof(w))) throw loader::FileReadException(std::format("at path: {} while trying to infer size of 'weights'", path));
        weights.resize(w);
        if(!file.read(reinterpret_cast<char*>(weights.data()), w * sizeof(float))) throw loader::FileReadException(std::format("at path: {} while trying to read 'weights'", path));

        if(!file.read(reinterpret_cast<char*>(&b), sizeof(b))) throw loader::FileReadException(std::format("at path: {} while trying to infer size of 'biases'", path));
        biases.resize(b);
        if(!file.read(reinterpret_cast<char*>(biases.data()), b * sizeof(float))) throw loader::FileReadException(std::format("at path: {} while trying to read 'biases'", path));
    }

}