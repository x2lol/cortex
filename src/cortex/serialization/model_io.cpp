#include <fstream>
#include <vector>

#include <cortex/serialization/model_io.hpp>
#include <cortex/exceptions/exceptions.hpp>

namespace cortex::serialization {

static const uint32_t MODEL_MAGIC = 0x4E4E4D44;
static const uint32_t MODEL_VERSION = 1;

void ModelIO::save(const std::string& path, const cortex::Network& net) {

    std::ofstream file(path, std::ios::binary);

    if(!file) {
        throw loader::FileOpenException(path);
    }

    ModelHeader header;
    header.magic = MODEL_MAGIC;
    header.version = MODEL_VERSION;
    header.layer_count = net.layers.size();

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for(const DenseLayer& layer : net.layers) {

        LayerHeader lh;

        lh.input_size = layer.W.cols();
        lh.output_size = layer.W.rows();
        lh.activation = static_cast<uint32_t>(layer.act);

        file.write(reinterpret_cast<const char*>(&lh), sizeof(lh));
    }

    for(const DenseLayer& layer : net.layers) {

        file.write(reinterpret_cast<const char*>(layer.W.data()), layer.W.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(layer.b.data()), layer.b.size() * sizeof(float));
    }
}

cortex::Network ModelIO::load(const std::string& path) {

    std::ifstream file(path, std::ios::binary);

    if(!file) {
        throw loader::FileOpenException(path);
    }

    ModelHeader header;

    if(!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw loader::FileReadException(path);
    }

    if(header.magic != MODEL_MAGIC) {
        throw loader::FileReadException("Invalid model format");
    }

    cortex::Network net;

    std::vector<LayerHeader> layer_headers(header.layer_count);

    for(uint32_t i = 0; i < header.layer_count; ++i) {

        if(!file.read(reinterpret_cast<char*>(&layer_headers[i]), sizeof(LayerHeader))) {
            throw loader::FileReadException(path);
        }

        size_t in_size = layer_headers[i].input_size;
        size_t out_size = layer_headers[i].output_size;

        activation::Type act = static_cast<activation::Type>(layer_headers[i].activation);

        net.layers.emplace_back(in_size, out_size, act);
    }

    for(DenseLayer& layer : net.layers) {

        if(!file.read(reinterpret_cast<char*>(layer.W.data()), layer.W.size() * sizeof(float))) {
            throw loader::FileReadException(path);
        }

        if(!file.read(reinterpret_cast<char*>(layer.b.data()), layer.b.size() * sizeof(float))) {
            throw loader::FileReadException(path);
        }
    }

    return net;
}

}