#include <fstream> 

#include "loader.hpp"
#include "../exceptions/exceptions.hpp"

namespace network::loader {

IDXDataset IDXLoader::load(const std::string& image_file, const std::string& label_file) {
    IDXDataset dataset;
    load_images(image_file, dataset);
    load_labels(label_file, dataset);
    return dataset;
}

uint32_t IDXLoader::read_be32(std::ifstream& f, const std::string& path) {
    std::array<uint8_t, 4> bytes;
    if (!f.read(reinterpret_cast<char*>(bytes.data()), 4)) {
        throw FileReadException(path);
    }

    // endiannes conversion
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

void IDXLoader::load_images(const std::string& path, IDXDataset& dataset) {
    std::ifstream file(path, std::ios::binary);
    
    
    if (!file) {
        throw FileOpenException(path);
    }
    
    uint16_t zero;
    uint8_t type;
    uint8_t dims;

    if(!file.read(reinterpret_cast<char*>(&zero), 2)) throw FileReadException(path);
    if(!file.read(reinterpret_cast<char*>(&type), 1)) throw FileReadException(path);
    if(!file.read(reinterpret_cast<char*>(&dims), 1)) throw FileReadException(path);

    dataset.count = read_be32(file, path);
    dataset.rows  = read_be32(file, path);
    dataset.cols  = read_be32(file, path);

    dataset.images.resize(dataset.count * dataset.rows * dataset.cols);
    if(!file.read(reinterpret_cast<char*>(dataset.images.data()), dataset.images.size())) throw FileReadException(path);

}

void IDXLoader::load_labels(const std::string& path, IDXDataset& dataset) {
    std::ifstream file(path, std::ios::binary);
        
    if (!file) {
        throw FileOpenException(path);
    }

    uint16_t zero;
    uint8_t type;
    uint8_t dims;

    if(!file.read(reinterpret_cast<char*>(&zero), 2)) throw FileReadException(path);
    if(!file.read(reinterpret_cast<char*>(&type), 1)) throw FileReadException(path);
    if(!file.read(reinterpret_cast<char*>(&dims), 1)) throw FileReadException(path);

    uint32_t count = read_be32(file, path);

    dataset.labels.resize(count);
    if(!file.read(reinterpret_cast<char*>(dataset.labels.data()), count)) throw FileReadException(path);
}

IDXDataset MNISTLoader::load(const std::string& base_path) {
    return IDXLoader::load(std::format("{}/train-images-idx3-ubyte", base_path), std::format("{}/train-labels-idx1-ubyte", base_path));
}


IDXDataset EMNISTLoader::load(const std::string& base_path, const std::string& split) {
    const std::string images = std::format("{}/emnist-{}-train-images-idx3-ubyte", base_path, split);
    const std::string labels = std::format("{}/emnist-{}-train-labels-idx1-ubyte", base_path, split);
    return IDXLoader::load(images, labels);
}

} // namespace network::loader
