#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace network::loader {

struct IDXDataset {
    size_t count;
    size_t rows;
    size_t cols;
    std::vector<uint8_t> images;
    std::vector<uint8_t> labels;
};

class IDXLoader {
public:
    static IDXDataset load(const std::string& image_file, const std::string& label_file);

private:
    static uint32_t read_be32(std::ifstream& f, const std::string& path);
    static void load_images(const std::string& path, IDXDataset& dataset);
    static void load_labels(const std::string& path, IDXDataset& dataset);
};

class EMNISTLoader {
public:
    static IDXDataset load(const std::string& base_path, const std::string& split="letters");
};

class MNISTLoader {
public:
    static IDXDataset load(const std::string& base_path);
};

} // namespace network::loader