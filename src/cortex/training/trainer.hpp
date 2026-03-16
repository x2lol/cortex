#pragma once

#include <cortex/nn/network.hpp>
#include <cortex/datasets/loader.hpp>
#include <cortex/math/matrix.hpp>

#include <vector>
#include <cstddef>

namespace cortex::training {

struct TrainConfig {
    size_t sgd_epochs;
    size_t batch_epochs;
    size_t batch_size;
    float sgd_lr;
    float batch_lr;
};

class Trainer {
public:

    Trainer(cortex::Network& net, const loader::IDXDataset& dataset);

    void train_sgd(size_t epochs, float learning_rate);

    void train_minibatch(size_t epochs, size_t batch_size, float learning_rate, const std::string& checkpoint_dir);

    float evaluate();

private:

    cortex::Network& net;
    const loader::IDXDataset& dataset;

    static math::Matrix<float> image_to_matrix(const std::vector<uint8_t>& images, size_t index, size_t image_size);

    static math::Matrix<float> label_to_onehot(uint8_t label, size_t classes);

    static math::Matrix<float> build_batch_images(
        const std::vector<uint8_t>& images,
        const std::vector<size_t>& indices,
        size_t start,
        size_t batch_size,
        size_t image_size
    );

    static math::Matrix<float> build_batch_labels(
        const std::vector<uint8_t>& labels,
        const std::vector<size_t>& indices,
        size_t start,
        size_t batch_size,
        size_t classes
    );

    static int predict_label(const math::Matrix<float>& y);

    static size_t batch_accuracy(const math::Matrix<float>& pred, const math::Matrix<float>& Y);
};

} // namespace cortex::training