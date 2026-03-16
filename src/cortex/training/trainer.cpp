#include <cortex/training/trainer.hpp>
#include <cortex/training/loss.hpp>
#include <cortex/serialization/model_io.hpp>

#include <iostream>
#include <format>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>

namespace cortex::training {

Trainer::Trainer(cortex::Network& net, const loader::IDXDataset& dataset) : net(net), dataset(dataset) {}

math::Matrix<float> Trainer::image_to_matrix(const std::vector<uint8_t>& images, size_t index, size_t image_size) {
    math::Matrix<float> x(image_size, 1);

    const size_t offset = index * image_size;

    for(size_t i = 0; i < image_size; ++i) {
        x(i,0) = static_cast<float>(images[offset + i]) / 255.0f;
    }

    return x;
}

math::Matrix<float> Trainer::label_to_onehot(uint8_t label, size_t classes) {
    math::Matrix<float> y(classes, 1);

    for(size_t i = 0; i < classes; ++i) {
        y(i,0) = 0.0f;
    }

    y(label,0) = 1.0f;

    return y;
}

math::Matrix<float> Trainer::build_batch_images(const std::vector<uint8_t>& images, const std::vector<size_t>& indices, size_t start, size_t batch_size, size_t image_size) {
    math::Matrix<float> X(image_size, batch_size);

    for(size_t b = 0; b < batch_size; ++b) {

        size_t idx = indices[start + b];
        size_t offset = idx * image_size;

        for(size_t i = 0; i < image_size; ++i) {
            X(i,b) = static_cast<float>(images[offset + i]) / 255.0f;
        }
    }

    return X;
}

math::Matrix<float> Trainer::build_batch_labels(const std::vector<uint8_t>& labels, const std::vector<size_t>& indices, size_t start, size_t batch_size, size_t classes) {
    math::Matrix<float> Y(classes, batch_size);

    for(size_t b = 0; b < batch_size; ++b) {

        uint8_t label = labels[indices[start + b]];

        for(size_t j = 0; j < classes; ++j) {
            Y(j,b) = 0.0f;
        }

        Y(label,b) = 1.0f;
    }

    return Y;
}

int Trainer::predict_label(const math::Matrix<float>& y) {
    float max_val = y(0,0);
    int idx = 0;

    for(size_t i = 1; i < y.rows(); ++i) {

        if(y(i,0) > max_val) {
            max_val = y(i,0);
            idx = static_cast<int>(i);
        }
    }

    return idx;
}

size_t Trainer::batch_accuracy(const math::Matrix<float>& pred, const math::Matrix<float>& Y) {
    size_t correct = 0;

    for(size_t j = 0; j < pred.cols(); ++j) {

        float max_val = pred(0,j);
        size_t idx = 0;

        for(size_t i = 1; i < pred.rows(); ++i) {
            if(pred(i,j) > max_val) {
                max_val = pred(i,j);
                idx = i;
            }
        }

        if(Y(idx,j) == 1.0f) {
            correct++;
        }
    }

    return correct;
}

void Trainer::train_sgd(size_t epochs, float learning_rate) {

    const size_t image_size = dataset.rows * dataset.cols;
    const size_t classes = 47;

    for(size_t e = 0; e < epochs; ++e) {

        float epoch_loss = 0.0f;
        size_t epoch_correct = 0;

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < dataset.count; ++i) {

            math::Matrix<float> x = image_to_matrix(dataset.images, i, image_size);
            math::Matrix<float> y = label_to_onehot(dataset.labels[i], classes);

            math::Matrix<float> pred = net.forward(x);

            float loss = cross_entropy(pred, y);

            math::Matrix<float> grad = cross_entropy_grad(pred, y);
            net.backward(grad, learning_rate);

            int predicted = predict_label(pred);

            if(predicted == dataset.labels[i]) {
                epoch_correct++;
            }

            epoch_loss += loss;

            if((i + 1) % 10000 == 0) {
                std::cout << std::format("Epoch {} | sample {} | loss {:.4f}\n", e, i + 1, loss);
            }
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        long long seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        std::cout << std::format("Epoch {} finished | avg loss {:.4f} | acc {:.2f}% | time {}s\n",
                                 e,
                                 epoch_loss / dataset.count,
                                 100.0f * epoch_correct / dataset.count,
                                 seconds);
    }
}

void Trainer::train_minibatch(size_t epochs, size_t batch_size, float learning_rate, const std::string& checkpoint_dir) {

    const size_t image_size = dataset.rows * dataset.cols;
    const size_t classes = net.layers.back().output.size();

    std::filesystem::create_directories(checkpoint_dir);

    std::vector<size_t> indices(dataset.count);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(std::random_device{}());

    for(size_t epoch = 0; epoch < epochs; ++epoch) {

        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        size_t batches = 0;

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        for(size_t start_idx = 0; start_idx < dataset.count; start_idx += batch_size) {

            size_t current_batch = std::min(batch_size, dataset.count - start_idx);

            math::Matrix<float> X = build_batch_images(dataset.images, indices, start_idx, current_batch, image_size);
            math::Matrix<float> Y = build_batch_labels(dataset.labels, indices, start_idx, current_batch, classes);

            math::Matrix<float> pred = net.forward(X);

            float loss = cross_entropy(pred, Y);

            math::Matrix<float> grad = cross_entropy_grad(pred, Y);
            net.backward(grad, learning_rate);

            epoch_loss += loss;

            if(batches % 50 == 0) {

                size_t correct = batch_accuracy(pred, Y);

                std::cout << std::format(
                    "Epoch {} | batch {} | loss {:.4f} | acc {:.2f}%\n",
                    epoch,
                    batches,
                    loss,
                    100.0f * correct / current_batch
                );
            }

            batches++;
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        long long seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        std::cout << std::format(
            "Epoch {} finished | avg loss {:.4f} | time {}s\n",
            epoch,
            epoch_loss / batches,
            seconds
        );

        std::string path = std::format("{}/model_epoch_{}.nn", checkpoint_dir, epoch);

        serialization::ModelIO::save(path, net);

        std::cout << std::format("Saved checkpoint {}\n", path);
    }
}

float Trainer::evaluate() {

    const size_t image_size = dataset.rows * dataset.cols;
    size_t correct = 0;

    for(size_t i = 0; i < dataset.count; ++i) {

        math::Matrix<float> x = image_to_matrix(dataset.images, i, image_size);

        math::Matrix<float> pred = net.forward(x);

        int predicted = predict_label(pred);

        if(predicted == dataset.labels[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / dataset.count;
}

} // namspace cortex::training