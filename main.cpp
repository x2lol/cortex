#include <iostream>
#include <format>
#include <vector>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
#include <chrono>
#include <algorithm>

#include "datasets/loader.hpp"
#include "nn/network.hpp"
#include "training/trainer.hpp"
#include "training/loss.hpp"
#include "serialization/model_io.hpp"

#include "math/matrix.hpp"

using namespace network;

constexpr size_t INPUT_SIZE  = 28 * 28;
constexpr size_t OUTPUT_SIZE = 10;

math::Matrix<float> image_to_matrix(const std::vector<uint8_t>& images, size_t index) {
    math::Matrix<float> x(INPUT_SIZE, 1);

    const size_t offset = index * INPUT_SIZE;

    for (size_t i = 0; i < INPUT_SIZE; ++i)
        x(i,0) = images[offset + i] / 255.0f;

    return x;
}

math::Matrix<float> label_to_onehot(uint8_t label) {
    math::Matrix<float> y(OUTPUT_SIZE, 1);

    for (size_t i = 0; i < OUTPUT_SIZE; ++i)
        y(i,0) = 0;

    y(label,0) = 1;

    return y;
}

math::Matrix<float> build_batch_images(const std::vector<uint8_t>& images, const std::vector<size_t>& indices, size_t start, size_t batch_size) {
    math::Matrix<float> X(INPUT_SIZE, batch_size);

    for (size_t b = 0; b < batch_size; ++b) {

        size_t idx = indices[start + b];
        size_t offset = idx * INPUT_SIZE;

        for (size_t i = 0; i < INPUT_SIZE; ++i)
            X(i,b) = images[offset + i] / 255.0f;
    }

    return X;
}

math::Matrix<float> build_batch_labels(const std::vector<uint8_t>& labels, const std::vector<size_t>& indices, size_t start, size_t batch_size) {
    math::Matrix<float> Y(OUTPUT_SIZE, batch_size);

    for (size_t b = 0; b < batch_size; ++b) {

        uint8_t label = labels[indices[start + b]];

        for (size_t j = 0; j < OUTPUT_SIZE; ++j)
            Y(j,b) = 0;

        Y(label,b) = 1;
    }

    return Y;
}

int predict_label(const math::Matrix<float>& y) {
    float max_val = y(0,0);
    int idx = 0;

    for (size_t i = 1; i < y.rows(); ++i) {

        if (y(i,0) > max_val) {
            max_val = y(i,0);
            idx = i;
        }
    }

    return idx;
}

size_t batch_accuracy(const math::Matrix<float>& pred, const math::Matrix<float>& Y) {
    size_t correct = 0;

    for (size_t j = 0; j < pred.cols(); ++j) {

        float max_val = pred(0,j);
        size_t idx = 0;

        for (size_t i = 1; i < pred.rows(); ++i) {
            if (pred(i,j) > max_val) {
                max_val = pred(i,j);
                idx = i;
            }
        }

        if (Y(idx,j) == 1) correct++;
    }

    return correct;
}

int main() {
    try {

        std::cout << "Loading MNIST dataset...\n";

        auto dataset = loader::MNISTLoader::load("../datasets/mnist");

        std::cout << std::format("Loaded {} samples\n", dataset.count);


        Network net;

        net.layers.emplace_back(INPUT_SIZE, 128, activation::Type::ReLU);
        net.layers.emplace_back(128, 64, activation::Type::ReLU);
        net.layers.emplace_back(64, OUTPUT_SIZE, activation::Type::Softmax);


        float learning_rate = 0.001f;
        const size_t sgd_epochs = 3;
        const size_t batch_epochs = 5;
        const size_t batch_size = 64;
        

        std::cout << "\n=== PHASE 1: SGD training ===\n"; 

        for (size_t e = 0; e < sgd_epochs; ++e) {

            float epoch_loss = 0;
            size_t epoch_correct = 0;

            float window_loss = 0;
            size_t window_correct = 0;
            size_t window_count = 0;

            auto epoch_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < dataset.count; ++i) {

                auto x = image_to_matrix(dataset.images, i);
                auto y = label_to_onehot(dataset.labels[i]);

                auto pred = net.forward(x);

                float loss = cross_entropy(pred, y);

                auto grad = cross_entropy_grad(pred, y);
                net.backward(grad, learning_rate);

                int predicted = predict_label(pred);

                bool correct = (predicted == dataset.labels[i]);

                epoch_loss += loss;
                epoch_correct += correct;

                window_loss += loss;
                window_correct += correct;
                window_count++;

                if ((i + 1) % 5000 == 0) {

                    float avg_loss = window_loss / window_count;
                    float acc = 100.0f * window_correct / window_count;

                    std::cout << std::format(
                        "Epoch {} | sample {} | loss {:.4f} | acc {:.2f}%\n",
                        e, i + 1, avg_loss, acc
                    );

                    window_loss = 0;
                    window_correct = 0;
                    window_count = 0;
                }
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

            std::cout << std::format("Epoch {} finished | avg loss {:.4f} | acc {:.2f}% | time {}s\n", e, epoch_loss / dataset.count, 100.0f * epoch_correct / dataset.count, sec);
        }

        // switching learning rate for mini-batch 
        learning_rate = 0.005f;


        std::cout << "\n=== PHASE 2: Mini-batch training ===\n";

        std::filesystem::create_directories("../serialization/saved");

        std::mt19937 rng(std::random_device{}());

        std::vector<size_t> indices(dataset.count);
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t epoch = 0; epoch < batch_epochs; ++epoch) {

            std::cout << std::format("\nEpoch {} started\n", epoch);

            auto epoch_start = std::chrono::high_resolution_clock::now();

            std::shuffle(indices.begin(), indices.end(), rng);

            float epoch_loss = 0;
            size_t batches = 0;

            for (size_t start = 0; start < dataset.count; start += batch_size) {

                size_t current_batch = std::min(batch_size, dataset.count - start);

                auto X = build_batch_images(dataset.images, indices, start, current_batch);
                auto Y = build_batch_labels(dataset.labels, indices, start, current_batch);

                auto pred = net.forward(X);

                float loss = cross_entropy(pred, Y);
                epoch_loss += loss;

                auto grad = cross_entropy_grad(pred, Y);
                net.backward(grad, learning_rate);

                size_t correct = batch_accuracy(pred, Y);

                if (batches % 20 == 0) { 
                    std::cout << std::format("Epoch {} | Batch {} | loss {:.4f} | acc {:.2f}%\n", epoch, batches, loss, 100.0f * correct / current_batch);
                }

                batches++;
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

            std::cout << std::format("Epoch {} finished | avg loss = {} | time = {}s\n", epoch, epoch_loss / batches, sec);

            // checkpoint
            std::string path = std::format("../serialization/saved/model_epoch_{}.bin", epoch);

            std::vector<float> weights;
            std::vector<float> biases;

            net.serialize(weights, biases);

            serialization::ModelIO::save(path, weights, biases);

            std::cout << std::format("Saved checkpoint {}\n", path);
        }

        std::cout << "\nEvaluating model...\n";

        size_t correct = 0;

        for (size_t i = 0; i < dataset.count; ++i) {

            auto x = image_to_matrix(dataset.images, i);

            auto y_pred = net.forward(x);

            int pred = predict_label(y_pred);

            if (pred == dataset.labels[i])
                ++correct;
        }

        float accuracy = static_cast<float>(correct) / dataset.count;

        std::cout << std::format("Final accuracy: {:.2f}%\n", accuracy * 100);

    }
    catch (const std::exception& e) {

        std::cerr << std::format("Fatal error: {}\n", e.what());

        return 1;
    }

    return 0;
}