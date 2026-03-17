#include <iostream>
#include <format>

#include <cortex/datasets/loader.hpp>
#include <cortex/nn/network.hpp>
#include <cortex/training/trainer.hpp>

using namespace cortex;

int main() {

    try {

        std::cout << "Loading EMNIST Balanced dataset...\n";

        loader::IDXDataset train_dataset = loader::EMNISTLoader::load("../src/cortex/datasets/emnist", "digits", "train");
        loader::IDXDataset test_dataset = loader::EMNISTLoader::load("../src/cortex/datasets/emnist", "digits", "test");

        std::cout << std::format("Loaded {} samples\n", train_dataset.count);
        std::cout << std::format("Loaded {} testing samples\n", test_dataset.count);

        Network net;

        net.layers.emplace_back(784, 256, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(256, 128, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(128, 64, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(64, 10, activation::Type::Softmax, InitType::Xavier);

        training::Trainer trainer(net, train_dataset, test_dataset);

        std::cout << "\n=== SGD warmup ===\n";
        trainer.train_sgd(3, 0.001f);

        std::cout << "\n=== Mini-batch training ===\n";
        trainer.train_minibatch(12, 64, 0.005f, "../src/cortex/serialization/saved");

        float accuracy = trainer.evaluate();

        std::cout << std::format("\nFinal accuracy {:.2f}%\n", accuracy * 100.0f);

    }
    catch(const std::exception& e) {

        std::cerr << std::format("Fatal error: {}\n", e.what());

        return 1;
    }

    return 0;
}