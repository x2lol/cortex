#include <iostream>
#include <format>

#include <cortex/datasets/loader.hpp>
#include <cortex/nn/network.hpp>
#include <cortex/training/trainer.hpp>

using namespace namespace;

int main() {

    try {

        std::cout << "Loading EMNIST Balanced dataset...\n";

        loader::IDXDataset dataset = loader::EMNISTLoader::load("/datasets/emnist", "balanced");

        std::cout << std::format("Loaded {} samples\n", dataset.count);

        Network net;

        net.layers.emplace_back(784, 256, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(256, 128, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(128, 64, activation::Type::ReLU, InitType::Xavier);
        net.layers.emplace_back(64, 47, activation::Type::Softmax, InitType::Xavier);

        training::Trainer trainer(net, dataset);

        std::cout << "\n=== SGD warmup ===\n";
        trainer.train_sgd(2, 0.001f);

        std::cout << "\n=== Mini-batch training ===\n";
        trainer.train_minibatch(12, 128, 0.003f, "serialization/saved/");

        float accuracy = trainer.evaluate();

        std::cout << std::format("\nFinal accuracy {:.2f}%\n", accuracy * 100.0f);

    }
    catch(const std::exception& e) {

        std::cerr << std::format("Fatal error: {}\n", e.what());

        return 1;
    }

    return 0;
}