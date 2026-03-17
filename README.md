# Cortex

Cortex is a minimal neural network library written in C++ using only the STL.

This project was my “hello world” in machine learning - an attempt to understand the mathematical foundations of neural networks by implementing an MLP from scratch.

## Notes

* Depends only on the C++ standard library
* C++20 is set due to `<format>` for logging; core logic should work with C++11+ if logging is replaced
* Not production-ready: no autotests, limited safety, and not heavily optimized
* Designed mainly for learning and experimentation

## Features

* Fully connected neural networks (MLP)
* Forward and backward propagation
* Cross-entropy loss
* SGD and mini-batch training
* IDX dataset support (e.g. EMNIST)
* Custom model serialization (`.nn` format)

## Usage

See `main.cpp` for an example of training on the EMNIST digits dataset.

## Setup

```bash
git clone https://github.com/x2lol/cortex
cd cortex
mkdir build && cd build
cmake ..
make
```

Download and extract EMNIST (IDX format) into the dataset directory expected by the loader.

## Example training log and Demo
<img width="800" alt="logs" src="https://github.com/user-attachments/assets/30c7866a-13a8-46fa-b41c-2159fefc7ce2" />

<div style="text-align:center;">
  <img width="400" alt="image0" src="https://github.com/user-attachments/assets/c00aef57-5139-488d-b71b-c4538e930549" />
  <img width="400" alt="image1" src="https://github.com/user-attachments/assets/32d3ec20-9a07-480c-a135-71610e9ba27a" />
</div>

You can probably check out a live demo at [bilol-abdilxayev.colab.duke.edu/models/character-recognition/](https://bilol-abdilxayev.colab.duke.edu/models/character-recognition/), unless my VM was shut down by the university. Keep in mind that EMNIST images are transposed relative to MNIST, so you should transpose the input to match the example here.

## Contact

If something breaks or is unclear, open an issue or email me.

## Disclaimer

Read the code before using it.
