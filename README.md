# Cortex

Cortex is a minimal neural network library written in C++ using only the STL.

This project was my “hello world” in machine learning - an attempt to understand the mathematical foundations of neural networks by implementing an MLP from scratch.

---

## Notes

* Depends only on the C++ standard library
* C++20 is set due to `<format>` for logging; core logic should work with C++11+ if logging is replaced
* Not production-ready: no autotests, limited safety, and not heavily optimized
* Designed mainly for learning and experimentation

---

## Features

* Fully connected neural networks (MLP)
* Forward and backward propagation
* Cross-entropy loss
* SGD and mini-batch training
* IDX dataset support (e.g. EMNIST)
* Custom model serialization (`.nn` format)

---

## Usage

See `main.cpp` for an example of training on the EMNIST digits dataset.

---

## Setup

```bash
git clone https://github.com/x2lol/cortex
cd cortex
mkdir build && cd build
cmake ..
make
```

Download and extract EMNIST (IDX format) into the dataset directory expected by the loader.

---

## Example output

(add training log screenshot here)

---

## Contact

If something breaks or is unclear, open an issue or email me.

---

## Disclaimer

Read the code before using it.
