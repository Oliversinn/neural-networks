# Neural Networks from Scratch

A fully manual implementation of a neural network built using only NumPy matrix operations — no TensorFlow, no PyTorch, no high-level ML abstractions. Every component is derived directly from the underlying mathematics.

The network is trained on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset to classify 10 categories of clothing items and achieves **~88% accuracy** after 400 training iterations.

---

## Motivation

The goal of this project is to demonstrate a deep understanding of how neural networks work at the mathematical level, and the ability to implement them from scratch without relying on autograd engines or pre-built layer abstractions. This includes hand-coded implementations of:

- **Sigmoid activation function**
- **Feed Forward** algorithm
- **Backpropagation** algorithm
- **Cross-entropy cost function**
- **L-BFGS-B optimization** via `scipy.optimize.minimize`

---

## Architecture

| Layer | Neurons |
|-------|---------|
| Input | 784 (28×28 pixel images) |
| Hidden | 100 |
| Output | 10 (one per clothing category) |

### Clothing categories

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## How It Works

### 1. Sigmoid Activation

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 2. Feed Forward

For each layer $l$, the activation matrix is computed as:

$$a^{(l+1)} = \sigma\left( \begin{bmatrix} 1 \mid a^{(l)} \end{bmatrix} \cdot \Theta^{(l)\,T} \right)$$

where a bias column of ones is prepended before multiplying by the weight matrix.

### 3. Cost Function (Cross-Entropy)

$$J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h^{(i)}) + (1 - y^{(i)}) \log(1 - h^{(i)}) \right]$$

### 4. Backpropagation

The error signal $\delta$ is propagated backwards through each layer:

$$\delta^{(l)} = \left(\delta^{(l+1)} \cdot \Theta^{(l)}_{[:,1:]}\right) \odot a^{(l)} \odot (1 - a^{(l)})$$

The gradient for each weight matrix is:

$$\frac{\partial J}{\partial \Theta^{(l)}} = \frac{1}{m} \, \delta^{(l+1)\,T} \cdot \begin{bmatrix} 1 \mid a^{(l)} \end{bmatrix}$$

### 5. Optimization

Weights are optimized using **L-BFGS-B** (`scipy.optimize.minimize`) with the backpropagation gradient, running for up to 400 iterations.

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

Open the notebook:

```bash
jupyter lab neural-networks/neural-networks.ipynb
```

---

## Results

| Metric | Value |
|--------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Hidden neurons | 100 |
| Optimization iterations | 400 |
| **Test accuracy** | **~88%** |

---

## Tech Stack

- **Python 3**
- **NumPy** — matrix operations
- **SciPy** — L-BFGS-B optimizer
- **scikit-learn** — train/test split utilities
- **Jupyter Lab** — interactive notebook environment
