# MyTorch

A lightweight neural network implementation using pure NumPy, designed for educational purposes and deep learning fundamentals. This project implements core deep learning components from scratch, helping users understand the inner workings of neural networks without the abstraction of larger frameworks.

## Features

### Models
- **Linear Models**: Simple linear regression implementation
- **Logistic Regression**: Binary classification with sigmoid activation
- **Dense Feed-Forward Networks**: Multi-layer neural networks with configurable:
  - Layer sizes
  - Activation functions
  - Weight initialization

### Activation Functions
- Sigmoid
- ReLU
- Tanh
- Softmax

### Optimization
- Multiple optimization algorithms:
  - Vanilla Gradient Descent
  - Momentum
  - RMSprop
- Loss functions:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy (BCE)
  - Categorical Cross-Entropy (CCE)
- Regularization:
  - L1 (Lasso)
  - L2 (Ridge)
- Early stopping

### Data Handling
- Support for:
  - Tabular data (CSV, TXT)
  - Image data
  - Text data
- Built-in data preprocessing
- Train/validation/test splitting

### Reporting
- Comprehensive visualization tools:
  - Loss curves
  - Confusion matrices
  - Parity plots
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sm110101/mytorch.git
cd mytorch
```

2. Create and activate venv:

```bash
python -m venv venv

## Windows
venv\Scripts\activate

## Mac/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Simple example training a feed-forward neural network

```python
from models import DenseFeedForwardNetwork
from optimizers import Optimizer, mse_loss
from data import split_data
import numpy as np
```

### Create Synthetic Data

```python
X = np.random.randn(100, 5)
y = np.sum(X, axis=1) + np.random.randn(100) 0.1
```

### Split Data

```python
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
```

### Create Model 

```python
model = DenseFeedForwardNetwork(
    layer_sizes=[5, 10, 1],
    activations=[np.tanh, lambda x: x] # Hidden layer: tanh, Output layer: linear
)
```

### Create Optimizer

```python
optimizer = Optimizer(
    model=model,
    loss_fn=mse_loss,
    lr=0.01,
    method='rmsprop'
)
```

### Train Model

```python
history = optimizer.fit(
    X_train, 
    y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=100,
    early_stopping_patience=5
)
```

## Project Structure
```bash
mytorch/
│
├── models.py # Neural network model implementations
├── optimizers.py # Optimization algorithms and training loops
├── data.py # Data loading and preprocessing utilities
└── report.py # Visualization and metrics reporting tools
```

## Further Documentation

### Models

The base `Model` class provides a template for all models:
- `forward()`: Defines the forward pass
- `get_parameters()`: Returns model parameters
- `set_parameters()`: Updates model parameters
- `get_flat_parameters()`: Returns flattened parameters for optimization
- `set_flat_parameters()`: Updates parameters from flattened form

### Optimizers

The `Optimizer` class handles:
- Gradient computation using finite differences
- Parameter updates with various optimization methods
- Training loop with early stopping
- L1/L2 regularization

### Data Utilities

The data module provides:
- Data loading for various formats
- Preprocessing functions
- Train/validation/test splitting
- Data saving utilities

### Reporting

The `Reporter` class generates:
- Training visualizations
- Performance metrics
- Model evaluation reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
