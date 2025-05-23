# Introduction to Deep Learning with PyTorch: Revision Notes

This document provides a detailed summary of the key concepts, techniques, and code examples from the **Introduction to Deep Learning with PyTorch** course on DataCamp. Use this as a quick reference to revisit core ideas, PyTorch syntax, and practical applications.

## 1. Introduction to Deep Learning and PyTorch

### Key Concepts
- **Deep Learning vs. Machine Learning**: Deep learning uses neural networks with multiple layers to learn complex patterns from data (e.g., images, text, audio), unlike traditional machine learning, which relies on feature engineering.
- **PyTorch Overview**: PyTorch is an open-source deep learning framework known for its dynamic computational graph, Python-first approach, and GPU support, making it ideal for research and prototyping.
- **Tensors**: The core data structure in PyTorch, similar to NumPy arrays but optimized for GPU computations. Tensors support operations like addition, multiplication, and reshaping.

### Key Operations
- **Creating Tensors**:
  - Create a tensor: `torch.tensor([1, 2, 3])`
  - Check shape: `tensor.shape`
  - Check data type: `tensor.dtype`
  - Convert to float: `tensor.float()`
- **Basic Operations**:
  - Addition: `tensor1 + tensor2`
  - Matrix multiplication: `torch.matmul(tensor1, tensor2)`
  - Reshape: `tensor.view(new_shape)`

### Example
```python
import torch
# Create a 2D tensor
tensor = torch.tensor([[1, 2], [3, 4]])
# Check shape and type
print(tensor.shape)  # torch.Size([2, 2])
print(tensor.dtype)  # torch.int64
# Perform operation
tensor_sum = tensor + torch.tensor([[1, 1], [1, 1]])
print(tensor_sum)  # tensor([[2, 3], [4, 5]])
```

## 2. Artificial Neural Networks (ANNs)

### Key Concepts
- **Neural Network Structure**: Consists of input layer, hidden layers, and output layer. Each layer applies a linear transformation followed by a non-linear activation function.
- **Forward Pass**: Input data passes through the network to produce predictions.
- **Loss Function**: Measures the error between predictions and actual values (e.g., Mean Squared Error for regression, Cross-Entropy Loss for classification).
- **Backpropagation**: Computes gradients of the loss with respect to model parameters.
- **Optimizer**: Updates model parameters using gradients (e.g., Stochastic Gradient Descent, Adam).
- **Activation Functions**: Introduce non-linearity (e.g., ReLU: `f(x) = max(0, x)`).

### PyTorch Implementation
- **Define a Neural Network**:
  - Use `torch.nn.Module` to create a custom network.
  - Example: A network with 784 input units (e.g., flattened MNIST images), 200 hidden units, and 10 output units (for 10 digit classes).
- **Loss Functions**:
  - Regression: `nn.MSELoss()`
  - Classification: `nn.CrossEntropyLoss()`
- **Optimizer**: `torch.optim.SGD(model.parameters(), lr=0.01)` or `torch.optim.Adam(model.parameters(), lr=0.001)`

### Example: Building and Training an ANN
```python
import torch
import torch.nn as nn

# Define a neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200)  # Input: 28x28 images, Hidden: 200
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 10)   # Output: 10 classes

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model, loss, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (simplified)
for epoch in range(10):
    for data, target in dataloader:  # Assume dataloader provides data
        optimizer.zero_grad()        # Clear gradients
        output = model(data)         # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## 3. Convolutional Neural Networks (CNNs)

### Key Concepts
- **CNNs for Images**: CNNs use convolutional layers to capture spatial patterns (e.g., edges, textures) in image data, making them ideal for tasks like image classification.
- **Convolutional Layers**: Apply filters to input data to extract features (`nn.Conv2d`).
- **Pooling Layers**: Reduce spatial dimensions (e.g., `nn.MaxPool2d`) to decrease computation and prevent overfitting.
- **Flattening**: Convert 2D feature maps to 1D vectors for fully connected layers.

### PyTorch Implementation
- **Define a CNN**:
  - Use `nn.Conv2d(in_channels, out_channels, kernel_size)` for convolutional layers.
  - Use `nn.MaxPool2d(kernel_size)` for pooling.
- **Example**: A CNN for MNIST with two convolutional layers, max-pooling, and fully connected layers.

### Example: Building a CNN
```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 input channel (grayscale)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 7x7 after pooling

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.fc(x)
        return x

# Instantiate and train (similar to ANN)
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 4. Evaluating and Improving Models

### Key Concepts
- **Model Evaluation**: Use metrics like accuracy, precision, or loss to assess performance.
- **TorchMetrics**: A PyTorch library for computing metrics (e.g., `torchmetrics.Accuracy`).
- **Hyperparameter Tuning**: Adjust learning rate, batch size, or network architecture to improve performance.
- **Overfitting Prevention**:
  - **Regularization**: Add L2 penalty to the loss.
  - **Dropout**: Randomly deactivate neurons during training (`nn.Dropout(p=0.5)`).
  - **Batch Normalization**: Normalize layer inputs to stabilize training (`nn.BatchNorm2d`).
- **Transfer Learning**: Use pre-trained models (not covered in depth in the intro course but introduced as a concept).

### Example: Evaluating with TorchMetrics
```python
from torchmetrics import Accuracy

# Initialize accuracy metric
accuracy = Accuracy(task="multiclass", num_classes=10)

# Evaluate model
model.eval()
with torch.no_grad():
    for data, target in test_dataloader:
        output = model(data)
        acc = accuracy(output, target)
    print(f"Test Accuracy: {acc.item():.4f}")
```

## 5. Practical Applications
- **MNIST Dataset**: Commonly used in the course for digit classification (28x28 grayscale images, 10 classes).
- **Workflow**:
  1. Load data using `torchvision.datasets.MNIST`.
  2. Preprocess with `transforms.ToTensor()` to convert images to tensors.
  3. Create a `DataLoader` for batching.
  4. Train and evaluate ANN or CNN models.
- **Example**:
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load and preprocess MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

## 6. Key Takeaways
- **Tensors**: Master tensor operations for data manipulation.
- **ANNs**: Build and train feedforward neural networks for tabular data or simple classification.
- **CNNs**: Use convolutional layers for image data, achieving better performance than ANNs.
- **Training**: Understand the training loop (forward, loss, backward, optimize).
- **Evaluation**: Use metrics to monitor and improve model performance.
- **PyTorch Flexibility**: Leverage PyTorch’s dynamic graphs for debugging and experimentation.

## 7. Tips for Revision
- **Practice Coding**: Re-run the examples above with the MNIST dataset to reinforce syntax.
- **Experiment**: Try modifying hyperparameters (e.g., learning rate, number of layers) or architectures.
- **Review Errors**: Debug common issues like shape mismatches or incorrect loss functions.
- **Next Steps**: Explore DataCamp’s **Intermediate Deep Learning with PyTorch** for advanced topics like RNNs, LSTMs, and transfer learning.