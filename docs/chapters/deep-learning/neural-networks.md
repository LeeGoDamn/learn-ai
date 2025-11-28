# 神经网络基础

::: info 本章概述
神经网络是深度学习的基础。本章将介绍神经网络的核心概念：感知机、激活函数、前向传播和反向传播算法。
:::

## 1. 从感知机到神经网络

### 1.1 感知机（Perceptron）

感知机是最简单的神经网络模型，由Frank Rosenblatt于1957年提出。它模拟了生物神经元的工作方式。

$$y = \begin{cases} 1 & \text{if } \sum_{i} w_i x_i + b > 0 \\ 0 & \text{otherwise} \end{cases}$$

```
   x₁ ─────┐
           │ w₁
   x₂ ─────┼──────►[Σ]──────►[激活]──────► y
           │ w₂
   x₃ ─────┘
           w₃
```

### 1.2 多层感知机（MLP）

单层感知机只能解决线性可分问题。通过堆叠多层神经元，我们可以学习复杂的非线性关系。

```
输入层        隐藏层1       隐藏层2        输出层
  ●──────────●───────────●──────────●
  ●──────────●───────────●──────────●
  ●──────────●───────────●
             ●───────────●
```

## 2. 激活函数

激活函数引入非线性，使神经网络能够学习复杂模式。

### 2.1 常用激活函数

| 函数 | 公式 | 特点 |
|------|------|------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出(0,1)，易梯度消失 |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 输出(-1,1)，零中心化 |
| **ReLU** | $f(x) = \max(0, x)$ | 计算简单，最常用 |
| **Leaky ReLU** | $f(x) = \max(0.01x, x)$ | 解决ReLU死神经元问题 |
| **Softmax** | $\sigma(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | 多分类输出层 |

```python
import numpy as np

# 激活函数实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 数值稳定性
    return exp_x / np.sum(exp_x)

# 导数（用于反向传播）
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

## 3. 前向传播

前向传播是数据从输入层经过隐藏层到输出层的计算过程。

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

```python
class NeuralNetwork:
    def __init__(self, layer_dims):
        """
        layer_dims: 各层神经元数量，如 [784, 128, 64, 10]
        """
        self.L = len(layer_dims) - 1  # 层数
        self.parameters = {}
        
        # 初始化参数（He初始化）
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] = np.random.randn(
                layer_dims[l], layer_dims[l-1]
            ) * np.sqrt(2 / layer_dims[l-1])
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    def forward(self, X):
        """前向传播"""
        self.cache = {'A0': X}
        A = X
        
        for l in range(1, self.L + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A) + b
            
            # 最后一层用softmax，其他用ReLU
            if l == self.L:
                A = self.softmax(Z)
            else:
                A = np.maximum(0, Z)  # ReLU
            
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        return A
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
```

## 4. 反向传播

反向传播是计算损失函数对每个参数梯度的算法，基于链式法则。

### 4.1 链式法则回顾

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### 4.2 反向传播公式

对于输出层（交叉熵损失 + Softmax）：

$$\mathbf{dZ}^{[L]} = \mathbf{A}^{[L]} - \mathbf{Y}$$

对于隐藏层（ReLU激活）：

$$\mathbf{dZ}^{[l]} = \mathbf{W}^{[l+1]T}\mathbf{dZ}^{[l+1]} * g'^{[l]}(\mathbf{Z}^{[l]})$$

参数梯度：

$$\mathbf{dW}^{[l]} = \frac{1}{m}\mathbf{dZ}^{[l]}\mathbf{A}^{[l-1]T}$$

$$\mathbf{db}^{[l]} = \frac{1}{m}\sum\mathbf{dZ}^{[l]}$$

```python
def backward(self, Y):
    """反向传播"""
    m = Y.shape[1]
    grads = {}
    
    # 输出层梯度
    dZ = self.cache[f'A{self.L}'] - Y
    
    for l in reversed(range(1, self.L + 1)):
        A_prev = self.cache[f'A{l-1}']
        
        grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
        grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            W = self.parameters[f'W{l}']
            dA = np.dot(W.T, dZ)
            # ReLU导数
            dZ = dA * (self.cache[f'Z{l-1}'] > 0)
    
    return grads

def update_parameters(self, grads, learning_rate):
    """更新参数"""
    for l in range(1, self.L + 1):
        self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
```

## 5. 完整实现示例

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 生成数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 转换为正确的形状
X_train = X_train.T  # (2, 800)
X_test = X_test.T    # (2, 200)
y_train = y_train.reshape(1, -1)  # (1, 800)
y_test = y_test.reshape(1, -1)    # (1, 200)

class SimpleNN:
    def __init__(self, n_input, n_hidden, n_output):
        # He初始化
        self.W1 = np.random.randn(n_hidden, n_input) * np.sqrt(2/n_input)
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_output, n_hidden) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros((n_output, 1))
    
    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid
        return self.A2
    
    def compute_loss(self, Y, A2):
        m = Y.shape[1]
        loss = -np.mean(Y * np.log(A2 + 1e-8) + (1-Y) * np.log(1-A2 + 1e-8))
        return loss
    
    def backward(self, X, Y):
        m = X.shape[1]
        
        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def train(self, X, Y, epochs, lr):
        losses = []
        for epoch in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            grads = self.backward(X, Y)
            self.update(grads, lr)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses
    
    def predict(self, X):
        A2 = self.forward(X)
        return (A2 > 0.5).astype(int)

# 训练
nn = SimpleNN(n_input=2, n_hidden=16, n_output=1)
losses = nn.train(X_train, y_train, epochs=1000, lr=0.5)

# 评估
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"\n测试集准确率: {accuracy:.4f}")
```

## 6. 常见问题与解决方案

### 6.1 梯度消失/爆炸

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 梯度消失 | Sigmoid/Tanh在饱和区梯度接近0 | 使用ReLU激活函数 |
| 梯度爆炸 | 权重过大导致梯度累积 | 梯度裁剪、BatchNorm |

### 6.2 过拟合

- **正则化**：L1/L2正则化
- **Dropout**：随机丢弃神经元
- **数据增强**：增加训练数据
- **早停**：验证集性能不再提升时停止训练

### 6.3 参数初始化

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| Xavier | $\mathcal{N}(0, \frac{1}{n_{in}})$ | Tanh/Sigmoid |
| He | $\mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU |

## 7. 本章小结

- **感知机** 是神经网络的基本单元
- **激活函数** 引入非线性，ReLU是最常用的选择
- **前向传播** 计算网络输出
- **反向传播** 利用链式法则计算梯度
- 理解这些基础对于使用深度学习框架至关重要

::: tip 下一步
掌握了神经网络基础后，可以学习PyTorch或TensorFlow等深度学习框架，它们会自动处理反向传播和参数更新。
:::
