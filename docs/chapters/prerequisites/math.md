# 数学基础

::: info 本章概述
数学是机器学习和深度学习的基础。本章将帮助你回顾并补充AI学习所需的核心数学知识，包括线性代数、微积分和概率统计。
:::

## 1. 线性代数基础

线性代数是机器学习中最重要的数学工具之一。几乎所有的机器学习算法都涉及到向量和矩阵运算。

### 1.1 向量 (Vector)

向量是一个有序的数字列表，可以表示为列向量或行向量。在机器学习中，向量常用于表示数据特征。

$$\vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

**常用向量运算：**

- **向量加法：** 对应元素相加
- **标量乘法：** 每个元素乘以标量
- **点积（内积）：** $\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$
- **向量范数：** $||\vec{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$（L2范数）

```python
import numpy as np

# 创建向量
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 向量加法
print(a + b)  # [5 7 9]

# 标量乘法
print(2 * a)  # [2 4 6]

# 点积
print(np.dot(a, b))  # 32

# L2范数
print(np.linalg.norm(a))  # 3.7416...
```

### 1.2 矩阵 (Matrix)

矩阵是一个二维数组，在机器学习中用于表示数据集、权重参数等。

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**常用矩阵运算：**

| 运算 | 描述 | 条件 |
|------|------|------|
| 矩阵加法 | 对应元素相加 | 相同形状 |
| 矩阵乘法 | $C_{ij} = \sum_k A_{ik} B_{kj}$ | A的列数 = B的行数 |
| 转置 | 行列互换 $A^T$ | 任意矩阵 |
| 逆矩阵 | $A^{-1}A = I$ | 方阵且可逆 |

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
print(np.dot(A, B))
# [[19 22]
#  [43 50]]

# 转置
print(A.T)
# [[1 3]
#  [2 4]]

# 逆矩阵
print(np.linalg.inv(A))
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)
```

::: tip 为什么重要？
在神经网络中，前向传播本质上就是矩阵乘法：$y = Wx + b$，其中W是权重矩阵，x是输入向量，b是偏置向量。
:::

## 2. 微积分基础

微积分在机器学习中主要用于优化算法，特别是梯度下降法。理解导数和偏导数对于理解模型训练过程至关重要。

### 2.1 导数 (Derivative)

导数表示函数在某一点的变化率，几何意义是曲线在该点的切线斜率。

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**常用导数公式：**

- $\frac{d}{dx}(x^n) = nx^{n-1}$
- $\frac{d}{dx}(e^x) = e^x$
- $\frac{d}{dx}(\ln x) = \frac{1}{x}$
- $\frac{d}{dx}(\sin x) = \cos x$

### 2.2 偏导数与梯度

对于多变量函数，偏导数表示函数对某一变量的变化率。梯度是所有偏导数组成的向量，指向函数增长最快的方向。

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

```python
# 使用数值方法计算梯度
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """计算函数f在点x处的数值梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 示例：f(x, y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

point = np.array([3.0, 4.0])
print(numerical_gradient(f, point))  # [6. 8.]
# 理论值：梯度 = [2x, 2y] = [6, 8]
```

### 2.3 链式法则

链式法则是反向传播算法的数学基础，用于计算复合函数的导数。

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

::: warning 核心概念
神经网络的反向传播算法本质上就是链式法则的应用。通过链式法则，我们可以高效地计算损失函数对每个参数的梯度。
:::

## 3. 概率论与统计基础

概率论为机器学习中的不确定性建模提供了数学框架，是理解贝叶斯学习、生成模型等的基础。

### 3.1 基本概念

- **概率：** 事件发生可能性的度量，范围[0, 1]
- **条件概率：** $P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **贝叶斯定理：** $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

### 3.2 常见概率分布

| 分布 | 公式 | 应用场景 |
|------|------|----------|
| 伯努利分布 | $P(X=1) = p$ | 二分类问题 |
| 高斯分布 | $f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | 连续变量建模 |
| 多项式分布 | 伯努利分布的推广 | 多分类问题 |

### 3.3 期望、方差与协方差

- **期望（均值）：** $E[X] = \sum_x x \cdot P(X=x)$
- **方差：** $Var(X) = E[(X - E[X])^2]$
- **协方差：** $Cov(X, Y) = E[(X - E[X])(Y - E[Y])]$

```python
import numpy as np
from scipy import stats

# 生成正态分布数据
data = np.random.normal(loc=0, scale=1, size=1000)

# 计算统计量
print(f"均值: {np.mean(data):.4f}")
print(f"方差: {np.var(data):.4f}")
print(f"标准差: {np.std(data):.4f}")

# 正态分布PDF
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)
```

## 4. 最优化基础

机器学习模型训练的本质是求解优化问题，即找到使损失函数最小的参数。

### 4.1 梯度下降法

梯度下降是最常用的优化算法，通过沿着梯度的反方向迭代更新参数。

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

其中 $\eta$ 是学习率，$\nabla L$ 是损失函数的梯度。

```python
import numpy as np

def gradient_descent(gradient_func, initial_params, learning_rate=0.01, epochs=100):
    """基础梯度下降算法"""
    params = initial_params.copy()
    history = [params.copy()]
    
    for _ in range(epochs):
        grad = gradient_func(params)
        params = params - learning_rate * grad
        history.append(params.copy())
    
    return params, history

# 示例：最小化 f(x) = x^2
def gradient(x):
    return 2 * x

x0 = np.array([5.0])
optimal_x, history = gradient_descent(gradient, x0)
print(f"最优解: {optimal_x[0]:.6f}")  # 接近0
```

### 4.2 梯度下降的变体

| 算法 | 特点 |
|------|------|
| 批量梯度下降 (BGD) | 使用全部数据计算梯度，稳定但慢 |
| 随机梯度下降 (SGD) | 每次使用单个样本，快但不稳定 |
| 小批量梯度下降 | 折中方案，最常用 |
| Adam | 自适应学习率，深度学习首选 |

## 5. 本章小结

- **线性代数：** 向量和矩阵运算是数据表示和模型计算的基础
- **微积分：** 导数和梯度用于模型优化，链式法则是反向传播的核心
- **概率统计：** 为不确定性建模提供数学框架
- **最优化：** 梯度下降法是训练模型的核心算法

::: info 推荐学习资源
- 3Blue1Brown《线性代数的本质》视频系列
- 《深度学习》（花书）第2-4章数学基础
- Khan Academy 微积分和概率统计课程
:::
