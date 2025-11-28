# 线性回归

::: info 本章概述
线性回归是最基础的机器学习算法，理解它对于掌握更复杂的模型至关重要。本章将从原理讲起，手把手实现梯度下降，并使用sklearn进行实战。
:::

## 1. 什么是线性回归

线性回归是一种用于预测连续数值的监督学习算法。它假设输入特征和输出之间存在线性关系。

### 1.1 一元线性回归

最简单的情况是只有一个特征的线性回归：

$$y = wx + b$$

- $y$ - 预测值（目标变量）
- $x$ - 输入特征
- $w$ - 权重（斜率）
- $b$ - 偏置（截距）

### 1.2 多元线性回归

当有多个特征时：

$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \mathbf{w}^T\mathbf{x} + b$$

用矩阵形式表示更简洁。

## 2. 损失函数

为了训练模型，我们需要一个衡量预测好坏的指标——损失函数。线性回归通常使用**均方误差（MSE）**：

$$L(w, b) = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 = \frac{1}{m}\sum_{i=1}^{m}(y_i - (wx_i + b))^2$$

其中 $m$ 是样本数量，$y_i$ 是真实值，$\hat{y}_i$ 是预测值。

::: tip 为什么用平方？
- 平方可以放大大误差，使模型更关注大偏差
- 平方使正负误差不会相互抵消
- 平方函数可导，便于优化
:::

## 3. 梯度下降

梯度下降是求解最优参数的核心算法。其思想是：沿着损失函数梯度的反方向更新参数，逐步找到最小值。

### 3.1 算法步骤

1. 随机初始化参数 $w$ 和 $b$
2. 计算损失函数对参数的梯度
3. 更新参数：$w = w - \eta \frac{\partial L}{\partial w}$
4. 重复步骤2-3直到收敛

### 3.2 梯度计算

对于一元线性回归，梯度公式为：

$$\frac{\partial L}{\partial w} = \frac{2}{m}\sum_{i=1}^{m}(wx_i + b - y_i)x_i$$

$$\frac{\partial L}{\partial b} = \frac{2}{m}\sum_{i=1}^{m}(wx_i + b - y_i)$$

## 4. 从零实现线性回归

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + 噪声

# 可视化数据
plt.scatter(X, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('训练数据')
plt.show()

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.losses = []
    
    def fit(self, X, y):
        m, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.w) + self.b
            
            # 计算损失
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # 计算梯度
            dw = -(2/m) * np.dot(X.T, (y - y_pred))
            db = -(2/m) * np.sum(y - y_pred)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b

# 训练模型
model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

print(f"\n学习到的参数:")
print(f"w = {model.w[0][0]:.4f} (真实值: 3)")
print(f"b = {model.b:.4f} (真实值: 4)")

# 绘制结果
plt.figure(figsize=(12, 4))

# 拟合曲线
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, label='数据点')
plt.plot(X, model.predict(X), color='red', label='拟合线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归拟合结果')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('训练损失曲线')

plt.tight_layout()
plt.show()
```

## 5. 使用sklearn实现

在实际项目中，我们通常使用成熟的库而不是从零实现：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 生成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"权重 w: {model.coef_[0]:.4f}")
print(f"偏置 b: {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```

## 6. 多元线性回归实战

使用加州房价数据集进行多特征回归：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# 加载加州房价数据集
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("特征名称:", housing.feature_names)
print("数据形状:", X.shape)
print("\n数据预览:")
print(X.head())

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 预测和评估
y_pred = model.predict(X_test_scaled)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 特征重要性
importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print("\n特征系数（绝对值排序）:")
print(importance)
```

## 7. 正则化

当特征很多时，模型容易过拟合。正则化通过限制参数大小来防止过拟合。

### 7.1 Ridge回归（L2正则化）

$$L = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^{n}w_j^2$$

### 7.2 Lasso回归（L1正则化）

$$L = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^{n}|w_j|$$

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R²: {ridge.score(X_test_scaled, y_test):.4f}")

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
print(f"Lasso R²: {lasso.score(X_test_scaled, y_test):.4f}")

# ElasticNet（L1和L2的组合）
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
print(f"ElasticNet R²: {elastic.score(X_test_scaled, y_test):.4f}")
```

::: tip 正则化选择
- **Ridge**：特征多且都有一定贡献时使用
- **Lasso**：需要特征选择时使用（会使部分系数变为0）
- **ElasticNet**：结合两者优点
:::

## 8. 学习率的影响

| 学习率 | 效果 | 问题 |
|--------|------|------|
| 太小（0.0001） | 收敛很慢 | 需要很多迭代 |
| 适中（0.01-0.1） | 稳定收敛 | 理想选择 |
| 太大（1.0） | 不收敛 | 损失震荡或发散 |

## 9. 本章小结

- **线性回归** 假设特征和目标之间存在线性关系
- **损失函数**（MSE）衡量预测与真实值的差距
- **梯度下降** 通过迭代更新参数来最小化损失
- **学习率** 控制每次更新的步长，需要适当选择
- **正则化**（Ridge/Lasso）防止过拟合
- 实际项目中使用sklearn等成熟库，但理解原理很重要
