# 机器学习概述

::: info 本章概述
本章深入介绍机器学习的三大范式（监督学习、无监督学习、强化学习），以及机器学习工作流程和常用算法分类。
:::

## 1. 什么是机器学习

**机器学习（Machine Learning）** 是人工智能的核心分支，它使计算机能够从数据中自动学习规律，无需显式编程。

::: tip 经典定义
**汤姆·米切尔（Tom Mitchell）**：如果一个程序在任务T上的性能P随着经验E的增加而提高，则称该程序从经验E中学习。
:::

举个例子：垃圾邮件过滤器

- **任务T**：将邮件分类为垃圾邮件或正常邮件
- **性能P**：分类准确率
- **经验E**：用户标记的邮件数据

## 2. 监督学习（Supervised Learning）

监督学习是最常见的机器学习类型，使用带有标签的数据进行训练。

### 2.1 基本概念

- **特征（Features）**：输入变量，用于描述样本的属性，通常表示为 X
- **标签（Label）**：目标变量，我们要预测的值，通常表示为 y
- **训练集**：用于训练模型的数据
- **测试集**：用于评估模型性能的数据

### 2.2 问题类型

| 类型 | 描述 | 例子 | 常用算法 |
|------|------|------|----------|
| **回归** | 预测连续数值 | 房价预测、温度预测 | 线性回归、随机森林 |
| **分类** | 预测离散类别 | 垃圾邮件识别、图像分类 | 逻辑回归、SVM、神经网络 |

### 2.3 代码示例

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 准备数据
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2.1, 4.2, 5.8, 8.1, 9.9, 12.1, 14.0, 16.2, 17.9, 20.1])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse:.4f}")
print(f"模型参数: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
```

## 3. 无监督学习（Unsupervised Learning）

无监督学习使用没有标签的数据，目标是发现数据中隐藏的结构和模式。

### 3.1 主要任务

| 任务 | 描述 | 例子 | 常用算法 |
|------|------|------|----------|
| **聚类** | 将相似样本分组 | 客户分群、文档分类 | K-means、DBSCAN、层次聚类 |
| **降维** | 减少特征数量 | 数据可视化、特征提取 | PCA、t-SNE、UMAP |
| **异常检测** | 识别异常样本 | 欺诈检测、故障诊断 | Isolation Forest、One-class SVM |

### 3.2 聚类示例

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-means聚类结果')
plt.savefig('kmeans_result.png')
plt.show()
```

## 4. 强化学习（Reinforcement Learning）

强化学习与监督学习和无监督学习不同，它通过智能体与环境的交互来学习最优策略。

### 4.1 核心概念

- **智能体（Agent）**：学习者，做出决策
- **环境（Environment）**：智能体交互的外部世界
- **状态（State）**：环境的当前情况
- **动作（Action）**：智能体可执行的操作
- **奖励（Reward）**：动作后环境给予的反馈
- **策略（Policy）**：从状态到动作的映射

```
┌──────────┐         动作          ┌──────────┐
│          │ ──────────────────▶  │          │
│  智能体   │                      │   环境    │
│          │ ◀──────────────────  │          │
└──────────┘    状态 + 奖励        └──────────┘
```

### 4.2 典型应用

- **游戏AI**：AlphaGo、Atari游戏、星际争霸
- **机器人控制**：机械臂操作、自主导航
- **推荐系统**：个性化推荐、广告投放
- **自动驾驶**：决策规划

## 5. 机器学习工作流程

一个完整的机器学习项目通常包含以下步骤：

1. **问题定义** → 明确业务目标和评估指标
2. **数据收集** → 获取相关数据
3. **数据探索** → EDA，理解数据特征
4. **数据预处理** → 清洗、转换、特征工程
5. **模型选择** → 根据问题类型选择合适的算法
6. **模型训练** → 在训练集上拟合模型
7. **模型评估** → 在测试集上评估性能
8. **模型调优** → 超参数优化、特征选择
9. **模型部署** → 将模型投入生产环境
10. **监控维护** → 持续监控和更新

## 6. 常用算法概览

### 6.1 线性模型
- 线性回归（Linear Regression）
- 逻辑回归（Logistic Regression）
- 正则化：Ridge、Lasso

### 6.2 树模型
- 决策树（Decision Tree）
- 随机森林（Random Forest）
- 梯度提升树（XGBoost、LightGBM）

### 6.3 支持向量机
- 线性SVM
- 核SVM（RBF、多项式核）

### 6.4 神经网络
- 多层感知机（MLP）
- 卷积神经网络（CNN）
- 循环神经网络（RNN/LSTM）
- Transformer

## 7. 模型评估指标

### 7.1 回归问题

| 指标 | 公式描述 | 特点 |
|------|----------|------|
| MSE | 均方误差 | 对大误差敏感 |
| RMSE | 均方根误差 | 与原始单位相同 |
| MAE | 平均绝对误差 | 对异常值鲁棒 |
| R² | 决定系数 | 解释方差比例 |

### 7.2 分类问题

| 指标 | 描述 | 适用场景 |
|------|------|----------|
| 准确率 | 正确预测的比例 | 类别平衡时 |
| 精确率 | 预测为正中真正为正的比例 | 关注假阳性时 |
| 召回率 | 真正为正中被预测为正的比例 | 关注假阴性时 |
| F1分数 | 精确率和召回率的调和平均 | 综合评估 |
| AUC-ROC | ROC曲线下面积 | 类别不平衡时 |

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# 假设有真实标签和预测标签
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# 计算各项指标
print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(f"精确率: {precision_score(y_true, y_pred):.4f}")
print(f"召回率: {recall_score(y_true, y_pred):.4f}")
print(f"F1分数: {f1_score(y_true, y_pred):.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_true, y_pred))

# 详细报告
print("\n分类报告:")
print(classification_report(y_true, y_pred))
```

## 8. 本章小结

- **监督学习**：从带标签数据学习，用于分类和回归问题
- **无监督学习**：从无标签数据发现结构，用于聚类和降维
- **强化学习**：通过交互学习最优策略
- 机器学习项目包含数据准备、模型训练、评估调优等多个阶段
- 选择合适的评估指标对于模型选择至关重要
