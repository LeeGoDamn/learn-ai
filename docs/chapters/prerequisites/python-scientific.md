# Python科学计算

::: info 本章概述
Python是AI和机器学习领域最流行的编程语言。本章介绍NumPy、Pandas、Matplotlib这三个核心科学计算库的基础用法。
:::

## 1. NumPy - 数值计算基础

NumPy（Numerical Python）是Python科学计算的基础包，提供了高效的多维数组对象和各种数学函数。

### 1.1 创建数组

```python
import numpy as np

# 从列表创建数组
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]

# 创建二维数组
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

# 常用创建函数
zeros = np.zeros((3, 4))      # 全零数组
ones = np.ones((2, 3))        # 全一数组
eye = np.eye(3)               # 单位矩阵
rand = np.random.rand(3, 3)   # 随机数组 [0, 1)
randn = np.random.randn(3, 3) # 标准正态分布

# 等间隔数组
linspace = np.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
arange = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
```

### 1.2 数组属性与索引

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 数组属性
print(arr.shape)    # (3, 3) - 形状
print(arr.dtype)    # int64 - 数据类型
print(arr.size)     # 9 - 元素总数
print(arr.ndim)     # 2 - 维度数

# 索引和切片
print(arr[0, 0])      # 1 - 单个元素
print(arr[0, :])      # [1 2 3] - 第一行
print(arr[:, 1])      # [2 5 8] - 第二列
print(arr[0:2, 1:3])  # 子矩阵

# 布尔索引
print(arr[arr > 5])   # [6 7 8 9]

# 花式索引
print(arr[[0, 2], :]) # 第0行和第2行
```

### 1.3 数组运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 逐元素运算
print(a + b)      # [5 7 9]
print(a * b)      # [4 10 18]
print(a ** 2)     # [1 4 9]
print(np.sqrt(a)) # [1. 1.414 1.732]

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))   # 矩阵乘法
print(A @ B)          # 矩阵乘法（Python 3.5+）
print(A * B)          # 逐元素乘法

# 聚合运算
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr))          # 21 - 总和
print(np.sum(arr, axis=0))  # [5 7 9] - 按列求和
print(np.sum(arr, axis=1))  # [6 15] - 按行求和
print(np.mean(arr))         # 3.5 - 均值
print(np.max(arr))          # 6 - 最大值
print(np.argmax(arr))       # 5 - 最大值索引
```

### 1.4 广播机制

广播（Broadcasting）允许NumPy在形状不同的数组间进行算术运算。

```python
# 广播示例
a = np.array([[1, 2, 3], [4, 5, 6]])  # 形状 (2, 3)
b = np.array([10, 20, 30])             # 形状 (3,)

# b被"广播"成 [[10, 20, 30], [10, 20, 30]]
print(a + b)
# [[11 22 33]
#  [14 25 36]]

# 标量广播
print(a * 2)
# [[ 2  4  6]
#  [ 8 10 12]]
```

::: tip 广播规则
1. 如果两个数组维度数不同，在维度少的数组前面补1
2. 在任一维度上，大小为1的可以与任意大小匹配
3. 两个数组在所有维度上兼容才能广播
:::

## 2. Pandas - 数据处理利器

Pandas提供了DataFrame和Series两种数据结构，非常适合处理表格型数据。

### 2.1 数据结构

```python
import pandas as pd

# Series - 一维数据
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s['a'])  # 1

# DataFrame - 二维表格数据
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['北京', '上海', '广州']
})
print(df)
#       name  age city
# 0    Alice   25   北京
# 1      Bob   30   上海
# 2  Charlie   35   广州

# 从文件读取
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
```

### 2.2 数据选择与过滤

```python
# 创建示例数据
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

# 选择列
print(df['name'])           # 单列，返回Series
print(df[['name', 'age']])  # 多列，返回DataFrame

# 选择行
print(df.iloc[0])           # 按位置索引
print(df.loc[0])            # 按标签索引
print(df.iloc[0:2])         # 切片

# 条件过滤
print(df[df['age'] > 28])   # age大于28的行
print(df[(df['age'] > 25) & (df['salary'] > 55000)])

# 使用query方法
print(df.query('age > 28 and salary > 55000'))
```

### 2.3 数据处理

```python
# 处理缺失值
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8]
})

print(df.isnull().sum())    # 统计缺失值
df_filled = df.fillna(0)    # 填充缺失值
df_dropped = df.dropna()    # 删除含缺失值的行

# 数据转换
df['A'] = df['A'].astype(float)  # 类型转换

# 应用函数
df['A_squared'] = df['A'].apply(lambda x: x**2 if pd.notna(x) else None)

# 分组聚合
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [10, 20, 30, 40]
})
print(df.groupby('category')['value'].mean())
# category
# A    15.0
# B    35.0

# 排序
df_sorted = df.sort_values('value', ascending=False)
```

### 2.4 合并数据

```python
# 合并DataFrame
df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
df2 = pd.DataFrame({'id': [1, 2], 'score': [90, 85]})

# merge - 类似SQL JOIN
merged = pd.merge(df1, df2, on='id')

# concat - 拼接
df3 = pd.concat([df1, df1], axis=0)  # 垂直拼接
df4 = pd.concat([df1, df2], axis=1)  # 水平拼接
```

## 3. Matplotlib - 数据可视化

Matplotlib是Python最基础的绘图库，可以创建各种静态、动态和交互式图表。

### 3.1 基础绘图

```python
import matplotlib.pyplot as plt
import numpy as np

# 折线图
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-')
plt.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('三角函数图像')
plt.legend()
plt.grid(True)
plt.savefig('plot.png')
plt.show()
```

### 3.2 常用图表类型

```python
# 散点图
plt.scatter(x, y, c='blue', alpha=0.5)

# 柱状图
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)

# 直方图
data = np.random.randn(1000)
plt.hist(data, bins=30, edgecolor='black')

# 饼图
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

# 热力图
data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot')
plt.colorbar()
```

### 3.3 子图布局

```python
# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 在各子图中绘制
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Cos')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('Tan')

axes[1, 1].plot(x, np.exp(-x))
axes[1, 1].set_title('Exp')

plt.tight_layout()
plt.show()
```

::: tip 可视化最佳实践
- 始终添加标题、轴标签和图例
- 选择合适的图表类型展示数据
- 使用适当的配色方案
- 保持图表简洁，避免信息过载
:::

## 4. Seaborn - 统计可视化

Seaborn是基于Matplotlib的高级可视化库，特别适合统计数据可视化。

```python
import seaborn as sns

# 使用内置数据集
tips = sns.load_dataset('tips')

# 分布图
sns.histplot(tips['total_bill'], kde=True)

# 散点图带回归线
sns.regplot(x='total_bill', y='tip', data=tips)

# 箱线图
sns.boxplot(x='day', y='total_bill', data=tips)

# 热力图（相关性矩阵）
corr = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# 成对关系图
sns.pairplot(tips, hue='sex')
```

## 5. 实战练习

::: details 练习：探索性数据分析

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 创建模拟数据
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples) * 2,
    'target': np.random.randint(0, 2, n_samples)
})

# 添加一些特征间的关系
data['feature3'] = data['feature1'] * 2 + np.random.randn(n_samples) * 0.5

# 2. 数据探索
print("数据形状:", data.shape)
print("\n数据统计:")
print(data.describe())
print("\n缺失值:")
print(data.isnull().sum())

# 3. 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 特征分布
axes[0, 0].hist(data['feature1'], bins=20, edgecolor='black')
axes[0, 0].set_title('Feature1 分布')

# 散点图
axes[0, 1].scatter(data['feature1'], data['feature3'], 
                   c=data['target'], cmap='coolwarm', alpha=0.6)
axes[0, 1].set_xlabel('feature1')
axes[0, 1].set_ylabel('feature3')
axes[0, 1].set_title('Feature1 vs Feature3')

# 箱线图
data.boxplot(column=['feature1', 'feature2', 'feature3'], ax=axes[1, 0])
axes[1, 0].set_title('特征箱线图')

# 相关性热力图
corr = data.corr()
im = axes[1, 1].imshow(corr, cmap='coolwarm')
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels(corr.columns)
axes[1, 1].set_yticklabels(corr.columns)
axes[1, 1].set_title('相关性矩阵')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('eda_example.png', dpi=150)
plt.show()
```
:::

## 6. 本章小结

- **NumPy：** 高效的数组运算，是所有科学计算库的基础
- **Pandas：** 强大的数据处理能力，适合表格数据的清洗和分析
- **Matplotlib：** 基础可视化工具，支持各种图表类型
- **Seaborn：** 高级统计可视化，更美观的默认样式
