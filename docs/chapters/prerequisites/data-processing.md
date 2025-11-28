# 数据处理基础

::: info 本章概述
在机器学习中，"数据决定上限，模型决定下限"。本章介绍数据预处理和特征工程的基本方法，这些技能在实际项目中至关重要。
:::

## 1. 数据预处理概述

真实世界的数据往往是"脏"的，需要经过清洗和转换才能用于模型训练。数据预处理的质量直接影响模型的性能。

### 1.1 数据质量问题

| 问题类型 | 描述 | 处理方法 |
|----------|------|----------|
| 缺失值 | 数据中存在空值或NaN | 删除、填充、插值 |
| 异常值 | 极端偏离正常范围的值 | 检测并处理或删除 |
| 重复值 | 相同的记录出现多次 | 去重 |
| 类型错误 | 数据类型不正确 | 类型转换 |
| 格式不一致 | 同类数据格式不统一 | 标准化格式 |

## 2. 缺失值处理

### 2.1 检测缺失值

```python
import pandas as pd
import numpy as np

# 创建包含缺失值的数据
df = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'salary': [50000, np.nan, 60000, np.nan, 80000],
    'city': ['北京', '上海', None, '广州', '深圳']
})

# 检测缺失值
print(df.isnull())           # 返回布尔矩阵
print(df.isnull().sum())     # 每列缺失值数量
print(df.isnull().sum().sum())  # 总缺失值数量

# 缺失值比例
missing_ratio = df.isnull().sum() / len(df)
print(missing_ratio)
```

### 2.2 处理缺失值

```python
# 方法1：删除
df_dropped = df.dropna()           # 删除任何含缺失值的行
df_dropped = df.dropna(subset=['age'])  # 只考虑特定列

# 方法2：填充固定值
df['age'] = df['age'].fillna(0)
df['city'] = df['city'].fillna('未知')

# 方法3：统计量填充
df['age'] = df['age'].fillna(df['age'].mean())      # 均值
df['salary'] = df['salary'].fillna(df['salary'].median())  # 中位数
df['city'] = df['city'].fillna(df['city'].mode()[0])  # 众数

# 方法4：前向/后向填充（时序数据）
df['value'] = df['value'].fillna(method='ffill')  # 前向填充
df['value'] = df['value'].fillna(method='bfill')  # 后向填充

# 方法5：插值
df['value'] = df['value'].interpolate(method='linear')
```

::: warning 注意事项
选择缺失值处理方法时要考虑：
1. 缺失的原因（随机缺失 vs 系统性缺失）
2. 缺失比例（高于30%可能需要删除该特征）
3. 业务含义（有时缺失本身就是有意义的信息）
:::

## 3. 异常值处理

### 3.1 检测异常值

```python
import numpy as np
import pandas as pd

# 创建示例数据
np.random.seed(42)
data = np.random.randn(100) * 10 + 50
data[0] = 200  # 添加异常值
data[50] = -50

df = pd.DataFrame({'value': data})

# 方法1：统计方法（3σ原则）
mean = df['value'].mean()
std = df['value'].std()
outliers_3sigma = df[(df['value'] < mean - 3*std) | (df['value'] > mean + 3*std)]
print(f"3σ异常值数量: {len(outliers_3sigma)}")

# 方法2：IQR方法（箱线图原理）
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = df[(df['value'] < lower) | (df['value'] > upper)]
print(f"IQR异常值数量: {len(outliers_iqr)}")
```

### 3.2 处理异常值

```python
# 方法1：删除异常值
df_cleaned = df[(df['value'] >= lower) & (df['value'] <= upper)]

# 方法2：截断（Winsorization）
df['value_clipped'] = df['value'].clip(lower=lower, upper=upper)

# 方法3：替换为边界值
df.loc[df['value'] < lower, 'value'] = lower
df.loc[df['value'] > upper, 'value'] = upper

# 方法4：用缺失值代替，再填充
df.loc[df['value'] < lower, 'value'] = np.nan
df['value'] = df['value'].fillna(df['value'].median())
```

## 4. 特征缩放

不同特征的量纲可能差异很大，这会影响某些算法（如梯度下降、KNN、SVM）的性能。

### 4.1 标准化（Standardization）

将数据转换为均值为0，标准差为1的分布：

```python
from sklearn.preprocessing import StandardScaler

# 创建示例数据
data = np.array([[1, 10000], [2, 20000], [3, 30000], [4, 40000]])
df = pd.DataFrame(data, columns=['feature1', 'feature2'])

# 使用StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print("原始数据:")
print(df)
print("\n标准化后:")
print(pd.DataFrame(scaled_data, columns=['feature1', 'feature2']))
```

### 4.2 归一化（Min-Max Scaling）

将数据缩放到[0, 1]范围：

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

print("归一化后:")
print(pd.DataFrame(normalized_data, columns=['feature1', 'feature2']))
# 所有值都在[0, 1]范围内
```

::: tip 何时使用哪种方法？
- **标准化：** 数据近似正态分布，或算法假设数据是正态分布（如逻辑回归、神经网络）
- **归一化：** 需要有界数据，或使用距离计算的算法（如KNN、神经网络的图像输入）
- **不缩放：** 决策树、随机森林等基于规则的算法对特征缩放不敏感
:::

## 5. 类别特征编码

机器学习算法通常需要数值输入，因此需要将类别特征转换为数值。

### 5.1 标签编码（Label Encoding）

```python
from sklearn.preprocessing import LabelEncoder

# 适用于有序类别
df = pd.DataFrame({
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])
print(df)
# small -> 2, medium -> 1, large -> 0（按字母顺序）

# 查看映射
print(dict(zip(le.classes_, range(len(le.classes_)))))
```

### 5.2 独热编码（One-Hot Encoding）

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 适用于无序类别
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue']
})

# 使用pandas
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
print(df_encoded)
#    color_blue  color_green  color_red
# 0           0            0          1
# 1           1            0          0
# 2           0            1          0
# ...

# 使用sklearn
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['color']])
print(encoded)
```

### 5.3 目标编码（Target Encoding）

```python
# 用于高基数类别特征（类别数量很多）
df = pd.DataFrame({
    'city': ['北京', '上海', '北京', '广州', '上海', '北京'],
    'target': [1, 0, 1, 0, 1, 1]
})

# 计算每个类别的目标均值
target_mean = df.groupby('city')['target'].mean()
df['city_encoded'] = df['city'].map(target_mean)
print(df)
```

## 6. 特征工程入门

特征工程是利用领域知识从原始数据中创造新特征的过程，好的特征可以显著提升模型性能。

### 6.1 特征创建

```python
import pandas as pd
import numpy as np

# 原始数据
df = pd.DataFrame({
    'length': [10, 20, 15, 25],
    'width': [5, 10, 8, 12],
    'date': pd.to_datetime(['2024-01-15', '2024-03-20', '2024-06-10', '2024-12-25'])
})

# 数值特征组合
df['area'] = df['length'] * df['width']          # 面积
df['perimeter'] = 2 * (df['length'] + df['width'])  # 周长
df['ratio'] = df['length'] / df['width']         # 长宽比

# 日期特征提取
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek       # 0=周一
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

print(df)
```

### 6.2 特征分箱（Binning）

```python
# 将连续变量离散化
df = pd.DataFrame({
    'age': [22, 35, 45, 18, 60, 28, 55]
})

# 等宽分箱
df['age_bin_equal'] = pd.cut(df['age'], bins=3, labels=['young', 'middle', 'old'])

# 自定义分箱
bins = [0, 25, 40, 60, 100]
labels = ['青年', '中年', '中老年', '老年']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

# 等频分箱（每个箱的样本数相近）
df['age_quantile'] = pd.qcut(df['age'], q=3, labels=['low', 'medium', 'high'])

print(df)
```

### 6.3 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 只有前两个特征有用

# 方法1：基于统计的特征选择
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
print(f"选择的特征索引: {selector.get_support(indices=True)}")

# 方法2：基于模型的特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 特征重要性
importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(10)],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

## 7. 数据预处理流水线

在实际项目中，通常使用Pipeline将多个预处理步骤组合起来。

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 定义数值和类别特征
numeric_features = ['age', 'salary']
categorical_features = ['city', 'gender']

# 数值特征处理流水线
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 填充缺失值
    ('scaler', StandardScaler())                     # 标准化
])

# 类别特征处理流水线
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 完整的机器学习流水线
from sklearn.linear_model import LogisticRegression

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 使用流水线
# full_pipeline.fit(X_train, y_train)
# predictions = full_pipeline.predict(X_test)
```

::: danger 重要提示
预处理操作必须先在训练集上fit，然后用同样的参数transform测试集。使用Pipeline可以避免数据泄露（data leakage）问题。
:::

## 8. 本章小结

- **缺失值处理：** 根据数据特点选择删除、填充或插值
- **异常值处理：** 使用统计方法检测，选择合适的处理策略
- **特征缩放：** 标准化和归一化是最常用的方法
- **类别编码：** 标签编码用于有序类别，独热编码用于无序类别
- **特征工程：** 创造新特征、分箱、特征选择等技术
- **Pipeline：** 组织预处理步骤，确保一致性
