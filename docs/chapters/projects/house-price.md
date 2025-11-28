# 房价预测实战

::: info 项目概述
本项目将使用加州房价数据集，完整演示一个机器学习项目的全流程：数据探索、特征工程、模型训练、评估和优化。
:::

## 1. 项目背景

房价预测是经典的回归问题。我们的目标是根据房屋的各种特征（如位置、面积、房龄等）预测其价格。

### 1.1 数据集介绍

我们使用加州房价数据集（California Housing Dataset），包含以下特征：

| 特征 | 描述 |
|------|------|
| MedInc | 街区居民收入中位数 |
| HouseAge | 房屋年龄中位数 |
| AveRooms | 平均房间数 |
| AveBedrms | 平均卧室数 |
| Population | 街区人口 |
| AveOccup | 平均入住人数 |
| Latitude | 纬度 |
| Longitude | 经度 |

## 2. 数据探索（EDA）

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# 加载数据
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 目标变量：房价中位数（单位：10万美元）

print("数据形状:", df.shape)
print("\n数据预览:")
print(df.head())

print("\n基本统计信息:")
print(df.describe())

print("\n缺失值检查:")
print(df.isnull().sum())
```

### 2.1 数据分布可视化

```python
# 目标变量分布
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['MedHouseVal'], bins=50, edgecolor='black')
plt.xlabel('房价中位数（10万美元）')
plt.ylabel('频数')
plt.title('房价分布')

plt.subplot(1, 2, 2)
df['MedHouseVal'].plot(kind='box')
plt.ylabel('房价中位数')
plt.title('房价箱线图')

plt.tight_layout()
plt.show()

# 特征分布
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(housing.feature_names):
    ax = axes[i // 4, i % 4]
    df[col].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

### 2.2 相关性分析

```python
# 相关性矩阵
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.show()

# 与目标变量的相关性
print("\n与房价的相关性:")
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
```

### 2.3 地理分布可视化

```python
# 地理位置与房价的关系
plt.figure(figsize=(12, 8))
plt.scatter(df['Longitude'], df['Latitude'], 
            c=df['MedHouseVal'], cmap='viridis', 
            alpha=0.5, s=df['Population']/100)
plt.colorbar(label='房价中位数')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('加州房价地理分布')
plt.show()
```

## 3. 数据预处理

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 分离特征和目标
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n标准化后的特征统计:")
print(f"均值: {X_train_scaled.mean(axis=0).round(2)}")
print(f"标准差: {X_train_scaled.std(axis=0).round(2)}")
```

## 4. 模型训练与比较

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

# 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    results = {
        'Model': name,
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'Train Time': train_time
    }
    return results, model

# 训练多个模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results_list = []
trained_models = {}

for name, model in models.items():
    results, trained_model = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test, name
    )
    results_list.append(results)
    trained_models[name] = trained_model
    print(f"{name}: Test R² = {results['Test R²']:.4f}, Test RMSE = {results['Test RMSE']:.4f}")

# 结果汇总
results_df = pd.DataFrame(results_list)
print("\n模型对比:")
print(results_df.to_string(index=False))
```

### 4.1 模型性能可视化

```python
# 模型对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² 分数对比
ax1 = axes[0]
x = range(len(results_df))
width = 0.35
ax1.bar([i - width/2 for i in x], results_df['Train R²'], width, label='Train')
ax1.bar([i + width/2 for i in x], results_df['Test R²'], width, label='Test')
ax1.set_xlabel('模型')
ax1.set_ylabel('R² Score')
ax1.set_title('各模型R²分数对比')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax1.legend()

# RMSE 对比
ax2 = axes[1]
ax2.bar([i - width/2 for i in x], results_df['Train RMSE'], width, label='Train')
ax2.bar([i + width/2 for i in x], results_df['Test RMSE'], width, label='Test')
ax2.set_xlabel('模型')
ax2.set_ylabel('RMSE')
ax2.set_title('各模型RMSE对比')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 5. 特征重要性分析

```python
# 使用随机森林的特征重要性
rf_model = trained_models['Random Forest']

feature_importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('重要性')
plt.title('随机森林特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("特征重要性排名:")
print(feature_importance)
```

## 6. 超参数调优

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# 对随机森林进行超参数调优
param_grid_small = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid_small, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")

# 用最佳模型评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(f"测试集R²: {r2_score(y_test, y_pred):.4f}")
print(f"测试集RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

## 7. 预测结果分析

```python
# 使用最佳模型进行预测
y_pred = best_model.predict(X_test_scaled)

# 预测值 vs 真实值
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值 vs 真实值')

# 残差分布
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布')
plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()

# 残差统计
print(f"残差均值: {residuals.mean():.4f}")
print(f"残差标准差: {residuals.std():.4f}")
```

## 8. 模型保存与加载

```python
import joblib

# 保存模型和标准化器
joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("模型已保存!")

# 加载模型
loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# 使用加载的模型进行预测
new_data = np.array([[8.3, 41, 6.98, 1.02, 322, 2.56, 37.88, -122.23]])
new_data_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)
print(f"预测房价: ${prediction[0] * 100000:.2f}")
```

## 9. 项目总结

::: tip 关键收获
- **数据探索**：理解数据分布和特征关系是建模的基础
- **特征工程**：标准化对某些算法至关重要
- **模型比较**：集成方法（如随机森林）通常优于单一模型
- **超参数调优**：可以进一步提升模型性能
- **模型评估**：关注训练集和测试集的差距，避免过拟合
:::

### 9.1 进一步改进方向

- 尝试更多特征工程（如交叉特征、多项式特征）
- 使用XGBoost或LightGBM等更强的模型
- 进行更细致的超参数调优
- 尝试神经网络模型
- 集成多个模型（Stacking）
