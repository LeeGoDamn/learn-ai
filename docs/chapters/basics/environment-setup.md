# 开发环境搭建

::: info 本章概述
本章将指导你搭建完整的AI开发环境，包括Anaconda、Jupyter Notebook、常用库安装以及GPU配置。
:::

## 1. Python环境管理

### 1.1 为什么需要环境管理？

不同的AI项目可能需要不同版本的Python和库。环境管理工具可以：
- 隔离不同项目的依赖，避免冲突
- 方便复制和共享环境配置
- 便于切换Python版本

### 1.2 Anaconda安装

**Anaconda** 是最流行的Python科学计算发行版，预装了大量常用库。

::: tip 下载安装
1. 访问 [Anaconda官网](https://www.anaconda.com/download)
2. 下载对应操作系统的安装包
3. 运行安装程序，按提示完成安装
4. 建议勾选"添加到环境变量"选项
:::

### 1.3 Conda常用命令

```bash
# 查看版本
conda --version

# 创建新环境
conda create -n ai_env python=3.10

# 激活环境
conda activate ai_env

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 删除环境
conda remove -n ai_env --all

# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

### 1.4 安装常用库

```bash
# 激活环境后安装库
conda activate ai_env

# 基础科学计算库
conda install numpy pandas matplotlib seaborn

# 机器学习库
conda install scikit-learn

# 深度学习框架（二选一）
# PyTorch（推荐）
conda install pytorch torchvision torchaudio -c pytorch

# TensorFlow
conda install tensorflow

# 其他常用库
conda install jupyter jupyterlab
pip install tqdm  # 进度条
```

## 2. Jupyter Notebook

**Jupyter Notebook** 是交互式编程环境，非常适合数据分析和机器学习实验。

### 2.1 启动Jupyter

```bash
# 启动 Jupyter Notebook
jupyter notebook

# 或启动 Jupyter Lab（更现代的界面）
jupyter lab
```

执行后会自动打开浏览器，访问 `http://localhost:8888`

### 2.2 常用快捷键

| 快捷键 | 功能 | 模式 |
|--------|------|------|
| Shift + Enter | 运行当前单元格，跳到下一个 | 通用 |
| Ctrl + Enter | 运行当前单元格 | 通用 |
| Esc | 进入命令模式 | 编辑模式 |
| Enter | 进入编辑模式 | 命令模式 |
| A / B | 在上方/下方插入单元格 | 命令模式 |
| D + D | 删除当前单元格 | 命令模式 |
| M / Y | 切换为Markdown/代码 | 命令模式 |

### 2.3 Notebook最佳实践

- 每个单元格只做一件事，保持简洁
- 使用Markdown单元格添加说明文档
- 定期保存，使用版本控制
- 运行顺序很重要，建议从头到尾运行
- 重要实验结果要截图或导出

## 3. Google Colab（免费GPU）

**Google Colab** 是谷歌提供的免费云端Jupyter环境，无需本地配置，还提供免费GPU。

### 3.1 使用Colab

1. 访问 [Google Colab](https://colab.research.google.com)
2. 登录Google账号
3. 新建笔记本或上传本地.ipynb文件

### 3.2 启用GPU

::: tip 启用GPU加速
1. 菜单栏 → 修改 → 笔记本设置
2. 硬件加速器选择"GPU"或"TPU"
3. 点击保存
:::

```python
# 检查GPU是否可用
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

### 3.3 Colab实用技巧

```python
# 挂载Google Drive（持久存储）
from google.colab import drive
drive.mount('/content/drive')

# 安装额外的库
!pip install transformers

# 下载文件
!wget https://example.com/data.csv

# 查看GPU使用情况
!nvidia-smi
```

## 4. GPU配置（本地）

如果你有NVIDIA GPU，可以配置本地CUDA环境以加速深度学习训练。

### 4.1 检查GPU

```bash
# Windows: 打开命令提示符
nvidia-smi

# 如果能显示GPU信息，说明驱动已安装
```

### 4.2 安装CUDA工具包

1. 确认GPU支持的CUDA版本
2. 安装对应版本的CUDA Toolkit
3. 安装cuDNN库
4. 配置环境变量

::: warning 版本兼容性
PyTorch和TensorFlow对CUDA版本有要求。建议先确认框架需要的CUDA版本，再安装对应版本。

最简单的方式是使用conda安装，它会自动处理依赖：
:::

```bash
# PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

## 5. 常用IDE配置

### 5.1 VS Code

Visual Studio Code是最流行的代码编辑器，对Python和Jupyter有很好的支持。

::: tip 推荐扩展
- **Python** - 微软官方Python支持
- **Jupyter** - 在VS Code中运行Notebook
- **Pylance** - Python语言服务器
- **GitHub Copilot** - AI编程助手
- **GitLens** - Git增强
:::

### 5.2 PyCharm

PyCharm是专业的Python IDE，社区版免费，对大型项目支持较好。

## 6. 验证环境

运行以下代码验证环境是否正确配置：

```python
import sys
print(f"Python版本: {sys.version}")

import numpy as np
print(f"NumPy版本: {np.__version__}")

import pandas as pd
print(f"Pandas版本: {pd.__version__}")

import sklearn
print(f"Scikit-learn版本: {sklearn.__version__}")

import matplotlib
print(f"Matplotlib版本: {matplotlib.__version__}")

# 深度学习框架
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch未安装")

try:
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"GPU可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError:
    print("TensorFlow未安装")

print("\n环境配置完成！可以开始学习了！")
```

## 7. 推荐的项目结构

```
my_ai_project/
├── data/                   # 数据文件
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── notebooks/             # Jupyter笔记本
│   ├── 01_eda.ipynb      # 探索性分析
│   ├── 02_training.ipynb # 模型训练
│   └── 03_evaluation.ipynb
├── src/                   # 源代码
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   └── utils.py
├── models/                # 保存的模型
├── results/               # 结果和图表
├── requirements.txt       # 依赖列表
├── environment.yml        # Conda环境配置
└── README.md             # 项目说明
```

## 8. 本章小结

- 使用Anaconda管理Python环境和依赖
- Jupyter Notebook是交互式实验的首选工具
- Google Colab提供免费GPU，适合没有本地GPU的学习者
- 本地GPU配置需要注意CUDA版本兼容性
- 良好的项目结构有助于代码管理和协作

::: tip 下一步
环境配置完成后，我们将开始学习机器学习的核心算法——从线性回归开始！
:::
