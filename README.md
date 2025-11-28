# AI学习之路 - 人工智能入门教程

面向软件工程专业毕业生的中文人工智能入门学习教程网站。

## ✨ 特性

- **内容与代码分离**：使用Markdown编写内容，方便更新维护
- **VitePress驱动**：现代化静态站点生成器，构建速度快
- **数学公式支持**：集成MathJax渲染数学公式
- **代码高亮**：自动语法高亮，支持多种编程语言
- **响应式设计**：支持桌面和移动端浏览
- **全文搜索**：内置本地搜索功能

## 📚 教程大纲

### 第0章：前置知识
- 数学基础 - 线性代数、微积分、概率论
- Python科学计算 - NumPy、Pandas、Matplotlib
- 数据处理基础 - 数据预处理、特征工程

### 第1章：AI基础概念
- 什么是人工智能 - AI定义、历史与分类
- 机器学习概述 - 监督学习、无监督学习、强化学习
- 开发环境搭建 - Anaconda、Jupyter、GPU配置

### 第2章：机器学习入门
- 线性回归 - 模型原理、梯度下降、代码实现

### 第3章：深度学习入门
- 神经网络基础 - 感知机、激活函数、反向传播

### 第4章：实战项目
- 房价预测 - 完整的回归任务流程

## 🚀 快速开始

### 安装依赖

```bash
npm install
```

### 本地开发

```bash
npm run docs:dev
```

访问 http://localhost:5173 预览网站。

### 构建生产版本

```bash
npm run docs:build
```

构建产物在 `docs/.vitepress/dist` 目录。

### 预览构建结果

```bash
npm run docs:preview
```

## 📁 项目结构

```
learn-ai/
├── docs/                          # 文档源文件
│   ├── .vitepress/               # VitePress配置
│   │   └── config.mts            # 站点配置
│   ├── chapters/                 # 教程章节
│   │   ├── prerequisites/        # 前置知识
│   │   ├── basics/               # AI基础
│   │   ├── machine-learning/     # 机器学习
│   │   ├── deep-learning/        # 深度学习
│   │   └── projects/             # 实战项目
│   ├── public/                   # 静态资源
│   ├── index.md                  # 首页
│   ├── outline.md                # 学习大纲
│   └── resources.md              # 推荐资源
├── package.json
└── README.md
```

## 📝 内容更新

所有教程内容都在 `docs/chapters/` 目录下，使用Markdown格式编写。

### 添加新章节

1. 在对应目录创建新的 `.md` 文件
2. 在 `docs/.vitepress/config.mts` 中更新侧边栏配置
3. 运行 `npm run docs:dev` 预览

### Markdown增强功能

```markdown
# 提示框
::: info 信息
这是一个信息提示
:::

::: tip 提示
这是一个提示
:::

::: warning 警告
这是一个警告
:::

# 数学公式
行内公式: $E = mc^2$

块级公式:
$$
\frac{\partial L}{\partial w} = \frac{2}{m}\sum_{i=1}^{m}(wx_i + b - y_i)x_i
$$

# 代码块
```python
import numpy as np
print("Hello, AI!")
```
```

## 🔗 参考资源

- [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
- [李沐《动手学深度学习》](https://d2l.ai/zh/)
- [Fast.ai深度学习课程](https://www.fast.ai/)
- [VitePress文档](https://vitepress.dev/)

## 📄 许可证

MIT License