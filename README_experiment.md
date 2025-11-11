# 欺诈检测模型对比实验框架

## 📖 项目简介

这是一个全面的欺诈检测模型对比实验框架，用于系统性地比较多种机器学习和深度学习方法在不同数据集上的表现。

## 🎯 实验目标

- 比较**12种不同类别**的机器学习算法
- 在**7个不同**的欺诈检测数据集上评估
- 提供**横向和纵向**的全面对比分析
- 生成详细的性能报告和可视化

## 📊 数据集

| 数据集 | 训练集 | 测试集 | 特征数 | 欺诈率 | 特点 |
|--------|--------|--------|--------|--------|------|
| IEEE | 100K | 100K | 81 | 0.03% | 高维PCA特征 |
| col14_behave | 100K | 59K | 15 | 6.8% | 行为特征 |
| col16_raw | 100K | 24K | 14 | 4.9% | 电商交易 |
| creditCardPCA | 100K | 57K | 34 | 0.17% | PCA降维 |
| creditCardTransaction | 100K | 100K | 13 | 1.0% | 地理特征 |
| counterfeit_products | 4K | 1K | 16 | 29.4% | 产品真伪 |
| counterfeit_transactions | 2.4K | 600 | 19 | 24.4% | 交易真伪 |

## 🤖 模型方法

### 监督学习 (Supervised Learning)
1. **Logistic Regression** - 线性baseline
2. **Random Forest** - 集成学习
3. **XGBoost** - 梯度提升
4. **LightGBM** - 高效梯度提升
5. **MLP** - 多层感知机
6. **KNN** - K近邻

### 降维+分类 (Dimensionality Reduction + Classification)
7. **PCA + SVM** - 主成分分析 + 支持向量机
8. **PCA + LR** - 主成分分析 + 逻辑回归

### 无监督/异常检测 (Unsupervised / Anomaly Detection)
9. **Isolation Forest** - 孤立森林
10. **One-Class SVM** - 单类支持向量机
11. **Autoencoder** - 自编码器

## 📈 评估指标

- **Accuracy** - 准确率
- **Precision** - 精确率
- **Recall** - 召回率
- **F1-Score** - F1分数
- **ROC-AUC** - ROC曲线下面积
- **PR-AUC** - PR曲线下面积
- **Training Time** - 训练时间
- **Inference Time** - 推理时间

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 验证环境
python src/test_environment.py
```

### 2. 运行实验

```bash
# 打开Jupyter Notebook
jupyter notebook src/experiment.ipynb
```

或者在VS Code中直接打开 `experiment.ipynb`

### 3. 按顺序运行所有单元格

实验会自动：
- ✅ 加载和预处理所有数据集
- ✅ 训练所有模型
- ✅ 评估性能指标
- ✅ 生成可视化图表
- ✅ 输出分析报告

## 📁 项目结构

```
FYP/
├── data/                          # 数据集目录
│   ├── IEEE/
│   ├── col14_behave/
│   ├── col16_raw/
│   ├── counterfeit_products/
│   ├── counterfeit_transactions/
│   ├── creditCardPCA/
│   └── creditCardTransaction/
├── json/                          # 数据集元信息
│   └── dataset_clean_summary.json
├── src/                           # 源代码
│   ├── experiment.ipynb          # 主实验notebook
│   ├── test_environment.py       # 环境测试脚本
│   └── quick_start.md            # 快速入门指南
├── results/                       # 实验结果
│   └── experiment_results.csv    # 详细结果表格
├── requirements.txt               # Python依赖
└── README_experiment.md          # 本文档
```

## 🔬 实验流程

### 阶段1: 数据准备
- 加载训练集和测试集
- 识别特征类型（数值/类别/ID）
- 类别特征编码
- 数值特征标准化

### 阶段2: 模型训练
- 监督学习模型（使用全部数据）
- 无监督模型（只使用正常样本）
- 自动处理类别不平衡

### 阶段3: 模型评估
- 在测试集上预测
- 计算多个评估指标
- 记录时间性能

### 阶段4: 结果分析
- 生成对比图表
- 汇总最佳模型
- 提供使用建议

## 📊 可视化输出

实验会生成以下可视化：

1. **F1-Score对比图** - 各模型在不同数据集上的F1分数
2. **ROC-AUC对比图** - AUC性能对比
3. **指标热力图** - 所有指标的热力图矩阵
4. **时间性能图** - 训练和推理时间对比
5. **数据集维度对比** - 不同数据集上的模型表现

## 💡 使用建议

### 场景1: 快速测试
```python
# 只测试一个小数据集
DATASETS = ['counterfeit_products']  # 修改第6节的数据集列表
```

### 场景2: 跳过慢速模型
```python
# 在大数据集上跳过KNN、SVM等慢速模型
runner.run_all_models(X_train, y_train, X_test, y_test, 
                     dataset_name, skip_slow=True)
```

### 场景3: 自定义模型
```python
# 在ExperimentRunner类中添加新方法
def run_your_model(self, X_train, y_train, X_test, y_test, dataset_name):
    # 实现你的模型
    pass
```

## 📈 预期结果

根据数据集特性，预期性能排名：

### 高性能模型（F1-Score）
1. XGBoost / LightGBM
2. Random Forest
3. MLP

### 高效模型（速度）
1. Logistic Regression
2. Isolation Forest
3. LightGBM

### 无监督方法
- Isolation Forest（最快）
- Autoencoder（最准确）
- One-Class SVM（适合小数据）

## ⚠️ 注意事项

1. **内存要求**: 运行所有数据集需要至少8GB内存
2. **GPU加速**: MLP和Autoencoder会使用GPU（如果可用）
3. **运行时间**: 完整实验约需1-3小时（取决于硬件）
4. **大数据集**: IEEE等数据集较大，部分慢速模型会自动跳过

## 🐛 故障排除

### 问题1: 导入错误
```bash
# 确保所有包都已安装
pip install --upgrade pip
pip install -r requirements.txt
```

### 问题2: CUDA错误
```python
# 模型会自动回退到CPU，不影响实验
# 如需GPU加速，确保安装正确的PyTorch版本
```

### 问题3: 内存不足
```python
# 减少数据集数量或使用采样
X_train = X_train.sample(n=10000, random_state=42)
```

## 📚 相关文档

- [快速入门指南](src/quick_start.md)
- [数据集详细信息](json/dataset_clean_summary.json)

## 🎓 参考文献

- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- Isolation Forest: Liu et al. (2008)
- Deep Learning: Goodfellow et al. (2016)

## 📊 示例输出

实验完成后，你将得到类似这样的结果：

```
🏆 最佳模型总结
==================================
最佳 ACCURACY      : XGBoost        (数据集: IEEE, 值: 0.9987)
最佳 PRECISION     : LightGBM       (数据集: col14_behave, 值: 0.9234)
最佳 RECALL        : Random Forest  (数据集: counterfeit_products, 值: 0.8912)
最佳 F1_SCORE      : XGBoost        (数据集: col16_raw, 值: 0.8456)
最佳 ROC_AUC       : XGBoost        (数据集: IEEE, 值: 0.9876)
```

## 🤝 贡献

欢迎提出改进建议！如果你：
- 发现了bug
- 有新的模型想要添加
- 想要改进可视化
- 有其他建议

请随时提出！

## 📝 许可

本项目仅用于学术研究和教育目的。

---

**祝实验顺利！如有问题，请查看 `quick_start.md` 或运行 `test_environment.py` 进行诊断。** 🚀
