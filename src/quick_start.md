# 模型训练实验快速入门指南

## 📋 概述

本实验框架提供了一个完整的欺诈检测模型对比系统，包含：
- 7个不同的数据集
- 12种不同类别的机器学习算法
- 完整的训练、评估和可视化流程

## 🚀 快速开始

### 1. 环境准备

确保安装了必要的依赖包：

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn
```

### 2. 运行完整实验

打开 `experiment.ipynb`，按顺序运行所有单元格：

1. **第1节**: 导入库和配置
2. **第2节**: 数据加载和预处理
3. **第3节**: 模型定义
4. **第4节**: 训练和评估框架
5. **第5节**: 结果可视化
6. **第6节**: 运行完整实验（这一步会花费较长时间）
7. **第7节**: 查看结果分析
8. **第8节**: 深入分析单个数据集（可选）
9. **第9节**: 查看总结和建议

### 3. 快速测试单个数据集

如果想快速测试，可以只运行单个数据集：

```python
# 选择一个较小的数据集测试
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))

# 创建实验运行器
runner = ExperimentRunner()

# 加载数据
loader = DatasetLoader('counterfeit_products')  # 小数据集，训练快
train_df, test_df = loader.load_data()
X_train, X_test, y_train, y_test, _ = loader.preprocess(train_df, test_df)

# 运行所有模型
models = runner.run_all_models(X_train, y_train, X_test, y_test, 
                               'counterfeit_products', skip_slow=False)

# 查看结果
results_df = runner.evaluator.get_results_df()
print(results_df)
```

## 📊 数据集说明

### 大型数据集（10万样本）
- **IEEE**: 81特征，高维PCA特征，极度不平衡
- **col14_behave**: 15特征，包含类别特征
- **col16_raw**: 14特征，电商交易数据
- **creditCardPCA**: 34特征，PCA处理的信用卡数据
- **creditCardTransaction**: 13特征，信用卡交易数据

### 小型数据集
- **counterfeit_products**: 4K训练/1K测试，16特征，产品真伪检测
- **counterfeit_transactions**: 2.4K训练/600测试，19特征，交易真伪检测

## 🤖 模型说明

### 监督学习方法
1. **Logistic Regression**: 线性baseline，速度快
2. **Random Forest**: 集成学习，可解释性强
3. **XGBoost**: 强大的梯度提升，处理不平衡数据好
4. **LightGBM**: 更快的梯度提升实现
5. **MLP**: 深度学习方法，需要GPU加速
6. **KNN**: 基于距离的方法，适合小数据集

### 降维+分类
7. **PCA+SVM**: 线性降维+支持向量机
8. **PCA+LR**: 线性降维+逻辑回归

### 无监督/异常检测
9. **Isolation Forest**: 快速异常检测
10. **One-Class SVM**: 单类分类，只用正常样本训练
11. **Autoencoder**: 深度学习异常检测

## 📈 评估指标

- **Accuracy**: 整体准确率
- **Precision**: 查准率（预测为欺诈中真正是欺诈的比例）
- **Recall**: 查全率（所有欺诈中被检测出来的比例）
- **F1-Score**: Precision和Recall的调和平均
- **ROC-AUC**: ROC曲线下面积
- **PR-AUC**: Precision-Recall曲线下面积
- **训练时间**: 模型训练耗时
- **推理时间**: 模型预测耗时

## 💡 使用建议

### 选择合适的数据集
- **快速测试**: 使用 `counterfeit_products` 或 `counterfeit_transactions`
- **完整评估**: 运行所有数据集
- **特定场景**: 根据你的应用场景选择相似的数据集

### 选择合适的模型
- **追求性能**: XGBoost, LightGBM, Random Forest
- **追求速度**: Logistic Regression, Isolation Forest
- **无标签数据**: Isolation Forest, One-Class SVM, Autoencoder
- **高维数据**: PCA+SVM, PCA+LR, MLP

### 处理类别不平衡
所有监督学习模型都已配置了类别权重平衡：
- `class_weight='balanced'` (sklearn模型)
- `scale_pos_weight` (XGBoost, LightGBM)

## 📁 输出文件

实验完成后会生成：
- `results/experiment_results.csv`: 所有模型的详细结果
- Notebook中的可视化图表

## 🔧 自定义配置

### 修改模型超参数

在 `ExperimentRunner` 类中修改各个模型的参数：

```python
def run_xgboost(self, X_train, y_train, X_test, y_test, dataset_name):
    model = xgb.XGBClassifier(
        n_estimators=200,  # 增加树的数量
        max_depth=8,       # 增加深度
        learning_rate=0.05,# 降低学习率
        # ... 其他参数
    )
```

### 添加新模型

在 `ExperimentRunner` 类中添加新的方法：

```python
def run_your_model(self, X_train, y_train, X_test, y_test, dataset_name):
    print("\n🚀 训练 Your Model...")
    start_time = time.time()
    
    # 训练你的模型
    model = YourModel()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    # 评估
    result = self.evaluator.evaluate_supervised(
        y_test, y_pred, y_pred_proba, 'Your Model', dataset_name,
        train_time, inference_time
    )
    self.evaluator.print_result(result)
    return model
```

然后在 `run_all_models` 方法中调用它。

## ⚠️ 注意事项

1. **大数据集**: IEEE、col14_behave等数据集较大，训练时间较长
2. **GPU加速**: MLP和Autoencoder会自动使用GPU（如果可用）
3. **内存占用**: 运行所有模型可能需要较大内存
4. **慢速模型**: KNN、PCA+SVM、One-Class SVM在大数据集上会被自动跳过

## 🐛 常见问题

### Q: 导入错误
A: 确保所有依赖包都已安装：`pip install -r requirements.txt`

### Q: CUDA/GPU错误
A: MLP和Autoencoder会自动切换到CPU，不影响其他模型

### Q: 内存不足
A: 减少数据集数量或使用 `skip_slow=True` 跳过慢速模型

### Q: 结果不理想
A: 检查数据预处理、调整模型超参数、尝试特征工程

## 📚 扩展阅读

- XGBoost文档: https://xgboost.readthedocs.io/
- scikit-learn文档: https://scikit-learn.org/
- PyTorch文档: https://pytorch.org/docs/

## 🤝 贡献

如果你有改进建议或发现bug，欢迎提出！
