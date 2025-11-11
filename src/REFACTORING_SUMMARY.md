# 实验代码重构总结 / Experiment Code Refactoring Summary

## 📋 重构概览 / Overview

已完成对实验代码的全面重构,将原始的单文件Jupyter notebook风格代码重构为模块化、专业的Python项目结构。

The experiment code has been fully refactored from a single Jupyter notebook-style file into a modular, professional Python project structure.

## ✅ 已满足的需求 / Requirements Met

### 1. 全英文输出 / All English Outputs ✓
- ✅ 所有日志消息使用英文
- ✅ 所有图表标题和标签使用英文  
- ✅ CSV文件列名使用英文
- ✅ 可视化图片中的所有文本使用英文

### 2. 实时日志系统 / Real-time Logging System ✓
- ✅ 使用Python logging模块
- ✅ 带时间戳的结构化日志
- ✅ 不同级别的日志(INFO, WARNING, ERROR)
- ✅ 实时显示代码执行进度

### 3. 独立日志文件 / Separate Log Files ✓
- ✅ 所有输出保存到独立的日志文件
- ✅ 日志文件命名带时间戳: `experiment_YYYYMMDD_HHMMSS.log`
- ✅ 同时输出到控制台和文件
- ✅ 可以用 `tail -f` 实时查看

### 4. 数据集描述和可视化 / Dataset Descriptions and Visualizations ✓
- ✅ 每个数据集生成详细的统计信息(JSON格式)
- ✅ 生成8个子图的综合可视化报告
- ✅ 包括类别分布、特征类型、相关性矩阵等
- ✅ 保存在 `results/visualizations/datasets/` 目录

### 5. 模型结果可视化 / Model Results Visualization ✓
- ✅ 每个指标的独立比较图(F1-Score, ROC-AUC等)
- ✅ 热力图展示所有指标
- ✅ 训练时间和推理时间分析
- ✅ 类别不均衡策略对比图
- ✅ 保存在 `results/visualizations/comparisons/` 目录

### 6. 智能采样策略 / Smart Sampling Strategy ✓
- ✅ 针对大数据集(>20K样本)的KNN和SVM模型采样
- ✅ 使用分层采样保持类别分布
- ✅ 保证欺诈样本数量
- ✅ 快速模型(XGBoost, LightGBM等)使用全量数据

### 7. 多种类别不均衡处理 / Multiple Imbalance Handling Strategies ✓
- ✅ 5种策略: None, SMOTE, ADASYN, SMOTE+Tomek, Undersampling
- ✅ 每个监督学习模型测试多种策略
- ✅ 生成对比分析和可视化
- ✅ 自动记录每种策略的效果

### 8. 自适应参数选择 / Adaptive Parameter Selection ✓
- ✅ 根据数据集大小(small/medium/large)调整参数
- ✅ 根据特征数量调整神经网络架构
- ✅ 自动优化训练轮数和批次大小
- ✅ 为每个模型选择合适的参数组合

## 📁 文件结构 / File Structure

```
src/
├── experiment_refactored.py      # 核心框架(日志、数据加载、工具函数)
│                                  # Core framework (logging, data loading, utilities)
│
├── experiment_models.py           # 模型训练和评估
│                                  # Model training and evaluation
│
├── experiment_main.py             # 主执行脚本和可视化
│                                  # Main execution script and visualization
│
├── test_refactored.py             # 测试脚本
│                                  # Test script
│
├── quick_start.sh                 # 快速启动脚本
│                                  # Quick start script
│
├── README_REFACTORED.md           # 详细使用文档
│                                  # Detailed usage documentation
│
└── REFACTORING_SUMMARY.md         # 本文件
                                   # This file

logs/
└── experiment_YYYYMMDD_HHMMSS.log # 详细执行日志
                                    # Detailed execution log

results/
├── experiment_results.csv          # 所有实验结果
│                                   # All experimental results
│
└── visualizations/
    ├── datasets/                   # 每个数据集的分析报告
    │   └── [dataset_name]/         # Per-dataset analysis reports
    │       ├── dataset_analysis.png
    │       └── statistics.json
    │
    └── comparisons/                # 对比分析图表
        ├── model_comparison_f1_score.png   # Comparative analyses
        ├── model_comparison_roc_auc.png
        ├── imbalance_comparison.png
        ├── time_analysis.png
        ├── all_metrics_heatmap.png
        └── summary_report.txt
```

## 🚀 使用方法 / Usage

### 方法1: 使用快速启动脚本 / Method 1: Quick Start Script
```bash
cd /usr1/home/s124mdg53_07/wang/FYP/src
./quick_start.sh
```

### 方法2: 直接运行Python脚本 / Method 2: Run Python Script Directly
```bash
cd /usr1/home/s124mdg53_07/wang/FYP/src
python experiment_main.py
```

### 方法3: 自定义运行 / Method 3: Custom Run
```python
from experiment_main import main
from experiment_refactored import DATASET_CONFIGS

# 选择要运行的数据集 / Select datasets to run
# 快速测试 / Quick test
datasets = ['counterfeit_products', 'counterfeit_transactions']

# 完整实验 / Full experiment
# datasets = list(DATASET_CONFIGS.keys())

results = main()
```

## 🔍 测试验证 / Testing

运行测试脚本验证所有组件: / Run test script to verify all components:
```bash
cd /usr1/home/s124mdg53_07/wang/FYP/src
python test_refactored.py
```

测试项目包括 / Test items include:
- ✅ 模块导入 / Module imports
- ✅ 日志系统 / Logging system
- ✅ 内存管理 / Memory management
- ✅ 类别不均衡处理 / Imbalance handling
- ✅ 自适应参数 / Adaptive parameters
- ✅ 数据加载 / Data loading
- ✅ 性能评估 / Performance evaluation

## 📊 主要特性 / Key Features

### 日志系统示例 / Logging System Example
```python
logger.section("Major Section", level=1)      # 主要部分
logger.section("Sub-section", level=2)         # 子部分  
logger.section("Detail", level=3)              # 细节
logger.info("Information message")             # 信息
logger.warning("Warning message")              # 警告
logger.error("Error message")                  # 错误
logger.progress(5, 10, "Progress")             # 进度: 5/10 (50.0%)
logger.timer_start('operation')                # 开始计时
logger.timer_end('operation')                  # 结束计时并记录
logger.metric("F1-Score", 0.8523)             # 指标: F1-Score: 0.8523
```

### 内存管理 / Memory Management
```python
clear_memory()         # 清理内存和GPU缓存
get_memory_usage()     # 获取当前内存使用
log_memory_usage()     # 记录内存使用到日志
```

### 自适应参数示例 / Adaptive Parameters Example
```python
# 小数据集 / Small dataset
get_model_params('random_forest', 'small', 10)
# -> {'n_estimators': 200, 'max_depth': None}

# 大数据集 / Large dataset  
get_model_params('random_forest', 'large', 50)
# -> {'n_estimators': 100, 'max_depth': 10}
```

## 📈 输出示例 / Output Examples

### 日志文件示例 / Log File Example
```
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | DATASET 1/7: creditCardPCA
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | Dataset Processing: 1/7 (14.3%)
2025-11-11 10:30:16 | INFO     | Loading dataset: creditCardPCA
2025-11-11 10:30:17 | INFO     | Dataset loaded - Train: (228000, 31), Test: (57000, 31)
2025-11-11 10:30:17 | INFO     | Feature Analysis:
2025-11-11 10:30:17 | INFO     |   Numerical: 30 features
2025-11-11 10:30:17 | INFO     |   Categorical: 0 features
2025-11-11 10:30:18 | INFO     | [TIMER START] lr_train
2025-11-11 10:30:25 | INFO     | [TIMER END] lr_train: 7.23s
2025-11-11 10:30:25 | INFO     | Model: Logistic Regression - Strategy: No Resampling
2025-11-11 10:30:25 | INFO     |   Accuracy: 0.9854
2025-11-11 10:30:25 | INFO     |   Precision: 0.9123
2025-11-11 10:30:25 | INFO     |   Recall: 0.8845
2025-11-11 10:30:25 | INFO     |   F1-Score: 0.8982
2025-11-11 10:30:25 | INFO     |   ROC-AUC: 0.9756
```

### CSV输出格式 / CSV Output Format
```csv
model,dataset,imbalance_strategy,accuracy,precision,recall,f1_score,roc_auc,train_time,inference_time
Logistic Regression,creditCardPCA,none,0.9854,0.9123,0.8845,0.8982,0.9756,7.23,0.0234
Random Forest,creditCardPCA,none,0.9876,0.9234,0.8956,0.9093,0.9823,45.67,0.0456
XGBoost,creditCardPCA,smote,0.9889,0.9345,0.9012,0.9176,0.9867,32.45,0.0289
```

## 🔧 与原始代码的主要区别 / Major Differences from Original

| 方面 / Aspect | 原始版本 / Original | 重构版本 / Refactored |
|--------------|-------------------|----------------------|
| 输出语言 / Output Language | 中英混合 / Mixed | 全英文 / All English |
| 日志系统 / Logging | print语句 / print statements | 结构化日志系统 / Structured logging |
| 日志文件 / Log Files | 仅notebook输出 / Notebook only | 独立时间戳日志 / Separate timestamped logs |
| 代码组织 / Code Organization | 单文件 / Single file | 3个模块文件 / 3 modular files |
| 数据集分析 / Dataset Analysis | 有限 / Limited | 全面EDA报告 / Comprehensive EDA |
| 可视化 / Visualizations | 组合图表 / Combined plots | 独立+对比图表 / Individual + comparative |
| 采样策略 / Sampling | 固定 / Fixed | 自适应 / Adaptive |
| 不均衡处理 / Imbalance Handling | 2种方法 / 2 methods | 5种方法+对比 / 5 methods + comparison |
| 参数选择 / Parameters | 固定 / Fixed | 自适应 / Adaptive |
| 内存管理 / Memory | 基础 / Basic | 主动监控 / Active monitoring |
| 错误处理 / Error Handling | 基础 / Basic | 完善的异常处理 / Comprehensive |

## 🎯 模型列表 / Model List

### 监督学习 / Supervised Learning (7个模型)
1. Logistic Regression - 逻辑回归
2. Random Forest - 随机森林
3. XGBoost - 极端梯度提升
4. LightGBM - 轻量级梯度提升
5. K-Nearest Neighbors (KNN) - K近邻
6. PCA + SVM - 主成分分析+支持向量机
7. Multi-Layer Perceptron (MLP) - 多层感知机

### 无监督学习 / Unsupervised Learning (2个模型)
8. Isolation Forest - 孤立森林
9. Autoencoder - 自编码器

### 评估指标 / Evaluation Metrics (7个指标)
1. Accuracy - 准确率
2. Precision - 精确率
3. Recall - 召回率
4. F1-Score - F1分数
5. ROC-AUC - ROC曲线下面积
6. Training Time - 训练时间
7. Inference Time - 推理时间

## 💡 使用建议 / Usage Tips

1. **从小数据集开始** / Start with small datasets
   - 先运行counterfeit数据集验证流程
   - First run counterfeit datasets to verify workflow

2. **监控日志文件** / Monitor log files
   ```bash
   tail -f ../logs/experiment_*.log
   ```

3. **检查内存使用** / Check memory usage
   - 大数据集可能需要调整批次大小
   - Large datasets may need batch size adjustment

4. **使用tmux/screen** / Use tmux/screen
   - 长时间实验建议使用后台运行
   - Recommended for long-running experiments

5. **结果增量保存** / Results saved incrementally
   - 可以随时停止和恢复
   - Can stop and resume anytime

## 🐛 故障排除 / Troubleshooting

### 问题: 内存不足 / Issue: Out of Memory
**解决方案 / Solution:**
- 减少批次大小 / Reduce batch sizes
- 启用更多模型采样 / Enable more model sampling
- 逐个处理数据集 / Process datasets one by one

### 问题: 模型训练太慢 / Issue: Slow model training
**解决方案 / Solution:**
- 减少max_samples / Reduce max_samples
- 降低树模型的n_estimators / Decrease n_estimators for tree models
- 减少深度学习epochs / Reduce deep learning epochs

### 问题: 导入错误 / Issue: Import errors
**解决方案 / Solution:**
```bash
pip install -r requirements.txt
```

## 📚 文档 / Documentation

详细文档请参考 / For detailed documentation, see:
- `README_REFACTORED.md` - 完整使用指南 / Complete usage guide
- `test_refactored.py` - 测试和验证 / Testing and validation
- 日志文件 / Log files - 详细执行记录 / Detailed execution records

## ✨ 未来改进 / Future Enhancements

潜在改进方向 / Potential improvements:
- [ ] 使用Optuna进行超参数调优 / Hyperparameter tuning with Optuna
- [ ] 所有模型的交叉验证 / Cross-validation for all models
- [ ] SHAP值分析 / SHAP value analysis
- [ ] 模型集成方法 / Model ensemble methods
- [ ] 置信区间计算 / Confidence intervals
- [ ] 并行处理多数据集 / Parallel processing for multiple datasets

## 📞 联系 / Contact

如有问题,请查看:
- 生成的日志文件 / Generated log files
- README_REFACTORED.md文档 / README_REFACTORED.md documentation

For questions, check:
- Generated log files for detailed error messages
- README_REFACTORED.md for comprehensive documentation

---

**总结 / Summary**: 重构后的代码保持了原有功能,同时大幅提升了代码质量、可维护性和用户体验。所有输出均为英文,满足论文和研究需求。

The refactored code maintains all original functionality while significantly improving code quality, maintainability, and user experience. All outputs are in English as required for academic papers and research.
