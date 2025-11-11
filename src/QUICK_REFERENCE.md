# Quick Reference Guide

## 🚀 Getting Started (3 Steps)

### Step 1: Navigate to directory
```bash
cd /usr1/home/s124mdg53_07/wang/FYP/src
```

### Step 2: Run test (verify everything works)
```bash
python test_refactored.py
```

### Step 3: Run experiment
```bash
# Easy way
./quick_start.sh

# Or direct way
python experiment_main.py
```

## 📝 Common Commands

### Monitor progress in real-time
```bash
tail -f ../logs/experiment_*.log
```

### Check results
```bash
# View CSV
cat ../results/experiment_results.csv

# Or open in pandas
python -c "import pandas as pd; print(pd.read_csv('../results/experiment_results.csv'))"
```

### View visualizations
```bash
ls -lh ../results/visualizations/comparisons/
ls -lh ../results/visualizations/datasets/
```

## 🔧 Configuration

### Run specific datasets only
Edit `experiment_main.py`, find the `main()` function, and modify:
```python
# Line ~180
datasets_to_run = ['creditCardPCA', 'col14_behave']  # Your choice
# Instead of:
# datasets_to_run = list(DATASET_CONFIGS.keys())  # All datasets
```

### Disable imbalance comparison (faster)
Edit `experiment_main.py`, find the runner initialization:
```python
# Line ~172
runner = ExperimentRunner(
    compare_imbalance=False,  # Change to False
    use_sampling_for_slow_models=True
)
```

### Adjust sampling threshold
Edit `experiment_models.py`, find KNN or PCA+SVM functions:
```python
# Default: max_samples=20000
def run_knn(self, ..., max_samples=10000):  # Lower for faster
```

## 📊 Understanding Outputs

### Log File Format
```
TIMESTAMP | LEVEL | MESSAGE
2025-11-11 10:30:15 | INFO | Dataset loaded - Train: (228000, 31)
```

### CSV Columns
```
model              - Model name (e.g., "XGBoost")
dataset            - Dataset name (e.g., "creditCardPCA")
imbalance_strategy - Strategy used (e.g., "smote")
accuracy           - Accuracy score (0-1)
precision          - Precision score (0-1)
recall             - Recall score (0-1)
f1_score           - F1-Score (0-1)
roc_auc            - ROC-AUC score (0-1)
train_time         - Training time (seconds)
inference_time     - Inference time (seconds)
```

### Visualization Files
```
comparisons/
├── model_comparison_f1_score.png   - Models ranked by F1-Score
├── model_comparison_roc_auc.png    - Models ranked by ROC-AUC
├── imbalance_comparison.png        - Strategy comparison
├── time_analysis.png                - Training/inference time
├── all_metrics_heatmap.png         - 5 metrics heatmap
└── summary_report.txt              - Text summary

datasets/[dataset_name]/
├── dataset_analysis.png             - 8-subplot EDA report
└── statistics.json                  - Dataset statistics
```

## 🐛 Troubleshooting

### Problem: Import error
```bash
# Solution: Install requirements
pip install -r requirements_refactored.txt
```

### Problem: GPU not detected
```
# Expected output in log:
# "Detected 2 GPUs: ..." or "No GPU detected, using CPU"
# This is normal - code runs on CPU if GPU unavailable
```

### Problem: Out of memory
```python
# Solution 1: Edit experiment_models.py, reduce batch size
# Find these lines and change values:
batch_size=256  # Change from 512

# Solution 2: Run fewer datasets at once
datasets_to_run = ['counterfeit_products']  # One at a time
```

### Problem: Takes too long
```python
# Solution 1: Disable imbalance comparison
runner = ExperimentRunner(compare_imbalance=False, ...)

# Solution 2: Reduce epochs for deep learning
# Edit get_model_params() in experiment_models.py
'epochs': 20  # Change from 50

# Solution 3: Sample all models
self.use_sampling_for_slow_models = True  # Already default
```

## 📈 Expected Runtime

| Dataset Size | Models | Imbalance Strategies | Time |
|-------------|--------|---------------------|------|
| Small (5K) | 9 | 2 | ~5 min |
| Medium (200K) | 9 | 2 | ~30 min |
| Large (1M) | 9 | 2 | ~2 hours |
| All 7 datasets | 9 | 2 | ~6-8 hours |

*Times are approximate and vary based on hardware*

## 🎯 Common Tasks

### Task: Quick test before full run
```python
# Edit experiment_main.py
datasets_to_run = ['counterfeit_products']
runner = ExperimentRunner(compare_imbalance=False, ...)
```

### Task: Run only best models
```python
# Edit experiment_models.py, in run_all_models()
# Comment out slow models:
# results[f'knn_{strategy}'] = ...
# results[f'pca_svm_{strategy}'] = ...
```

### Task: Add custom model
```python
# 1. Add to experiment_models.py:
def run_my_model(self, X_train, y_train, X_test, y_test, ...):
    logger.section("My Custom Model", level=3)
    # ... your model code ...
    return result

# 2. Call in run_all_models():
results['my_model'] = self.run_my_model(...)
```

### Task: Export results to Excel
```python
import pandas as pd
df = pd.read_csv('../results/experiment_results.csv')
df.to_excel('../results/experiment_results.xlsx', index=False)
```

## 💡 Pro Tips

1. **Start small**: Test with 1-2 small datasets first
2. **Use tmux/screen**: For long experiments that survive disconnect
3. **Monitor log**: Keep `tail -f` running in another terminal
4. **Save intermediate**: Results saved after each dataset
5. **GPU utilization**: Check with `nvidia-smi` in another terminal

## 🔗 File Reference

| File | Purpose |
|------|---------|
| `experiment_refactored.py` | Core framework |
| `experiment_models.py` | Model implementations |
| `experiment_main.py` | Main execution |
| `test_refactored.py` | Test suite |
| `quick_start.sh` | Quick starter |
| `README_REFACTORED.md` | Full documentation |
| `REFACTORING_SUMMARY.md` | Refactoring summary |
| `COMPARISON.md` | Before/after comparison |
| `requirements_refactored.txt` | Dependencies |

## 📞 Need Help?

1. Check log file: `../logs/experiment_*.log`
2. Run test suite: `python test_refactored.py`
3. Read full docs: `README_REFACTORED.md`
4. Check comparison: `COMPARISON.md`

## ✅ Checklist Before Running

- [ ] Python 3.7+ installed
- [ ] All requirements installed (`pip install -r requirements_refactored.txt`)
- [ ] Test suite passes (`python test_refactored.py`)
- [ ] At least one dataset available in `../data/`
- [ ] Have 10GB+ free disk space (for large datasets + logs)
- [ ] Have 16GB+ RAM (or reduce batch sizes)
- [ ] (Optional) GPU available for faster training

## 🎓 For Your Dissertation

### What to include:
1. **Methods section**: Describe adaptive parameters and sampling strategies
2. **Results section**: Use generated CSV and visualizations
3. **Appendix**: Include example log excerpts showing experiment details

### Key figures to use:
- `model_comparison_f1_score.png` - Main results
- `imbalance_comparison.png` - Strategy analysis
- `all_metrics_heatmap.png` - Comprehensive view
- `dataset/[name]/dataset_analysis.png` - Dataset characteristics

### Tables to generate:
```python
import pandas as pd
df = pd.read_csv('../results/experiment_results.csv')

# Top 10 models by F1-Score
print(df.nlargest(10, 'f1_score')[['model', 'dataset', 'f1_score', 'roc_auc']])

# Average performance by model
print(df.groupby('model')[['accuracy', 'f1_score', 'roc_auc']].mean())

# Best model per dataset
print(df.loc[df.groupby('dataset')['f1_score'].idxmax()])
```

---

**Remember**: All outputs are in English and publication-ready! 📄✨
