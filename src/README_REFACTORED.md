# Fraud Detection Experiment - Refactored Version

## Overview

This is a comprehensive refactoring of the fraud detection model comparison experiment. The code has been restructured for better maintainability, logging, and English-only outputs.

## Key Features

### ✅ Requirements Met

1. **All English Outputs** - All logs, tables, CSV files, and visualizations are in English
2. **Real-time Logging** - Detailed logging system tracks code execution progress
3. **Separate Log Files** - All print statements are saved to a dedicated log file
4. **Dataset Descriptions** - Comprehensive EDA reports with visualizations for each dataset
5. **Model Visualizations** - Individual and comparative visualizations for all model results
6. **Smart Sampling** - Intelligent sampling for large datasets on specific models
7. **Multiple Imbalance Strategies** - Comparison of various class imbalance handling methods
8. **Adaptive Parameters** - Model parameters automatically adjusted based on dataset characteristics

## File Structure

```
src/
├── experiment_refactored.py    # Main framework (logging, data loading, utilities)
├── experiment_models.py         # Model training and evaluation
├── experiment_main.py           # Main execution script and visualization
└── README_REFACTORED.md         # This file

Generated Outputs:
logs/
└── experiment_YYYYMMDD_HHMMSS.log  # Detailed execution log

results/
├── experiment_results.csv           # All experimental results
└── visualizations/
    ├── datasets/                    # Per-dataset EDA reports
    │   ├── IEEE/
    │   │   ├── dataset_analysis.png
    │   │   └── statistics.json
    │   └── ...
    ├── models/                      # Per-model visualizations
    └── comparisons/                 # Comparative analyses
        ├── model_comparison_f1_score.png
        ├── model_comparison_roc_auc.png
        ├── imbalance_comparison.png
        ├── time_analysis.png
        ├── all_metrics_heatmap.png
        └── summary_report.txt
```

## Usage

### Basic Usage

```python
# Run the complete experiment
python src/experiment_main.py
```

### Running Specific Datasets

Edit `experiment_main.py` and modify the `datasets_to_run` variable:

```python
# In main() function:
datasets_to_run = ['counterfeit_products', 'creditCardPCA']  # Quick test
# datasets_to_run = list(DATASET_CONFIGS.keys())  # All datasets
```

### Custom Configuration

You can adjust experiment parameters in `experiment_main.py`:

```python
runner = ExperimentRunner(
    compare_imbalance=True,           # Enable imbalance strategy comparison
    use_sampling_for_slow_models=True # Enable smart sampling for KNN/SVM
)
```

## Features in Detail

### 1. Logging System

The refactored code includes a comprehensive logging system:

```python
# All operations are logged with timestamps
logger.info("Processing dataset...")
logger.warning("Large dataset detected")
logger.error("Failed to load data")

# Section headers for better organization
logger.section("Data Preprocessing", level=1)  # Major section
logger.section("Encoding Features", level=2)    # Sub-section
logger.section("Applying SMOTE", level=3)       # Detail

# Progress tracking
logger.progress(5, 10, "Dataset Processing")  # Output: Dataset Processing: 5/10 (50.0%)

# Timer functionality
logger.timer_start('model_training')
# ... training code ...
duration = logger.timer_end('model_training')  # Logs: [TIMER END] model_training: 45.23s

# Metric logging
logger.metric("F1-Score", 0.8523)  # Output:   F1-Score: 0.8523
```

### 2. Dataset Analysis

For each dataset, the system generates:

- **Statistical Summary** (JSON format)
  - Sample counts (train/test)
  - Feature counts (numerical/categorical/ID/timestamp)
  - Class distribution
  - Imbalance ratio
  - Fraud rate

- **Comprehensive Visualization** (PNG format)
  - Class distribution plots (train/test)
  - Feature type distribution
  - Feature correlation heatmap
  - Feature variance analysis
  - Missing value analysis
  - Dataset summary table

### 3. Imbalance Handling Strategies

The experiment compares multiple imbalance handling techniques:

1. **No Resampling** - Baseline with original imbalanced data
2. **SMOTE** - Synthetic Minority Over-sampling Technique
3. **ADASYN** - Adaptive Synthetic Sampling
4. **SMOTE+Tomek** - SMOTE with Tomek Links cleaning
5. **Random Undersampling** - Reduce majority class

Each strategy is tested on all supervised models, and results are compared.

### 4. Adaptive Model Parameters

Model parameters are automatically adjusted based on:

- **Dataset Size**: small (< 10K), medium (10K-500K), large (> 500K)
- **Number of Features**: affects neural network architecture
- **Examples**:
  ```python
  # Small dataset
  Random Forest: n_estimators=200, max_depth=None
  
  # Large dataset  
  Random Forest: n_estimators=100, max_depth=10
  ```

### 5. Smart Sampling Strategy

For computationally expensive models (KNN, SVM) on large datasets:

- **Automatic sampling** when dataset > 20,000 samples
- **Stratified sampling** to maintain class distribution
- **Guaranteed fraud samples** for extremely imbalanced datasets
- **Applies only to specific models** (fast models use full data)

### 6. Models Included

**Supervised Learning:**
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Support Vector Machine (with PCA)
- Multi-Layer Perceptron (Deep Learning)

**Unsupervised/Anomaly Detection:**
- Isolation Forest
- Autoencoder (Deep Learning)

### 7. Evaluation Metrics

All models are evaluated on:
- **Accuracy** - Overall correctness
- **Precision** - Fraud detection precision
- **Recall** - Fraud detection coverage
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Training Time** - Model training duration
- **Inference Time** - Prediction speed

### 8. Visualization Outputs

**Per-Dataset Visualizations:**
- Dataset analysis report with 8 subplots

**Comparative Visualizations:**
- Model comparison by F1-Score
- Model comparison by ROC-AUC
- Imbalance strategy comparison
- Time analysis (training and inference)
- All metrics heatmap (5 metrics)
- Summary report (text file)

## Output Examples

### Log File Example

```
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | Experiment Logger Initialized: experiment_20251111_103015
2025-11-11 10:30:15 | INFO     | Log file: /path/to/logs/experiment_20251111_103015.log
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | 
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | Initialization Complete
2025-11-11 10:30:15 | INFO     | ================================================================================
2025-11-11 10:30:15 | INFO     | Base Directory: /usr1/home/s124mdg53_07/wang/FYP
2025-11-11 10:30:15 | INFO     | Results Directory: /usr1/home/s124mdg53_07/wang/FYP/results
```

### CSV Output Example

| model | dataset | imbalance_strategy | accuracy | precision | recall | f1_score | roc_auc | train_time | inference_time |
|-------|---------|-------------------|----------|-----------|--------|----------|---------|------------|----------------|
| XGBoost | creditCardPCA | smote | 0.9856 | 0.9245 | 0.8934 | 0.9087 | 0.9823 | 45.23 | 0.0234 |
| Random Forest | IEEE | none | 0.9654 | 0.8567 | 0.7892 | 0.8215 | 0.9456 | 89.12 | 0.0456 |

## Memory Management

The refactored code includes aggressive memory management:

```python
# Clear memory after each dataset
clear_memory()

# Monitor memory usage
log_memory_usage()

# Automatic GPU cache clearing
```

## Error Handling

Robust error handling ensures the experiment continues even if individual models fail:

```python
try:
    # Train model
    results = runner.run_all_models(X_train, y_train, X_test, y_test, ...)
except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    # Continue with next dataset
    continue
```

## GPU Support

Automatic GPU detection and utilization:
- Multi-GPU support for deep learning models (MLP, Autoencoder)
- XGBoost GPU acceleration (`tree_method='gpu_hist'`)
- Automatic fallback to CPU if GPU unavailable

## Tips for Running

1. **Start with small datasets** to verify everything works
2. **Monitor the log file** in real-time: `tail -f logs/experiment_*.log`
3. **Check memory usage** if running large datasets
4. **Use tmux/screen** for long-running experiments
5. **Results are saved incrementally** - safe to stop and resume

## Differences from Original

| Aspect | Original | Refactored |
|--------|----------|------------|
| Output Language | Mixed Chinese/English | All English |
| Logging | Print statements | Structured logging system |
| Log Files | Notebook output only | Dedicated timestamped log files |
| Dataset Analysis | Limited | Comprehensive EDA with visualizations |
| Model Visualization | Combined plots | Individual + comparative plots |
| Sampling Strategy | Fixed | Adaptive based on model and dataset |
| Imbalance Handling | 2 methods | 5 methods with comparison |
| Parameter Selection | Fixed | Adaptive based on dataset characteristics |
| Code Organization | Single file | Modular (3 files) |
| Memory Management | Basic | Aggressive with monitoring |

## Troubleshooting

### Issue: Out of Memory
**Solution**: 
- Reduce batch sizes in model parameters
- Enable sampling for more models
- Process datasets one at a time

### Issue: Models take too long
**Solution**:
- Reduce max_samples in smart_sample()
- Decrease n_estimators for tree-based models
- Reduce epochs for deep learning models

### Issue: Import errors
**Solution**:
```bash
pip install -r requirements.txt
```

## Future Enhancements

Potential improvements:
- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validation for all models
- [ ] SHAP value analysis for interpretability
- [ ] Model ensemble methods
- [ ] Confidence intervals for metrics
- [ ] Parallel processing for multiple datasets

## Contact

For questions or issues, refer to the main experiment documentation or check the generated log files for detailed error messages.

---

**Note**: This refactored version maintains all functionality of the original while adding significant improvements in logging, visualization, and code organization. All outputs are guaranteed to be in English as requested.
