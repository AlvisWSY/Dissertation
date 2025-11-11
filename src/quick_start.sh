#!/bin/bash

# Quick Start Script for Refactored Experiment
# ============================================

echo "=========================================="
echo "Fraud Detection Experiment - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if required packages are installed
echo "Checking required packages..."
python3 << EOF
import sys
required_packages = [
    'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
    'torch', 'xgboost', 'lightgbm', 'imbalanced-learn', 'psutil'
]

missing = []
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        missing.append(package)

if missing:
    print(f"✗ Missing packages: {', '.join(missing)}")
    print("\nInstall missing packages with:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("✓ All required packages are installed")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Create necessary directories
echo "Creating output directories..."
mkdir -p ../logs
mkdir -p ../results/visualizations/datasets
mkdir -p ../results/visualizations/models
mkdir -p ../results/visualizations/comparisons
echo "✓ Directories created"
echo ""

# Ask user for experiment mode
echo "Select experiment mode:"
echo "  1) Quick test (2 small datasets)"
echo "  2) Medium test (3-4 datasets)"
echo "  3) Full experiment (all 7 datasets)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Running quick test mode..."
        DATASETS="['counterfeit_products', 'counterfeit_transactions']"
        ;;
    2)
        echo "Running medium test mode..."
        DATASETS="['counterfeit_products', 'counterfeit_transactions', 'creditCardPCA', 'col14_behave']"
        ;;
    3)
        echo "Running full experiment mode..."
        DATASETS="list(DATASET_CONFIGS.keys())"
        ;;
    *)
        echo "Invalid choice. Using quick test mode."
        DATASETS="['counterfeit_products', 'counterfeit_transactions']"
        ;;
esac

echo ""
echo "Starting experiment..."
echo "Log file will be created in ../logs/"
echo "Results will be saved in ../results/"
echo ""
echo "You can monitor progress with:"
echo "  tail -f ../logs/experiment_*.log"
echo ""

# Create temporary run script
cat > /tmp/run_experiment_temp.py << EOF
import sys
sys.path.append('.')

from experiment_main import main
from experiment_refactored import DATASET_CONFIGS

# Override datasets_to_run in main
import experiment_main
original_main = experiment_main.main

def custom_main():
    datasets_to_run = $DATASETS
    
    # Import necessary components
    from experiment_refactored import logger, RESULTS_DIR, VIZ_DIR, log_memory_usage, clear_memory, DatasetLoader
    from experiment_models import ExperimentRunner
    from experiment_main import ResultsVisualizer
    
    logger.section("FRAUD DETECTION EXPERIMENT - MAIN EXECUTION", level=1)
    logger.info("Starting comprehensive fraud detection model comparison experiment")
    log_memory_usage()
    
    runner = ExperimentRunner(
        compare_imbalance=True,
        use_sampling_for_slow_models=True
    )
    
    logger.info(f"Datasets to process: {datasets_to_run}")
    logger.info(f"Total datasets: {len(datasets_to_run)}")
    
    all_results = {}
    for idx, dataset_name in enumerate(datasets_to_run, 1):
        try:
            logger.section(f"DATASET {idx}/{len(datasets_to_run)}: {dataset_name}", level=1)
            logger.progress(idx, len(datasets_to_run), "Dataset Processing")
            
            config = DATASET_CONFIGS[dataset_name]
            dataset_size = config['size_category']
            
            logger.section(f"Loading Dataset: {dataset_name}", level=2)
            loader = DatasetLoader(
                dataset_name,
                max_samples=config['max_samples'],
                handle_sparse=config['handle_sparse']
            )
            
            train_df, test_df = loader.load_data()
            X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
                train_df, test_df, apply_sampling=False
            )
            
            loader.generate_dataset_report(
                train_df, test_df, X_train, y_train, X_test, y_test, feature_types
            )
            
            log_memory_usage()
            
            dataset_results = runner.run_all_models(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size
            )
            
            all_results[dataset_name] = dataset_results
            
            del train_df, test_df, X_train, X_test, y_train, y_test, loader
            clear_memory()
            
            logger.info(f"Dataset {dataset_name} completed and memory cleared")
            log_memory_usage()
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.section("COMPILING RESULTS", level=1)
    results_df = runner.evaluator.get_results_df()
    
    results_csv_path = RESULTS_DIR / 'experiment_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to: {results_csv_path}")
    
    visualizer = ResultsVisualizer(results_df)
    visualizer.generate_all_visualizations()
    
    logger.section("EXPERIMENT COMPLETE", level=1)
    logger.info(f"Total datasets processed: {len(all_results)}")
    logger.info(f"Total experiments run: {len(results_df)}")
    logger.info(f"Results saved to: {results_csv_path}")
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info("All outputs are in English as requested")
    log_memory_usage()
    
    return results_df

if __name__ == "__main__":
    results = custom_main()
    from experiment_refactored import logger
    logger.info("Experiment finished successfully!")
EOF

# Run the experiment
python3 /tmp/run_experiment_temp.py

# Clean up
rm /tmp/run_experiment_temp.py

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Check the following locations:"
echo "  Logs: ../logs/experiment_*.log"
echo "  Results CSV: ../results/experiment_results.csv"
echo "  Visualizations: ../results/visualizations/"
echo ""
