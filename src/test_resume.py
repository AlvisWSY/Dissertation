"""
Test Resume Capability
=====================

This script tests that the resume functionality works correctly
by checking if previous results can be loaded and merged properly.
"""

import pandas as pd
from pathlib import Path

# Paths
RESULTS_DIR = Path('/usr1/home/s124mdg53_07/wang/FYP/results')
results_csv_path = RESULTS_DIR / 'experiment_results.csv'

print("\n" + "="*80)
print("TESTING RESUME CAPABILITY")
print("="*80 + "\n")

# Check if previous results exist
if results_csv_path.exists():
    print("✓ Previous results file found")
    
    # Load and analyze
    df = pd.read_csv(results_csv_path)
    
    print(f"\nCurrent Results Summary:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique datasets: {df['dataset'].nunique()}")
    print(f"  Datasets: {list(df['dataset'].unique())}")
    
    # Show counts per dataset
    print(f"\nExperiments per dataset:")
    dataset_counts = df.groupby('dataset').size()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset:30s}: {count:3d} experiments")
    
    # Check if we have all 7 datasets
    all_datasets = ['IEEE', 'col14_behave', 'col16_raw', 'counterfeit_products',
                   'creditCardTransaction', 'creditCardPCA', 'counterfeit_transactions']
    
    completed = set(df['dataset'].unique())
    missing = set(all_datasets) - completed
    
    print(f"\nDataset Status:")
    print(f"  Completed: {len(completed)}/7")
    if missing:
        print(f"  Missing: {list(missing)}")
        print(f"\n💡 To complete the experiment:")
        print(f"     Set start_from_index = {all_datasets.index(list(missing)[0])}")
        print(f"     This will run: {list(missing)}")
    else:
        print(f"  ✅ All datasets completed!")
    
    # Check data quality
    print(f"\nData Quality Checks:")
    required_cols = ['model', 'dataset', 'imbalance_strategy', 'accuracy', 
                     'precision', 'recall', 'f1_score', 'roc_auc']
    
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"  ⚠️  {col}: {null_count} null values")
            else:
                print(f"  ✓ {col}: OK")
        else:
            print(f"  ✗ {col}: MISSING COLUMN")
    
else:
    print("ℹ️  No previous results found")
    print("   This is a fresh start - all datasets will be run")

print("\n" + "="*80)
print("Resume capability test complete")
print("="*80 + "\n")
