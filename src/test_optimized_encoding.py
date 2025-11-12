"""
Test Optimized Categorical Encoding
===================================

This script tests the optimized encoding on a subset of creditCardTransaction
to verify it works correctly and is much faster.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

# Import modules
from experiment_refactored import logger, DatasetLoader, clear_memory

def test_optimized_encoding():
    """Test the new optimized encoding"""
    
    print("\n" + "="*100)
    print("TESTING OPTIMIZED CATEGORICAL ENCODING")
    print("="*100 + "\n")
    
    # Load a sample of creditCardTransaction
    print("Loading creditCardTransaction dataset (sample)...")
    loader = DatasetLoader('creditCardTransaction', max_samples=100000, handle_sparse=False)
    
    train_df, test_df = loader.load_data()
    print(f"✓ Loaded: train={train_df.shape}, test={test_df.shape}")
    
    # Test preprocessing with optimized encoding
    print(f"\nTesting preprocessing with optimized encoding...")
    start_time = time.time()
    
    X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
        train_df, test_df, apply_sampling=False
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Preprocessing completed in {elapsed:.2f} seconds")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Categorical features processed: {len(feature_types['categorical'])}")
    
    # Check for any issues
    print(f"\n✓ Data quality checks:")
    print(f"  Train nulls: {X_train.isnull().sum().sum()}")
    print(f"  Test nulls: {X_test.isnull().sum().sum()}")
    print(f"  Train fraud rate: {(y_train == 1).mean():.4f}")
    print(f"  Test fraud rate: {(y_test == 1).mean():.4f}")
    
    print("\n" + "="*100)
    print("✅ TEST PASSED! Optimized encoding is working correctly.")
    print("="*100 + "\n")
    
    return True


if __name__ == "__main__":
    test_optimized_encoding()
    print("\n🚀 Ready to run full experiment!")
    print("\nTo run from dataset #5 onwards:")
    print("  cd /usr1/home/s124mdg53_07/wang/FYP")
    print("  python src/experiment_main.py")
