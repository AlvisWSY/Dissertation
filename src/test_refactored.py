#!/usr/bin/env python
# coding: utf-8

"""
Test Script for Refactored Experiment
======================================

This script runs a minimal test to verify all components work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from experiment_refactored import (
            logger, clear_memory, get_memory_usage, log_memory_usage,
            DatasetLoader, ImbalanceHandler, DATASET_CONFIGS
        )
        print("✓ experiment_refactored imports successful")
    except Exception as e:
        print(f"✗ experiment_refactored import failed: {e}")
        return False
    
    try:
        from experiment_models import (
            ExperimentRunner, PerformanceEvaluator, get_model_params
        )
        print("✓ experiment_models imports successful")
    except Exception as e:
        print(f"✗ experiment_models import failed: {e}")
        return False
    
    try:
        from experiment_main import ResultsVisualizer, main
        print("✓ experiment_main imports successful")
    except Exception as e:
        print(f"✗ experiment_main import failed: {e}")
        return False
    
    return True


def test_logger():
    """Test logging functionality"""
    print("\nTesting logger...")
    try:
        from experiment_refactored import logger
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.section("Test Section", level=1)
        logger.progress(5, 10, "Test Progress")
        logger.timer_start("test_timer")
        import time
        time.sleep(0.1)
        logger.timer_end("test_timer")
        logger.metric("Test Metric", 0.95)
        
        print("✓ Logger functionality works")
        return True
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False


def test_memory_utils():
    """Test memory utilities"""
    print("\nTesting memory utilities...")
    try:
        from experiment_refactored import clear_memory, get_memory_usage, log_memory_usage
        
        mem_stats = get_memory_usage()
        print(f"  RAM usage: {mem_stats['ram_gb']:.2f} GB")
        
        clear_memory()
        log_memory_usage()
        
        print("✓ Memory utilities work")
        return True
    except Exception as e:
        print(f"✗ Memory utilities test failed: {e}")
        return False


def test_imbalance_handler():
    """Test imbalance handling"""
    print("\nTesting imbalance handler...")
    try:
        import pandas as pd
        import numpy as np
        from experiment_refactored import ImbalanceHandler
        
        # Create dummy imbalanced data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series([0] * 900 + [1] * 100)
        
        # Test imbalance ratio
        ratio = ImbalanceHandler.get_imbalance_ratio(y)
        print(f"  Imbalance ratio: {ratio:.2f}:1")
        
        # Test SMOTE
        X_smote, y_smote = ImbalanceHandler.apply_smote(X, y)
        print(f"  SMOTE: {len(y)} -> {len(y_smote)} samples")
        
        # Test class weights
        weights = ImbalanceHandler.get_class_weights(y)
        print(f"  Class weights: {weights}")
        
        print("✓ Imbalance handler works")
        return True
    except Exception as e:
        print(f"✗ Imbalance handler test failed: {e}")
        return False


def test_model_params():
    """Test adaptive model parameters"""
    print("\nTesting adaptive model parameters...")
    try:
        from experiment_models import get_model_params
        
        # Test small dataset
        params_small = get_model_params('random_forest', 'small', 10)
        print(f"  Small dataset RF params: {params_small}")
        
        # Test large dataset
        params_large = get_model_params('random_forest', 'large', 50)
        print(f"  Large dataset RF params: {params_large}")
        
        # Test MLP
        params_mlp = get_model_params('mlp', 'medium', 30)
        print(f"  Medium dataset MLP params: {params_mlp}")
        
        print("✓ Adaptive model parameters work")
        return True
    except Exception as e:
        print(f"✗ Model parameters test failed: {e}")
        return False


def test_dataset_loader():
    """Test dataset loader (if data available)"""
    print("\nTesting dataset loader...")
    try:
        from experiment_refactored import DatasetLoader, DATA_DIR
        
        # Check if any dataset exists
        available_datasets = []
        for dataset_name in ['counterfeit_products', 'counterfeit_transactions', 
                            'creditCardPCA', 'col14_behave']:
            dataset_path = DATA_DIR / dataset_name / 'train'
            if dataset_path.exists():
                available_datasets.append(dataset_name)
        
        if not available_datasets:
            print("  ⚠ No datasets found, skipping dataset loader test")
            return True
        
        # Test with first available dataset
        test_dataset = available_datasets[0]
        print(f"  Testing with dataset: {test_dataset}")
        
        loader = DatasetLoader(test_dataset)
        train_df, test_df = loader.load_data()
        print(f"  Loaded: train={train_df.shape}, test={test_df.shape}")
        
        # Test preprocessing
        X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
            train_df, test_df, apply_sampling=False
        )
        print(f"  Preprocessed: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"  Feature types: {len(feature_types['numerical'])} numerical, "
              f"{len(feature_types['categorical'])} categorical")
        
        print("✓ Dataset loader works")
        return True
    except Exception as e:
        print(f"✗ Dataset loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_evaluator():
    """Test performance evaluator"""
    print("\nTesting performance evaluator...")
    try:
        import numpy as np
        from experiment_models import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator()
        
        # Create dummy predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.2, 0.6, 0.85])
        
        result = evaluator.evaluate_supervised(
            y_true, y_pred, y_pred_proba,
            'Test Model', 'Test Dataset', 1.0, 0.01, 'none'
        )
        
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f}")
        print(f"  ROC-AUC: {result['roc_auc']:.4f}")
        
        # Test results DataFrame
        results_df = evaluator.get_results_df()
        print(f"  Results DataFrame shape: {results_df.shape}")
        
        print("✓ Performance evaluator works")
        return True
    except Exception as e:
        print(f"✗ Performance evaluator test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Refactored Experiment - Component Test Suite")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Logger Test", test_logger),
        ("Memory Utilities Test", test_memory_utils),
        ("Imbalance Handler Test", test_imbalance_handler),
        ("Model Parameters Test", test_model_params),
        ("Dataset Loader Test", test_dataset_loader),
        ("Performance Evaluator Test", test_performance_evaluator),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The refactored code is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
