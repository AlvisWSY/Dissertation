#!/usr/bin/env python
# coding: utf-8

"""
Fraud Detection Model Comparison Experiment (Refactored Version)
=================================================================

This experiment compares various machine learning and deep learning methods
on different fraud detection datasets.

Features:
- Real-time logging system with progress tracking
- Comprehensive dataset visualization and analysis
- Multiple imbalance handling strategies comparison
- Adaptive model parameters based on dataset characteristics
- Smart sampling for large datasets on specific models
- Individual visualization for each model's performance
- All outputs in English

Author: Experiment Framework
Date: 2025-11-11
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import json
import warnings
import time
import gc
import os
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

# Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold

# Imbalance handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

# Supervised learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

# Unsupervised learning models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = True

# Settings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Path configuration
BASE_DIR = Path('/usr1/home/s124mdg53_07/wang/FYP')
DATA_DIR = BASE_DIR / 'data'
JSON_DIR = BASE_DIR / 'json'
RESULTS_DIR = BASE_DIR / 'results'
LOG_DIR = BASE_DIR / 'logs'
VIZ_DIR = RESULTS_DIR / 'visualizations'

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)
(VIZ_DIR / 'datasets').mkdir(exist_ok=True)
(VIZ_DIR / 'models').mkdir(exist_ok=True)
(VIZ_DIR / 'comparisons').mkdir(exist_ok=True)


# ============================================================================
# 2. LOGGING SYSTEM
# ============================================================================

class ExperimentLogger:
    """Centralized logging system for the experiment"""
    
    def __init__(self, log_dir: Path = LOG_DIR, experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*80)
        self.logger.info(f"Experiment Logger Initialized: {experiment_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*80)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def section(self, title: str, level: int = 1):
        """Log section header"""
        if level == 1:
            self.logger.info("\n" + "="*80)
            self.logger.info(f"{title}")
            self.logger.info("="*80)
        elif level == 2:
            self.logger.info("\n" + "-"*80)
            self.logger.info(f"{title}")
            self.logger.info("-"*80)
        else:
            self.logger.info(f"\n{'>'*3} {title}")
    
    def progress(self, current: int, total: int, prefix: str = "Progress"):
        """Log progress"""
        percentage = (current / total) * 100
        self.logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
    
    def metric(self, name: str, value: float, unit: str = ""):
        """Log metric"""
        self.logger.info(f"  {name}: {value:.4f} {unit}")
    
    def timer_start(self, name: str):
        """Start timer"""
        if not hasattr(self, '_timers'):
            self._timers = {}
        self._timers[name] = time.time()
        self.logger.info(f"[TIMER START] {name}")
    
    def timer_end(self, name: str):
        """End timer and log duration"""
        if hasattr(self, '_timers') and name in self._timers:
            duration = time.time() - self._timers[name]
            self.logger.info(f"[TIMER END] {name}: {duration:.2f}s")
            del self._timers[name]
            return duration
        return None


# Initialize global logger
logger = ExperimentLogger()


# ============================================================================
# 3. MEMORY MANAGEMENT UTILITIES
# ============================================================================

def clear_memory():
    """Clear memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    
    memory_stats = {
        'ram_gb': mem_info.rss / 1024**3
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_stats[f'gpu_{i}_allocated_gb'] = allocated
            memory_stats[f'gpu_{i}_reserved_gb'] = reserved
    
    return memory_stats


def log_memory_usage():
    """Log current memory usage"""
    mem_stats = get_memory_usage()
    logger.info(f"Memory Usage: RAM={mem_stats['ram_gb']:.2f}GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: Allocated={mem_stats.get(f'gpu_{i}_allocated_gb', 0):.2f}GB, "
                       f"Reserved={mem_stats.get(f'gpu_{i}_reserved_gb', 0):.2f}GB")


def log_gpu_info():
    """Log GPU information"""
    if torch.cuda.is_available():
        logger.info(f"Detected {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("No GPU detected, using CPU")


# ============================================================================
# 4. DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIGS = {
    'IEEE': {
        'max_samples': None,
        'handle_sparse': True,
        'size_category': 'large',
        'expected_samples': 472000,
        'expected_features': 81,
        'description': 'IEEE-CIS Fraud Detection dataset with high-dimensional PCA features'
    },
    'col14_behave': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'medium',
        'expected_samples': 238000,
        'expected_features': 15,
        'description': 'Behavioral fraud detection dataset with categorical features'
    },
    'col16_raw': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'large',
        'expected_samples': 1470000,
        'expected_features': 14,
        'description': 'E-commerce transaction fraud detection dataset'
    },
    'creditCardPCA': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'medium',
        'expected_samples': 228000,
        'expected_features': 34,
        'description': 'PCA-processed credit card fraud detection dataset'
    },
    'creditCardTransaction': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'large',
        'expected_samples': 1300000,
        'expected_features': 13,
        'description': 'Credit card transaction fraud detection dataset'
    },
    'counterfeit_products': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'small',
        'expected_samples': 4000,
        'expected_features': 16,
        'description': 'Product authenticity detection dataset'
    },
    'counterfeit_transactions': {
        'max_samples': None,
        'handle_sparse': False,
        'size_category': 'small',
        'expected_samples': 2400,
        'expected_features': 19,
        'description': 'Transaction authenticity detection dataset'
    },
}


# ============================================================================
# 5. IMBALANCE HANDLING STRATEGIES
# ============================================================================

class ImbalanceHandler:
    """Handles class imbalance with various strategies"""
    
    @staticmethod
    def get_imbalance_ratio(y: pd.Series) -> float:
        """Calculate class imbalance ratio"""
        counts = y.value_counts()
        if len(counts) < 2:
            return 1.0
        return counts.max() / counts.min()
    
    @staticmethod
    def apply_smote(X: pd.DataFrame, y: pd.Series, 
                   sampling_strategy='auto', k_neighbors=5) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE oversampling"""
        try:
            # Adjust k_neighbors if minority class has few samples
            min_class_count = y.value_counts().min()
            k_neighbors = min(k_neighbors, min_class_count - 1)
            if k_neighbors < 1:
                logger.warning("Not enough minority samples for SMOTE, returning original data")
                return X, y
            
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE applied: {len(X)} -> {len(X_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_adasyn(X: pd.DataFrame, y: pd.Series, 
                    sampling_strategy='auto') -> Tuple[pd.DataFrame, pd.Series]:
        """Apply ADASYN adaptive oversampling"""
        try:
            adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            logger.info(f"ADASYN applied: {len(X)} -> {len(X_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"ADASYN failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_smote_tomek(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE + Tomek Links combined sampling"""
        try:
            smt = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smt.fit_resample(X, y)
            logger.info(f"SMOTE+Tomek applied: {len(X)} -> {len(X_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"SMOTE+Tomek failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_undersampling(X: pd.DataFrame, y: pd.Series, 
                          sampling_strategy='auto') -> Tuple[pd.DataFrame, pd.Series]:
        """Apply random undersampling"""
        try:
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            logger.info(f"Undersampling applied: {len(X)} -> {len(X_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"Undersampling failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_smote_enn(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE + ENN combined sampling"""
        try:
            smote_enn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
            logger.info(f"SMOTE+ENN applied: {len(X)} -> {len(X_resampled)} samples")
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            logger.warning(f"SMOTE+ENN failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def get_class_weights(y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for balanced training"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))


# Define imbalance strategies for comparison
IMBALANCE_STRATEGIES = {
    'none': {
        'name': 'No Resampling',
        'handler': None,
        'description': 'Original imbalanced data'
    },
    'smote': {
        'name': 'SMOTE',
        'handler': ImbalanceHandler.apply_smote,
        'description': 'Synthetic Minority Over-sampling Technique'
    },
    'adasyn': {
        'name': 'ADASYN',
        'handler': ImbalanceHandler.apply_adasyn,
        'description': 'Adaptive Synthetic Sampling'
    },
    'smote_tomek': {
        'name': 'SMOTE+Tomek',
        'handler': ImbalanceHandler.apply_smote_tomek,
        'description': 'SMOTE with Tomek Links cleaning'
    },
    'undersampling': {
        'name': 'Random Undersampling',
        'handler': ImbalanceHandler.apply_undersampling,
        'description': 'Random undersampling of majority class'
    },
}


# ============================================================================
# 6. SMART SAMPLING FOR LARGE DATASETS
# ============================================================================

def smart_sample(X: pd.DataFrame, y: pd.Series, 
                max_samples: int = 100000,
                strategy: str = 'stratified',
                min_fraud_samples: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Smart sampling with guaranteed fraud samples for sparse datasets
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Label data
    max_samples : int
        Maximum number of samples
    strategy : str
        Sampling strategy: 'stratified' (maintains class ratio), 'random'
    min_fraud_samples : int
        Minimum number of fraud samples to guarantee in sample
    
    Returns:
    --------
    X_sample, y_sample : Sampled data
    """
    if len(X) <= max_samples:
        return X, y
    
    logger.info(f"Large dataset ({len(X):,} samples), sampling to {max_samples:,}")
    
    # Check fraud rate
    fraud_count = (y == 1).sum()
    fraud_rate = fraud_count / len(y)
    logger.info(f"Original fraud rate: {fraud_rate*100:.4f}% ({fraud_count:,} fraud samples)")
    
    if strategy == 'stratified':
        # For extremely imbalanced datasets, ensure minimum fraud samples
        if fraud_count > 0 and fraud_count < min_fraud_samples:
            logger.warning(f"Very low fraud samples, ensuring at least {min_fraud_samples} fraud samples")
            # Sample all fraud cases
            fraud_idx = y[y == 1].index
            normal_idx = y[y == 0].index
            
            # Sample normal cases
            n_normal = min(max_samples - len(fraud_idx), len(normal_idx))
            normal_sample_idx = np.random.choice(normal_idx, n_normal, replace=False)
            
            # Combine indices
            sample_idx = np.concatenate([fraud_idx, normal_sample_idx])
            np.random.shuffle(sample_idx)
            
            X_sample = X.loc[sample_idx]
            y_sample = y.loc[sample_idx]
        else:
            # Standard stratified sampling
            try:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, train_size=max_samples, stratify=y, random_state=42
                )
            except ValueError:
                # Fallback to random sampling if stratification fails
                logger.warning("Stratified sampling failed, using random sampling")
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X.iloc[indices]
                y_sample = y.iloc[indices]
    else:
        # Random sampling
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
    
    # Verify fraud samples in result
    sampled_fraud = (y_sample == 1).sum()
    sampled_fraud_rate = sampled_fraud / len(y_sample)
    logger.info(f"Sampled fraud rate: {sampled_fraud_rate*100:.4f}% ({sampled_fraud:,} fraud samples)")
    
    return X_sample, y_sample


# Log initialization complete
logger.section("Initialization Complete", level=1)
logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Results Directory: {RESULTS_DIR}")
logger.info(f"Visualization Directory: {VIZ_DIR}")
logger.info(f"Available Datasets: {len(DATASET_CONFIGS)}")
logger.info(f"Imbalance Strategies: {len(IMBALANCE_STRATEGIES)}")
log_gpu_info()
log_memory_usage()


# ============================================================================
# 7. DATASET LOADER AND PREPROCESSOR
# ============================================================================

class DatasetLoader:
    """Dataset loader and preprocessor with comprehensive analysis"""
    
    def __init__(self, dataset_name: str, data_dir: Path = DATA_DIR,
                 handle_sparse: bool = True, max_samples: Optional[int] = None):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.handle_sparse = handle_sparse
        self.max_samples = max_samples
        self.is_sparse = False
        
        logger.section(f"Initializing DatasetLoader: {dataset_name}", level=2)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test sets"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        train_path = self.data_dir / self.dataset_name / 'train'
        test_path = self.data_dir / self.dataset_name / 'test'
        
        # Find CSV files
        train_files = list(train_path.glob('*.csv'))
        test_files = list(test_path.glob('*.csv'))
        
        if not train_files or not test_files:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_name}")
        
        logger.info(f"Loading train file: {train_files[0].name}")
        logger.info(f"Loading test file: {test_files[0].name}")
        
        train_df = pd.read_csv(train_files[0])
        test_df = pd.read_csv(test_files[0])
        
        logger.info(f"Dataset loaded - Train: {train_df.shape}, Test: {test_df.shape}")
        
        return train_df, test_df
    
    def identify_label_column(self, df: pd.DataFrame) -> str:
        """Identify label column"""
        label_candidates = ['is_fraud', 'isFraud', 'fraud', 'label', 'target', 'Class']
        for col in label_candidates:
            if col in df.columns:
                logger.info(f"Identified label column: {col}")
                return col
        raise ValueError(f"Cannot identify label column for dataset {self.dataset_name}")
    
    def identify_feature_types(self, df: pd.DataFrame, label_col: str) -> Dict[str, List[str]]:
        """Identify feature types: numerical, categorical, ID, timestamp"""
        features = [col for col in df.columns if col != label_col]
        
        numerical_features = []
        categorical_features = []
        id_features = []
        timestamp_features = []
        
        for col in features:
            # Timestamp features
            if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                timestamp_features.append(col)
            # ID features
            elif '_id' in col.lower() or col.lower().endswith('id'):
                id_features.append(col)
            # Categorical features
            elif df[col].dtype == 'object' or df[col].nunique() < 20:
                categorical_features.append(col)
            # Numerical features
            else:
                numerical_features.append(col)
        
        feature_types = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'id': id_features,
            'timestamp': timestamp_features
        }
        
        logger.info("Feature Analysis:")
        logger.info(f"  Numerical: {len(numerical_features)} features")
        logger.info(f"  Categorical: {len(categorical_features)} features")
        logger.info(f"  ID: {len(id_features)} features (will be removed)")
        logger.info(f"  Timestamp: {len(timestamp_features)} features (will be removed)")
        
        return feature_types
    
    def handle_sparse_features(self, X_train: pd.DataFrame, 
                              X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle sparse features for highly sparse datasets"""
        if not self.handle_sparse:
            return X_train, X_test
        
        # Check sparsity on training set only
        sparsity = (X_train == 0).sum().sum() / (X_train.shape[0] * X_train.shape[1])
        logger.info(f"Data sparsity: {sparsity*100:.2f}%")
        
        if sparsity > 0.5:  # If sparsity > 50%
            self.is_sparse = True
            logger.warning("High sparsity detected, applying sparse feature processing")
            
            # Remove all-zero columns (fit on train, apply to both)
            zero_cols = X_train.columns[(X_train == 0).all()]
            if len(zero_cols) > 0:
                logger.info(f"Removing {len(zero_cols)} all-zero columns")
                X_train = X_train.drop(columns=zero_cols)
                X_test = X_test.drop(columns=zero_cols)
            
            # Remove low-variance columns (fit on train only)
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train)
            
            selected_cols = X_train.columns[selector.get_support()]
            removed_cols = len(X_train.columns) - len(selected_cols)
            if removed_cols > 0:
                logger.info(f"Removing {removed_cols} low-variance columns")
                X_train = X_train[selected_cols]
                X_test = X_test[selected_cols]
        
        return X_train, X_test
    
    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                  encode_categorical: bool = True, 
                  apply_sampling: bool = True) -> Tuple:
        """Preprocess data with no data leakage"""
        logger.section("Data Preprocessing", level=3)
        
        # Identify label column
        label_col = self.identify_label_column(train_df)
        
        # Identify feature types
        feature_types = self.identify_feature_types(train_df, label_col)
        
        # Separate features and labels
        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col]
        
        # Class imbalance analysis
        imbalance_ratio = ImbalanceHandler.get_imbalance_ratio(y_train)
        fraud_rate = (y_train == 1).sum() / len(y_train)
        logger.info(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
        logger.info(f"Fraud Rate: {fraud_rate*100:.4f}% ({(y_train == 1).sum():,} fraud samples)")
        
        # Remove ID and timestamp features
        drop_cols = feature_types['id'] + feature_types['timestamp']
        if drop_cols:
            logger.info(f"Removing {len(drop_cols)} ID/timestamp columns")
            X_train = X_train.drop(columns=drop_cols, errors='ignore')
            X_test = X_test.drop(columns=drop_cols, errors='ignore')
        
        # Encode categorical features (OPTIMIZED for large datasets)
        if encode_categorical and feature_types['categorical']:
            logger.info(f"Encoding {len(feature_types['categorical'])} categorical features")
            
            # Use optimized encoding for large datasets (> 500K samples)
            if len(X_train) > 500000:
                logger.info("⚡ Using optimized batch encoding for large dataset")
                
                for col in feature_types['categorical']:
                    if col in X_train.columns:
                        # Use pandas Categorical for memory efficiency
                        logger.info(f"  Encoding: {col}")
                        
                        # Convert to categorical (much faster than LabelEncoder for large data)
                        X_train[col] = X_train[col].astype('category')
                        
                        # Get the categories from training set
                        train_categories = X_train[col].cat.categories
                        
                        # Apply same categories to test set
                        X_test[col] = pd.Categorical(
                            X_test[col], 
                            categories=train_categories
                        )
                        
                        # Convert to codes (numerical)
                        X_train[col] = X_train[col].cat.codes
                        X_test[col] = X_test[col].cat.codes  # Unseen categories become -1
                        
                        # Store for reference (though we don't use LabelEncoder here)
                        self.label_encoders[col] = train_categories
                        
                        clear_memory()
            else:
                # Original encoding for smaller datasets
                for col in feature_types['categorical']:
                    if col in X_train.columns:
                        le = LabelEncoder()
                        le.fit(X_train[col].astype(str))
                        X_train[col] = le.transform(X_train[col].astype(str))
                        # Handle unseen categories in test set
                        X_test[col] = X_test[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                        self.label_encoders[col] = le
        
        # Handle sparse features
        X_train, X_test = self.handle_sparse_features(X_train, X_test)
        
        # Smart sampling (on training set only)
        if apply_sampling and self.max_samples and len(X_train) > self.max_samples:
            X_train, y_train = smart_sample(X_train, y_train, self.max_samples)
        
        # Ensure y is 1D Series
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
        
        # Reset index
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Standardize features
        logger.info("Standardizing numerical features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        logger.info(f"Preprocessing complete!")
        logger.info(f"  Final features: {X_train_scaled.shape[1]}")
        logger.info(f"  Training samples: {X_train_scaled.shape[0]:,}")
        logger.info(f"  Test samples: {X_test_scaled.shape[0]:,}")
        
        train_fraud_rate = (y_train == 1).sum() / len(y_train)
        test_fraud_rate = (y_test == 1).sum() / len(y_test)
        logger.info(f"  Train fraud rate: {train_fraud_rate:.4f}")
        logger.info(f"  Test fraud rate: {test_fraud_rate:.4f}")
        
        clear_memory()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_types
    
    def generate_dataset_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               feature_types: Dict[str, List[str]]):
        """
        Generate comprehensive dataset analysis report with visualizations
        
        Focus on:
        - Correlation with fraud label
        - Train/Test distribution comparison
        - Individual fraud distribution analysis per feature
        """
        logger.section(f"Generating Dataset Report: {self.dataset_name}", level=2)
        
        # Create output directory
        dataset_viz_dir = VIZ_DIR / 'datasets' / self.dataset_name
        dataset_viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Dataset statistics
        stats = {
            'dataset_name': self.dataset_name,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(test_df),
            'num_features': X_train.shape[1],
            'numerical_features': len(feature_types['numerical']),
            'categorical_features': len(feature_types['categorical']),
            'train_fraud_count': (y_train == 1).sum(),
            'train_fraud_rate': (y_train == 1).sum() / len(y_train),
            'test_fraud_count': (y_test == 1).sum(),
            'test_fraud_rate': (y_test == 1).sum() / len(y_test),
            'imbalance_ratio': ImbalanceHandler.get_imbalance_ratio(y_train)
        }
        
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"  Features: {stats['num_features']}")
        logger.info(f"  Fraud rate (train): {stats['train_fraud_rate']*100:.4f}%")
        logger.info(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
        
        # ====================================================================
        # MAIN VISUALIZATION: Overview with Fraud Correlation Focus
        # ====================================================================
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. Train/Test Class Distribution Comparison (Combined)
        ax1 = fig.add_subplot(gs[0, 0])
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        
        x = np.arange(2)
        width = 0.35
        
        train_values = [train_counts.get(0, 0), train_counts.get(1, 0)]
        test_values = [test_counts.get(0, 0), test_counts.get(1, 0)]
        
        bars1 = ax1.bar(x - width/2, train_values, width, label='Train', 
                       color=['skyblue', 'coral'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_values, width, label='Test',
                       color=['lightblue', 'salmon'], alpha=0.8)
        
        ax1.set_title('Train vs Test Class Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Normal', 'Fraud'])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        # 2. Feature Type Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        feature_type_counts = {
            'Numerical': len(feature_types['numerical']),
            'Categorical': len(feature_types['categorical'])
        }
        # Remove zero-count types
        feature_type_counts = {k: v for k, v in feature_type_counts.items() if v > 0}
        
        colors_ft = ['#2ecc71', '#3498db'][:len(feature_type_counts)]
        ax2.pie(feature_type_counts.values(), labels=feature_type_counts.keys(),
               autopct='%1.1f%%', startangle=90, colors=colors_ft)
        ax2.set_title('Feature Type Distribution', fontsize=12, fontweight='bold')
        
        # 3. Dataset Summary Text
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        summary_text = f"""
Dataset: {self.dataset_name}
{'='*40}

Samples:
  Train: {stats['train_samples']:,}
  Test: {stats['test_samples']:,}
  Total: {stats['total_samples']:,}

Features:
  Total: {stats['num_features']}
  Numerical: {stats['numerical_features']}
  Categorical: {stats['categorical_features']}

Class Distribution:
  Train Fraud: {stats['train_fraud_rate']*100:.4f}%
  Test Fraud: {stats['test_fraud_rate']*100:.4f}%
  Imbalance: {stats['imbalance_ratio']:.2f}:1

Type: {DATASET_CONFIGS[self.dataset_name]['size_category'].upper()}
        """
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. Correlation with Fraud Label (Most Important!)
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Combine features with label for correlation analysis
        if isinstance(X_train, pd.DataFrame):
            X_train_with_label = X_train.copy()
        else:
            X_train_with_label = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        
        X_train_with_label['is_fraud'] = y_train.values
        
        # Calculate correlation with fraud
        corr_with_fraud = X_train_with_label.corr()['is_fraud'].drop('is_fraud')
        
        # Sort by absolute correlation and take top 20
        top_corr = corr_with_fraud.abs().nlargest(20)
        top_features = top_corr.index
        top_corr_values = corr_with_fraud[top_features].sort_values()
        
        # Plot
        colors = ['red' if x < 0 else 'green' for x in top_corr_values]
        top_corr_values.plot(kind='barh', ax=ax4, color=colors, alpha=0.7)
        ax4.set_title('Top 20 Features by Correlation with Fraud Label', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Correlation Coefficient', fontsize=10)
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Feature Correlation Heatmap (including fraud label)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Select top correlated features with fraud for heatmap
        n_features_heatmap = min(15, len(top_features))
        selected_features = list(top_corr.head(n_features_heatmap).index) + ['is_fraud']
        
        corr_matrix = X_train_with_label[selected_features].corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=ax5,
                   cbar_kws={'label': 'Correlation'}, annot=False,
                   xticklabels=True, yticklabels=True)
        ax5.set_title(f'Feature Correlation Matrix\n(Top {n_features_heatmap} + Fraud Label)',
                     fontsize=11, fontweight='bold')
        ax5.tick_params(axis='both', labelsize=8)
        
        # 6. Feature Variance Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        if isinstance(X_train, pd.DataFrame):
            feature_vars = X_train.var().sort_values(ascending=False)
        else:
            feature_vars = pd.Series(np.var(X_train, axis=0)).sort_values(ascending=False)
        
        n_vars_to_show = min(20, len(feature_vars))
        ax6.bar(range(n_vars_to_show), feature_vars.head(n_vars_to_show).values,
               color='steelblue', alpha=0.7)
        ax6.set_title('Top 20 Features by Variance', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Feature Rank', fontsize=10)
        ax6.set_ylabel('Variance', fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Fraud Rate by Feature Quartiles (for top correlated feature)
        ax7 = fig.add_subplot(gs[2, 1])
        
        if len(top_features) > 0:
            top_feature_name = top_features[0]
            if isinstance(X_train, pd.DataFrame):
                feature_values = X_train[top_feature_name]
            else:
                feature_idx = int(top_feature_name.split('_')[-1]) if 'feature_' in str(top_feature_name) else 0
                feature_values = X_train[:, feature_idx]
            
            # Create quartiles
            quartiles = pd.qcut(feature_values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            fraud_rates = []
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = quartiles == q
                if mask.sum() > 0:
                    fraud_rate = y_train[mask].mean()
                    fraud_rates.append(fraud_rate * 100)
                else:
                    fraud_rates.append(0)
            
            ax7.bar(['Q1', 'Q2', 'Q3', 'Q4'], fraud_rates, color='coral', alpha=0.7)
            ax7.set_title(f'Fraud Rate by Quartiles\n(Feature: {top_feature_name})',
                         fontsize=11, fontweight='bold')
            ax7.set_ylabel('Fraud Rate (%)', fontsize=10)
            ax7.set_xlabel('Quartile', fontsize=10)
            ax7.grid(axis='y', alpha=0.3)
            
            for i, v in enumerate(fraud_rates):
                ax7.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # 8. Class Imbalance Visualization
        ax8 = fig.add_subplot(gs[2, 2])
        
        normal_count = (y_train == 0).sum()
        fraud_count = (y_train == 1).sum()
        
        ax8.barh(['Normal', 'Fraud'], [normal_count, fraud_count],
                color=['skyblue', 'coral'], alpha=0.8)
        ax8.set_title('Class Imbalance (Training Set)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Count', fontsize=10)
        ax8.grid(axis='x', alpha=0.3)
        
        # Add ratio annotation
        ax8.text(normal_count/2, 0, f'{normal_count:,}', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax8.text(fraud_count/2, 1, f'{fraud_count:,}',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax8.text(0.95, 0.95, f'Ratio: {stats["imbalance_ratio"]:.1f}:1',
                transform=ax8.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.suptitle(f'Dataset Analysis Report: {self.dataset_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save main figure
        output_path = dataset_viz_dir / 'dataset_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Main dataset report saved: {output_path}")
        
        # ====================================================================
        # INDIVIDUAL FRAUD DISTRIBUTION ANALYSIS (Separate Plots)
        # ====================================================================
        logger.info("Generating individual fraud distribution plots for top features...")
        
        # Select top 8 features most correlated with fraud
        n_features_detail = min(8, len(top_features))
        top_features_for_detail = list(top_corr.head(n_features_detail).index)
        
        for feature_name in top_features_for_detail:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Get feature values
                if isinstance(X_train, pd.DataFrame):
                    feature_values = X_train[feature_name].values
                else:
                    feature_idx = int(feature_name.split('_')[-1]) if 'feature_' in str(feature_name) else 0
                    feature_values = X_train[:, feature_idx]
                
                fraud_mask = y_train.values == 1
                normal_mask = y_train.values == 0
                
                # 1. Distribution comparison (KDE)
                ax = axes[0, 0]
                try:
                    # Normal transactions
                    ax.hist(feature_values[normal_mask], bins=50, alpha=0.5, 
                           label='Normal', color='skyblue', density=True)
                    # Fraud transactions
                    ax.hist(feature_values[fraud_mask], bins=50, alpha=0.5,
                           label='Fraud', color='coral', density=True)
                    ax.set_xlabel('Feature Value', fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.set_title('Distribution: Normal vs Fraud', fontsize=11, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                except:
                    ax.text(0.5, 0.5, 'Unable to plot distribution', 
                           ha='center', va='center', transform=ax.transAxes)
                
                # 2. Box plot comparison
                ax = axes[0, 1]
                data_to_plot = [feature_values[normal_mask], feature_values[fraud_mask]]
                bp = ax.boxplot(data_to_plot, labels=['Normal', 'Fraud'], patch_artist=True)
                bp['boxes'][0].set_facecolor('skyblue')
                bp['boxes'][1].set_facecolor('coral')
                ax.set_ylabel('Feature Value', fontsize=10)
                ax.set_title('Box Plot Comparison', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 3. Fraud rate by value bins
                ax = axes[1, 0]
                try:
                    # Create 10 bins
                    bins = pd.qcut(feature_values, q=10, labels=False, duplicates='drop')
                    fraud_rates_by_bin = []
                    bin_labels = []
                    
                    for bin_idx in sorted(np.unique(bins)):
                        mask = bins == bin_idx
                        if mask.sum() > 0:
                            fraud_rate = y_train.values[mask].mean() * 100
                            fraud_rates_by_bin.append(fraud_rate)
                            bin_labels.append(f'B{bin_idx+1}')
                    
                    ax.bar(bin_labels, fraud_rates_by_bin, color='coral', alpha=0.7)
                    ax.set_xlabel('Value Bin', fontsize=10)
                    ax.set_ylabel('Fraud Rate (%)', fontsize=10)
                    ax.set_title('Fraud Rate by Value Bins', fontsize=11, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels
                    for i, v in enumerate(fraud_rates_by_bin):
                        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
                except:
                    ax.text(0.5, 0.5, 'Unable to create bins',
                           ha='center', va='center', transform=ax.transAxes)
                
                # 4. Statistics comparison table
                ax = axes[1, 1]
                ax.axis('off')
                
                # Calculate statistics
                normal_stats = {
                    'Mean': np.mean(feature_values[normal_mask]),
                    'Median': np.median(feature_values[normal_mask]),
                    'Std': np.std(feature_values[normal_mask]),
                    'Min': np.min(feature_values[normal_mask]),
                    'Max': np.max(feature_values[normal_mask])
                }
                
                fraud_stats = {
                    'Mean': np.mean(feature_values[fraud_mask]),
                    'Median': np.median(feature_values[fraud_mask]),
                    'Std': np.std(feature_values[fraud_mask]),
                    'Min': np.min(feature_values[fraud_mask]),
                    'Max': np.max(feature_values[fraud_mask])
                }
                
                stats_text = f"""
Feature: {feature_name}
Correlation with Fraud: {corr_with_fraud[feature_name]:.4f}
{'='*50}

                Normal          Fraud
{'─'*50}
Mean      {normal_stats['Mean']:12.4f}  {fraud_stats['Mean']:12.4f}
Median    {normal_stats['Median']:12.4f}  {fraud_stats['Median']:12.4f}
Std Dev   {normal_stats['Std']:12.4f}  {fraud_stats['Std']:12.4f}
Min       {normal_stats['Min']:12.4f}  {fraud_stats['Min']:12.4f}
Max       {normal_stats['Max']:12.4f}  {fraud_stats['Max']:12.4f}

Sample Counts:
  Normal: {normal_mask.sum():,}
  Fraud: {fraud_mask.sum():,}
                """
                
                ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                plt.suptitle(f'Fraud Distribution Analysis: {feature_name}',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save individual feature plot
                feature_output_path = dataset_viz_dir / f'fraud_distribution_{feature_name}.png'
                plt.savefig(feature_output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"  Saved: fraud_distribution_{feature_name}.png")
                
            except Exception as e:
                logger.warning(f"  Failed to create plot for {feature_name}: {e}")
                plt.close()
        
        logger.info(f"Individual fraud distribution plots completed!")
        
        # Save statistics as JSON (convert numpy types to Python native types)
        stats_path = dataset_viz_dir / 'statistics.json'
        try:
            # Convert numpy types to Python native types for JSON serialization
            stats_json = {}
            for key, value in stats.items():
                if hasattr(value, 'item'):  # numpy scalar
                    stats_json[key] = value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    stats_json[key] = value.item()
                else:
                    stats_json[key] = value
            
            with open(stats_path, 'w') as f:
                json.dump(stats_json, f, indent=2)
            logger.info(f"Dataset statistics saved: {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save statistics JSON: {e}")
        
        return stats


# ============================================================================
# 8. DEEP LEARNING MODELS
# ============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron Classifier"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    """Autoencoder for Anomaly Detection"""
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [128, 64, 32]):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for encoding_dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, encoding_dim),
                nn.BatchNorm1d(encoding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = encoding_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        for decoding_dim in decoding_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, decoding_dim),
                nn.BatchNorm1d(decoding_dim) if decoding_dim != input_dim else nn.Identity(),
                nn.ReLU() if decoding_dim != input_dim else nn.Identity()
            ])
            prev_dim = decoding_dim
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


def train_mlp(model: nn.Module, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series,
             epochs: int = 50, batch_size: int = 512,
             lr: float = 0.001, patience: int = 10,
             use_multi_gpu: bool = True) -> nn.Module:
    """Train MLP model with multi-GPU support"""
    logger.info("Training MLP model...")
    
    # GPU configuration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
        
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    # Data preparation
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.values),
        torch.FloatTensor(y_train.values).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(device)
    
    # Calculate class weights
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            if isinstance(model, nn.DataParallel):
                outputs = model.module.network[:-1](batch_X)
            else:
                outputs = model.network[:-1](batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                val_outputs = model.module.network[:-1](X_val_tensor)
            else:
                val_outputs = model.network[:-1](X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    clear_memory()
    return model


def train_autoencoder(model: nn.Module, X_train: pd.DataFrame,
                     epochs: int = 50, batch_size: int = 512,
                     lr: float = 0.001, use_multi_gpu: bool = True) -> nn.Module:
    """Train Autoencoder model"""
    logger.info("Training Autoencoder model...")
    
    # GPU configuration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
        
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    # Data preparation (normal samples only)
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, in train_loader:
            batch_X = batch_X.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}")
    
    clear_memory()
    return model
