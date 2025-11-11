"""
Model Training and Evaluation Module
====================================

This module contains all model training functions and the performance evaluator.
It supports both supervised and unsupervised learning models with comprehensive
evaluation metrics.

Models included:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Support Vector Machine (with PCA)
- Multi-Layer Perceptron
- Isolation Forest
- Autoencoder

Author: Experiment Framework
Date: 2025-11-11
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

import xgboost as xgb
import lightgbm as lgb

# Import from main experiment file
from experiment_refactored import (
    logger, clear_memory, smart_sample,
    MLPClassifier, Autoencoder, train_mlp, train_autoencoder,
    IMBALANCE_STRATEGIES
)


# ============================================================================
# MODEL PARAMETER CONFIGURATIONS
# ============================================================================

def get_model_params(model_name: str, dataset_size: str, n_features: int) -> Dict[str, Any]:
    """
    Get adaptive model parameters based on dataset characteristics
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    dataset_size : str
        Size category: 'small', 'medium', 'large'
    n_features : int
        Number of features in the dataset
    
    Returns:
    --------
    params : dict
        Model parameters
    """
    # Logistic Regression
    if model_name == 'logistic_regression':
        if dataset_size == 'small':
            return {'max_iter': 500, 'C': 1.0, 'solver': 'lbfgs'}
        elif dataset_size == 'medium':
            return {'max_iter': 1000, 'C': 1.0, 'solver': 'saga'}
        else:  # large
            return {'max_iter': 1000, 'C': 0.1, 'solver': 'saga'}
    
    # Random Forest
    elif model_name == 'random_forest':
        if dataset_size == 'small':
            return {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
        elif dataset_size == 'medium':
            return {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 5}
        else:  # large
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10}
    
    # XGBoost
    elif model_name == 'xgboost':
        if dataset_size == 'small':
            return {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1}
        elif dataset_size == 'medium':
            return {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1}
        else:  # large
            return {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    
    # LightGBM
    elif model_name == 'lightgbm':
        if dataset_size == 'small':
            return {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'num_leaves': 31}
        elif dataset_size == 'medium':
            return {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 31}
        else:  # large
            return {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31}
    
    # KNN
    elif model_name == 'knn':
        if dataset_size == 'small':
            return {'n_neighbors': 5}
        elif dataset_size == 'medium':
            return {'n_neighbors': 7}
        else:  # large
            return {'n_neighbors': 10}
    
    # SVM
    elif model_name == 'svm':
        if dataset_size == 'small':
            return {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
        else:
            return {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'}
    
    # MLP
    elif model_name == 'mlp':
        if n_features < 20:
            hidden_dims = [64, 32]
        elif n_features < 50:
            hidden_dims = [128, 64, 32]
        else:
            hidden_dims = [256, 128, 64]
        
        if dataset_size == 'small':
            return {'hidden_dims': hidden_dims, 'epochs': 100, 'batch_size': 256}
        elif dataset_size == 'medium':
            return {'hidden_dims': hidden_dims, 'epochs': 50, 'batch_size': 512}
        else:  # large
            return {'hidden_dims': hidden_dims, 'epochs': 30, 'batch_size': 1024}
    
    # Isolation Forest
    elif model_name == 'isolation_forest':
        if dataset_size == 'small':
            return {'n_estimators': 200, 'max_samples': 'auto'}
        elif dataset_size == 'medium':
            return {'n_estimators': 150, 'max_samples': 256}
        else:  # large
            return {'n_estimators': 100, 'max_samples': 256}
    
    # Autoencoder
    elif model_name == 'autoencoder':
        if n_features < 20:
            encoding_dims = [32, 16]
        elif n_features < 50:
            encoding_dims = [64, 32, 16]
        else:
            encoding_dims = [128, 64, 32]
        
        if dataset_size == 'small':
            return {'encoding_dims': encoding_dims, 'epochs': 100, 'batch_size': 256}
        elif dataset_size == 'medium':
            return {'encoding_dims': encoding_dims, 'epochs': 50, 'batch_size': 512}
        else:  # large
            return {'encoding_dims': encoding_dims, 'epochs': 30, 'batch_size': 1024}
    
    return {}


# ============================================================================
# PERFORMANCE EVALUATOR
# ============================================================================

class PerformanceEvaluator:
    """Evaluates model performance with comprehensive metrics"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_supervised(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray, model_name: str,
                          dataset_name: str, train_time: float,
                          inference_time: float, imbalance_strategy: str = 'none') -> Dict:
        """Evaluate supervised learning model"""
        try:
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'imbalance_strategy': imbalance_strategy,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
                'train_time': train_time,
                'inference_time': inference_time
            }
            
            self.results.append(result)
            self._log_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return None
    
    def evaluate_unsupervised(self, y_true: np.ndarray, anomaly_scores: np.ndarray,
                            model_name: str, dataset_name: str,
                            train_time: float, inference_time: float,
                            contamination: float, imbalance_strategy: str = 'none') -> Dict:
        """Evaluate unsupervised learning model (anomaly detection)"""
        try:
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (anomaly_scores > optimal_threshold).astype(int)
            
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'imbalance_strategy': imbalance_strategy,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, anomaly_scores) if len(np.unique(y_true)) > 1 else 0.0,
                'train_time': train_time,
                'inference_time': inference_time
            }
            
            self.results.append(result)
            self._log_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return None
    
    def _log_result(self, result: Dict):
        """Log evaluation result"""
        logger.info(f"Model: {result['model']} - Strategy: {result['imbalance_strategy']}")
        logger.metric("Accuracy", result['accuracy'])
        logger.metric("Precision", result['precision'])
        logger.metric("Recall", result['recall'])
        logger.metric("F1-Score", result['f1_score'])
        logger.metric("ROC-AUC", result['roc_auc'])
        logger.info(f"  Training Time: {result['train_time']:.2f}s")
        logger.info(f"  Inference Time: {result['inference_time']:.4f}s")
    
    def get_results_df(self) -> pd.DataFrame:
        """Get all results as DataFrame"""
        return pd.DataFrame(self.results)
    
    def clear_results(self):
        """Clear all stored results"""
        self.results = []


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Main experiment runner with all models"""
    
    def __init__(self, compare_imbalance: bool = True, 
                 use_sampling_for_slow_models: bool = True):
        self.evaluator = PerformanceEvaluator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compare_imbalance = compare_imbalance
        self.use_sampling_for_slow_models = use_sampling_for_slow_models
    
    def _apply_imbalance_strategy(self, X: pd.DataFrame, y: pd.Series,
                                 strategy: str = 'none') -> Tuple[pd.DataFrame, pd.Series]:
        """Apply imbalance handling strategy"""
        if strategy == 'none' or IMBALANCE_STRATEGIES[strategy]['handler'] is None:
            return X, y
        
        logger.info(f"Applying imbalance strategy: {IMBALANCE_STRATEGIES[strategy]['name']}")
        strategy_func = IMBALANCE_STRATEGIES[strategy]['handler']
        return strategy_func(X, y)
    
    def run_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               dataset_name: str, dataset_size: str,
                               imbalance_strategy: str = 'none') -> Dict:
        """Train and evaluate Logistic Regression"""
        logger.section(f"Logistic Regression [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        params = get_model_params('logistic_regression', dataset_size, X_train.shape[1])
        
        logger.timer_start('lr_train')
        model = LogisticRegression(
            max_iter=params['max_iter'],
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('lr_train')
        
        logger.timer_start('lr_inference')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = logger.timer_end('lr_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'Logistic Regression', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         dataset_name: str, dataset_size: str,
                         imbalance_strategy: str = 'none') -> Dict:
        """Train and evaluate Random Forest"""
        logger.section(f"Random Forest [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        params = get_model_params('random_forest', dataset_size, X_train.shape[1])
        
        logger.timer_start('rf_train')
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params.get('max_depth'),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('rf_train')
        
        logger.timer_start('rf_inference')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = logger.timer_end('rf_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'Random Forest', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   dataset_name: str, dataset_size: str,
                   imbalance_strategy: str = 'none') -> Dict:
        """Train and evaluate XGBoost"""
        logger.section(f"XGBoost [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        params = get_model_params('xgboost', dataset_size, X_train.shape[1])
        scale_pos_weight = (y_train_proc == 0).sum() / max((y_train_proc == 1).sum(), 1)
        
        logger.timer_start('xgb_train')
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            gpu_id=0 if torch.cuda.is_available() else None
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('xgb_train')
        
        logger.timer_start('xgb_inference')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = logger.timer_end('xgb_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'XGBoost', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    dataset_name: str, dataset_size: str,
                    imbalance_strategy: str = 'none') -> Dict:
        """Train and evaluate LightGBM"""
        logger.section(f"LightGBM [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        params = get_model_params('lightgbm', dataset_size, X_train.shape[1])
        scale_pos_weight = (y_train_proc == 0).sum() / max((y_train_proc == 1).sum(), 1)
        
        logger.timer_start('lgb_train')
        model = lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            num_leaves=params.get('num_leaves', 31),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            device='cpu',
            n_jobs=-1
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('lgb_train')
        
        logger.timer_start('lgb_inference')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = logger.timer_end('lgb_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'LightGBM', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_knn(self, X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               dataset_name: str, dataset_size: str,
               imbalance_strategy: str = 'none', max_samples: int = 20000) -> Dict:
        """Train and evaluate KNN (with smart sampling for large datasets)"""
        logger.section(f"K-Nearest Neighbors [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        # Smart sampling for large datasets
        if self.use_sampling_for_slow_models and len(X_train) > max_samples:
            X_train_sampled, y_train_sampled = smart_sample(X_train, y_train, max_samples)
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(
            X_train_sampled, y_train_sampled, imbalance_strategy
        )
        
        params = get_model_params('knn', dataset_size, X_train.shape[1])
        
        logger.timer_start('knn_train')
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], n_jobs=-1)
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('knn_train')
        
        logger.timer_start('knn_inference')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = logger.timer_end('knn_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'KNN', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_pca_svm(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   dataset_name: str, dataset_size: str,
                   imbalance_strategy: str = 'none',
                   n_components: float = 0.95, max_samples: int = 20000) -> Dict:
        """Train and evaluate PCA + SVM (with smart sampling for large datasets)"""
        logger.section(f"PCA + SVM [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        # Smart sampling for large datasets
        if self.use_sampling_for_slow_models and len(X_train) > max_samples:
            X_train_sampled, y_train_sampled = smart_sample(X_train, y_train, max_samples)
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        logger.timer_start('pca_svm_train')
        
        # PCA dimensionality reduction
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_sampled)
        X_test_pca = pca.transform(X_test)
        logger.info(f"PCA: {X_train_sampled.shape[1]} -> {X_train_pca.shape[1]} dimensions")
        
        # Apply imbalance handling
        X_train_pca_df = pd.DataFrame(X_train_pca)
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(
            X_train_pca_df, y_train_sampled, imbalance_strategy
        )
        
        # SVM classification
        params = get_model_params('svm', dataset_size, X_train.shape[1])
        model = SVC(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = logger.timer_end('pca_svm_train')
        
        logger.timer_start('pca_svm_inference')
        y_pred = model.predict(X_test_pca)
        y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
        inference_time = logger.timer_end('pca_svm_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'PCA+SVM', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del pca, model, X_train_proc, y_train_proc, X_train_pca, X_test_pca
        clear_memory()
        return result
    
    def run_mlp(self, X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series,
               dataset_name: str, dataset_size: str,
               imbalance_strategy: str = 'none') -> Dict:
        """Train and evaluate MLP"""
        logger.section(f"Multi-Layer Perceptron [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]", level=3)
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_proc, y_train_proc, test_size=0.2, random_state=42,
            stratify=y_train_proc if len(np.unique(y_train_proc)) > 1 else None
        )
        
        params = get_model_params('mlp', dataset_size, X_train.shape[1])
        
        logger.timer_start('mlp_train')
        model = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=params['hidden_dims'])
        model = train_mlp(model, X_train_split, y_train_split, X_val, y_val,
                         epochs=params['epochs'], batch_size=params['batch_size'],
                         use_multi_gpu=True)
        train_time = logger.timer_end('mlp_train')
        
        # Prediction
        model.eval()
        logger.timer_start('mlp_inference')
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
            if isinstance(model, nn.DataParallel):
                y_pred_proba = model.module(X_test_tensor).cpu().numpy().flatten()
            else:
                y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        inference_time = logger.timer_end('mlp_inference')
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'MLP', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_isolation_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            dataset_name: str, dataset_size: str) -> Dict:
        """Train and evaluate Isolation Forest"""
        logger.section("Isolation Forest (Unsupervised)", level=3)
        
        contamination = y_train.mean()
        params = get_model_params('isolation_forest', dataset_size, X_train.shape[1])
        
        logger.timer_start('if_train')
        model = IsolationForest(
            contamination=contamination,
            n_estimators=params['n_estimators'],
            max_samples=params.get('max_samples', 'auto'),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train)
        train_time = logger.timer_end('if_train')
        
        logger.timer_start('if_inference')
        anomaly_scores = -model.score_samples(X_test)
        inference_time = logger.timer_end('if_inference')
        
        result = self.evaluator.evaluate_unsupervised(
            y_test, anomaly_scores, 'Isolation Forest', dataset_name,
            train_time, inference_time, contamination=contamination
        )
        
        del model
        clear_memory()
        return result
    
    def run_autoencoder(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       dataset_name: str, dataset_size: str) -> Dict:
        """Train and evaluate Autoencoder"""
        logger.section("Autoencoder (Unsupervised)", level=3)
        
        # Use only normal samples for training
        X_train_normal = X_train[y_train == 0]
        logger.info(f"Using {len(X_train_normal):,} normal samples for training")
        
        params = get_model_params('autoencoder', dataset_size, X_train.shape[1])
        
        logger.timer_start('ae_train')
        model = Autoencoder(input_dim=X_train.shape[1], encoding_dims=params['encoding_dims'])
        model = train_autoencoder(model, X_train_normal, epochs=params['epochs'],
                                 batch_size=params['batch_size'], use_multi_gpu=True)
        train_time = logger.timer_end('ae_train')
        
        # Calculate reconstruction error
        model.eval()
        logger.timer_start('ae_inference')
        X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
        if isinstance(model, nn.DataParallel):
            anomaly_scores = model.module.get_reconstruction_error(X_test_tensor)
        else:
            anomaly_scores = model.get_reconstruction_error(X_test_tensor)
        inference_time = logger.timer_end('ae_inference')
        
        result = self.evaluator.evaluate_unsupervised(
            y_test, anomaly_scores, 'Autoencoder', dataset_name,
            train_time, inference_time, contamination=y_train.mean()
        )
        
        del model
        clear_memory()
        return result
    
    def run_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      dataset_name: str, dataset_size: str) -> Dict[str, Dict]:
        """Run all models on a dataset"""
        logger.section(f"Running All Models on Dataset: {dataset_name}", level=1)
        
        # Determine imbalance strategies to test
        if self.compare_imbalance:
            strategies = ['none', 'smote', 'adasyn']
            logger.info(f"Testing imbalance strategies: {[IMBALANCE_STRATEGIES[s]['name'] for s in strategies]}")
        else:
            strategies = ['none']
        
        results = {}
        
        # Supervised models with imbalance handling
        for strategy in strategies:
            logger.section(f"Imbalance Strategy: {IMBALANCE_STRATEGIES[strategy]['name']}", level=2)
            
            # Fast models (can handle large datasets)
            results[f'lr_{strategy}'] = self.run_logistic_regression(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            results[f'rf_{strategy}'] = self.run_random_forest(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            results[f'xgb_{strategy}'] = self.run_xgboost(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            results[f'lgb_{strategy}'] = self.run_lightgbm(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            results[f'mlp_{strategy}'] = self.run_mlp(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            
            # Slow models (with smart sampling for large datasets)
            results[f'knn_{strategy}'] = self.run_knn(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
            results[f'pca_svm_{strategy}'] = self.run_pca_svm(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size, strategy
            )
        
        # Unsupervised models (run once, independent of imbalance handling)
        logger.section("Unsupervised Models", level=2)
        results['isolation_forest'] = self.run_isolation_forest(
            X_train, y_train, X_test, y_test, dataset_name, dataset_size
        )
        results['autoencoder'] = self.run_autoencoder(
            X_train, y_train, X_test, y_test, dataset_name, dataset_size
        )
        
        logger.section(f"Completed All Models for {dataset_name}", level=1)
        return results
