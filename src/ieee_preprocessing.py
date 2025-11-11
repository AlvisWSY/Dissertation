"""
Improved Preprocessing for High-Dimensional Datasets like IEEE
==============================================================

This module provides advanced preprocessing specifically designed
for high-dimensional fraud detection datasets.

Features:
- Intelligent feature selection using mutual information
- Target encoding for categorical features
- Proper missing value handling
- Variance-based filtering

Author: Experiment Framework
Date: 2025-11-11
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class HighDimensionalPreprocessor:
    """
    Advanced preprocessing for high-dimensional datasets
    
    This preprocessor is specifically designed for datasets with:
    - High feature count (> 100 features)
    - Low samples-to-features ratio (< 50)
    - Many categorical features
    - Severe class imbalance
    
    Features:
    - Intelligent feature selection
    - Target encoding for categoricals (avoids dimension explosion)
    - Proper missing value handling
    - Variance-based filtering
    """
    
    def __init__(self, 
                 target_n_features: int = 150,
                 variance_threshold: float = 0.01,
                 max_missing_ratio: float = 0.7):
        """
        Parameters:
        -----------
        target_n_features : int
            Target number of features to keep (default: 150)
        variance_threshold : float
            Minimum variance for feature to be kept (default: 0.01)
        max_missing_ratio : float
            Maximum ratio of missing values allowed (default: 0.7)
        """
        self.target_n_features = target_n_features
        self.variance_threshold = variance_threshold
        self.max_missing_ratio = max_missing_ratio
        
        self.feature_selector = None
        self.variance_selector = None
        self.scaler = None
        
        self.selected_features = None
        self.feature_importance = None
        self.removed_high_missing = []
        self.removed_low_variance = 0
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical features"""
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical, categorical
    
    def _remove_low_variance_features(self, X: np.ndarray) -> np.ndarray:
        """Remove features with very low variance"""
        from experiment_refactored import logger as exp_logger
        exp_logger.info(f"Removing features with variance < {self.variance_threshold}")
        
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_filtered = selector.fit_transform(X)
        
        self.removed_low_variance = X.shape[1] - X_filtered.shape[1]
        exp_logger.info(f"Removed {self.removed_low_variance} low-variance features")
        
        self.variance_selector = selector
        return X_filtered
    
    def _remove_high_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with too many missing values"""
        from experiment_refactored import logger as exp_logger
        
        missing_ratios = df.isnull().sum() / len(df)
        high_missing = missing_ratios[missing_ratios > self.max_missing_ratio].index.tolist()
        
        if len(high_missing) > 0:
            exp_logger.info(f"Removing {len(high_missing)} features with >{self.max_missing_ratio*100:.0f}% missing values")
            self.removed_high_missing = high_missing
            df = df.drop(columns=high_missing)
        
        return df
    
    def _select_best_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select top K features using mutual information"""
        from experiment_refactored import logger as exp_logger
        
        # Ensure we don't try to select more features than available
        k = min(self.target_n_features, X.shape[1])
        
        exp_logger.info(f"Selecting top {k} features using mutual information...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get feature importance scores
        self.feature_importance = pd.DataFrame({
            'feature_idx': range(len(selector.scores_)),
            'importance': selector.scores_
        }).sort_values('importance', ascending=False)
        
        self.feature_selector = selector
        exp_logger.info(f"Selected {X_selected.shape[1]} features (reduced from {X.shape[1]})")
        
        return X_selected
    
    def _simple_target_encode(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                             categorical_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple target encoding for categorical features"""
        from experiment_refactored import logger as exp_logger
        
        if len(categorical_cols) == 0:
            return train_df, test_df
        
        exp_logger.info(f"Target encoding {len(categorical_cols)} categorical features")
        
        y_train = train_df[target_col]
        
        for col in categorical_cols:
            # Calculate mean target value for each category
            means = train_df.groupby(col)[target_col].mean()
            global_mean = y_train.mean()
            
            # Add smoothing (5 samples minimum)
            counts = train_df.groupby(col).size()
            smoothing = 1.0
            smoothed_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
            
            # Apply to train
            train_df[col] = train_df[col].map(smoothed_means).fillna(global_mean)
            
            # Apply to test (unseen categories get global mean)
            test_df[col] = test_df[col].map(smoothed_means).fillna(global_mean)
        
        return train_df, test_df
    
    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     target_col: str = 'is_fraud') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        train_df : DataFrame
            Training data with target column
        test_df : DataFrame
            Test data with target column
        target_col : str
            Name of target column (default: 'is_fraud')
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
        """
        from experiment_refactored import logger as exp_logger
        
        exp_logger.section("HIGH-DIMENSIONAL PREPROCESSING PIPELINE", level=2)
        
        # Separate features and target
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        
        initial_shape = X_train.shape
        exp_logger.info(f"Initial shape: {initial_shape}")
        exp_logger.info(f"Samples-to-features ratio: {initial_shape[0]/initial_shape[1]:.2f}")
        
        # Step 1: Remove high missing features
        exp_logger.info("\n[Step 1/6] Removing high-missing features...")
        X_train = self._remove_high_missing_features(X_train)
        X_test = X_test[X_train.columns]
        exp_logger.info(f"Shape after removing high-missing: {X_train.shape}")
        
        # Step 2: Identify feature types
        exp_logger.info("\n[Step 2/6] Identifying feature types...")
        numerical_cols, categorical_cols = self._identify_feature_types(X_train)
        exp_logger.info(f"Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")
        
        # Step 3: Encode categoricals
        if len(categorical_cols) > 0:
            exp_logger.info("\n[Step 3/6] Encoding categorical features with target encoding...")
            # Temporarily add target back for encoding
            train_with_target = X_train.copy()
            train_with_target[target_col] = y_train
            test_with_target = X_test.copy()
            test_with_target[target_col] = y_test
            
            train_with_target, test_with_target = self._simple_target_encode(
                train_with_target, test_with_target, categorical_cols, target_col
            )
            
            X_train = train_with_target.drop(columns=[target_col])
            X_test = test_with_target.drop(columns=[target_col])
        else:
            exp_logger.info("\n[Step 3/6] No categorical features found, skipping encoding")
        
        # Step 4: Handle missing values
        exp_logger.info("\n[Step 4/6] Handling missing values...")
        X_train = X_train.fillna(-999)
        X_test = X_test.fillna(-999)
        exp_logger.info("Filled missing values with -999")
        
        # Step 5: Remove low variance features
        exp_logger.info("\n[Step 5/6] Removing low-variance features...")
        X_train_array = self._remove_low_variance_features(X_train.values)
        X_test_array = self.variance_selector.transform(X_test.values)
        exp_logger.info(f"Shape after variance filter: {X_train_array.shape}")
        
        # Step 6: Feature selection
        exp_logger.info("\n[Step 6/6] Selecting best features...")
        if X_train_array.shape[1] > self.target_n_features:
            X_train_array = self._select_best_features(X_train_array, y_train)
            X_test_array = self.feature_selector.transform(X_test_array)
        else:
            exp_logger.info(f"Feature count ({X_train_array.shape[1]}) already below target ({self.target_n_features}), skipping selection")
        
        final_shape = X_train_array.shape
        reduction_ratio = (1 - final_shape[1] / initial_shape[1]) * 100
        new_ratio = final_shape[0] / final_shape[1]
        
        exp_logger.info(f"\nPreprocessing Summary:")
        exp_logger.info(f"  Initial shape: {initial_shape}")
        exp_logger.info(f"  Final shape: {final_shape}")
        exp_logger.info(f"  Dimension reduction: {initial_shape[1]} → {final_shape[1]} features")
        exp_logger.info(f"  Reduction ratio: {reduction_ratio:.1f}%")
        exp_logger.info(f"  New samples-to-features ratio: {new_ratio:.2f}")
        exp_logger.info(f"  Removed due to missing: {len(self.removed_high_missing)}")
        exp_logger.info(f"  Removed due to low variance: {self.removed_low_variance}")
        
        return X_train_array, X_test_array, y_train, y_test
    
    def get_feature_importance_report(self) -> str:
        """Get feature importance report"""
        if self.feature_importance is None:
            return "Feature selection not yet performed"
        
        report = "\nTOP 20 MOST IMPORTANT FEATURES:\n"
        report += "="*70 + "\n"
        
        for idx, row in self.feature_importance.head(20).iterrows():
            report += f"  {idx+1:2d}. Feature {int(row['feature_idx']):4d}: {row['importance']:.6f}\n"
        
        return report


# Convenience function
def preprocess_high_dimensional_dataset(train_df: pd.DataFrame, 
                                       test_df: pd.DataFrame,
                                       target_col: str = 'is_fraud',
                                       target_n_features: int = 150) -> Tuple:
    """
    Convenience function to preprocess high-dimensional datasets
    
    Parameters:
    -----------
    train_df, test_df : DataFrame
        Training and test dataframes
    target_col : str
        Name of target column
    target_n_features : int
        Target number of features (default: 150)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = HighDimensionalPreprocessor(target_n_features=target_n_features)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        train_df, test_df, target_col=target_col
    )
    
    return X_train, X_test, y_train, y_test, preprocessor
