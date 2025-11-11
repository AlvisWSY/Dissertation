#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection Model comparison Experiment
# 
# This experiment aims to compare various machine learning and deep learning methods on different fraud detection datasets.
# 
# ## Experiment Design
# 
# ### Dataset Overview (FULL DATA - No Sampling)
# - **IEEE**: 472K train/118K test, 81 features, High-dimensional PCA features
# - **col14_behave**: 238K train/59K test, 15 features, Contains categorical features
# - **col16_raw**: 1.47M train/24K test, 14 features, E-commerce transaction data
# - **counterfeit_products**: 4K train/1K test, 16 features, Product authenticity detection
# - **counterfeit_transactions**: 2.4K train/600 test, 19 features, Transaction authenticity detection
# - **creditCardPCA**: 228K train/57K test, 34 features, PCA-processed credit card data
# - **creditCardTransaction**: 1.30M train/556K test, 13 features, Credit card transaction data
# 
# ### Model Methods
# **Supervised Learning:**
# 1. **MLP (Multi-Layer Perceptron)** - Deep learning baseline
# 2. **XGBoost** - Ensemble learning, handles non-linear relationships
# 3. **Random Forest** - Ensemble learning, high interpretability
# 4. **Logistic Regression** - Linear baseline
# 5. **LightGBM** - Efficient gradient boosting
# 
# **Dimensionality Reduction + Classification:**
# 6. **PCA + SVM** - Linear dimensionality reduction + Support Vector Machine
# 7. **PCA + Logistic Regression** - Linear dimensionality reduction + Linear classification
# 
# **Distance/Similarity Methods:**
# 8. **KNN** - Distance-based classification
# 
# **Unsupervised/Semi-supervised Methods:**
# 9. **Isolation Forest** - Anomaly detection
# 10. **One-Class SVM** - One-class classification
# 11. **Autoencoder** - Deep learning anomaly detection
# 12. **DBSCAN** - Density-based clustering
# 
# ### Evaluation Metrics
# - **Accuracy**: Overall accuracy
# - **Precision**: Precision rate (for fraud class)
# - **Recall**: Recall rate (for fraud class)
# - **F1-Score**: Harmonic mean of precision and recall
# - **ROC-AUC**: Area under ROC curve
# - **PR-AUC**: Area under Precision-Recall curve
# - **Training Time**: Model training time
# - **Inference Time**: Model prediction time
# 

# ## 1. Import Libraries and Configuration
# 

# In[1]:


# # Core libraries
import pandas as pd
import numpy as np
import json
import warnings
import time
import gc
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# # Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold

# imbalance handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

# supervised learning model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

# 无supervised learning model
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

# # Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# # Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# GPUConfiguration
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use first two GPUs
torch.backends.cudnn.benchmark = True

# # Settings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# # Path configuration
BASE_DIR = Path('/usr1/home/s124mdg53_07/wang/FYP')
DATA_DIR = BASE_DIR / 'data'
JSON_DIR = BASE_DIR / 'json'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# GPU Information
if torch.cuda.is_available():
    print(f"✅ Detected {torch.cuda.device_count()} GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  未DetectedGPU，Will useCPU")

print("✅ All libraries imported successfully！")


# ### 1.1 Helper Functions: Memory Management and GPU Configuration

# In[2]:


def clear_memory():
    """Clear memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def get_memory_usage():
    """Get memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"📊 Memory Usage: {mem_info.rss / 1024**3:.2f} GB")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def smart_sample(X, y, max_samples=100000, strategy='stratified', min_fraud_samples=100):
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
    """
    if len(X) <= max_samples:
        return X, y
    
    print(f"   📉 Large dataset ({len(X):,} samples), sampling to {max_samples:,} samples")
    
    # Check fraud rate
    fraud_count = (y == 1).sum()
    fraud_rate = fraud_count / len(y)
    print(f"   Original fraud rate: {fraud_rate*100:.4f}% ({fraud_count:,} fraud samples)")
    
    if strategy == 'stratified':
        # For extremely imbalanced datasets, ensure minimum fraud samples
        if fraud_count > 0 and fraud_count < min_fraud_samples:
            print(f"   ⚠️  Very low fraud samples, ensuring at least {min_fraud_samples} fraud samples in sample")
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
            from sklearn.model_selection import train_test_split
            try:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, train_size=max_samples, stratify=y, random_state=42
                )
            except ValueError:
                # Fallback to random sampling if stratification fails
                print(f"   ⚠️  Stratified sampling failed, using random sampling")
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
    print(f"   Sampled fraud rate: {sampled_fraud_rate*100:.4f}% ({sampled_fraud:,} fraud samples)")
    
    return X_sample, y_sample

print("✅ Utility functions defined!")
get_memory_usage()


# ### 1.2 Imbalance Handling Methods

# In[3]:


class ImbalanceHandler:
    """imbalance handling器"""
    
    @staticmethod
    def get_imbalance_ratio(y):
        """Calculate imbalance ratio"""
        counts = y.value_counts()
        if len(counts) < 2:
            return 1.0
        return counts.max() / counts.min()
    
    @staticmethod
    def apply_smote(X, y, sampling_strategy='auto', k_neighbors=5):
        """SMOTEOversampling"""
        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            print(f"   ⚠️  SMOTEfailed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_adasyn(X, y, sampling_strategy='auto'):
        """ADASYNAdaptive Oversampling"""
        try:
            adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            print(f"   ⚠️  ADASYNfailed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_smote_tomek(X, y):
        """SMOTE + Tomek Links Combined sampling"""
        try:
            smt = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smt.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            print(f"   ⚠️  SMOTE-Tomekfailed: {e}, using original data")
            return X, y
    
    @staticmethod
    def apply_undersampling(X, y, sampling_strategy='auto'):
        """Random undersampling"""
        try:
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            print(f"   ⚠️  Undersamplingfailed: {e}, using original data")
            return X, y
    
    @staticmethod
    def get_class_weights(y):
        """Calculate class weights"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))


# Define imbalance strategies
IMBALANCE_STRATEGIES = {
    'none': {'name': 'None', 'handler': None},
    'smote': {'name': 'SMOTE', 'handler': ImbalanceHandler.apply_smote},
    'adasyn': {'name': 'ADASYN', 'handler': ImbalanceHandler.apply_adasyn},
    'smote_tomek': {'name': 'SMOTE+Tomek', 'handler': ImbalanceHandler.apply_smote_tomek},
    'undersample': {'name': 'Undersampling', 'handler': ImbalanceHandler.apply_undersampling},
}

print("✅ Imbalance handler definition completed!")
print(f"   Available strategies: {list(IMBALANCE_STRATEGIES.keys())}")


# ## 2. Data Loading and Preprocessing Module

# In[4]:


class DatasetLoader:
    """Dataset loader and preprocessor (with sparse data and large dataset optimization)"""
    
    def __init__(self, dataset_name, data_dir=DATA_DIR, handle_sparse=True, max_samples=None):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.handle_sparse = handle_sparse  # Whether to handle sparse features
        self.max_samples = max_samples  # Maximum samples (for large dataset sampling)
        self.is_sparse = False  # Whether it's a sparse dataset
        
    def load_data(self):
        """Load training and test sets"""
        train_path = self.data_dir / self.dataset_name / 'train'
        test_path = self.data_dir / self.dataset_name / 'test'
        
        # Find CSV files
        train_files = list(train_path.glob('*.csv'))
        test_files = list(test_path.glob('*.csv'))
        
        if not train_files or not test_files:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_name}")
        
        train_df = pd.read_csv(train_files[0])
        test_df = pd.read_csv(test_files[0])
        
        print(f"📊 {self.dataset_name} - Train: {train_df.shape}, Test: {test_df.shape}")
        
        return train_df, test_df
    
    def identify_label_column(self, df):
        """Identify label column"""
        label_candidates = ['is_fraud', 'isFraud', 'fraud', 'label', 'target', 'Class']
        for col in label_candidates:
            if col in df.columns:
                return col
        raise ValueError(f"Cannot identify label column for dataset {self.dataset_name}")
    
    def identify_feature_types(self, df, label_col):
        """Identify feature types: numerical, categorical, ID"""
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
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'id': id_features,
            'timestamp': timestamp_features
        }
    
    def handle_sparse_features(self, X_train, X_test):
        """
        Handle sparse features (for highly sparse datasets like IEEE)
        NOTE: Fit on training set only to avoid data leakage
        """
        if not self.handle_sparse:
            return X_train, X_test
        
        # Check sparsity on training set only
        sparsity = (X_train == 0).sum().sum() / (X_train.shape[0] * X_train.shape[1])
        print(f"   Sparsity: {sparsity*100:.2f}%")
        
        if sparsity > 0.5:  # If sparsity > 50%
            self.is_sparse = True
            print(f"   ⚠️  High sparsity detected, applying sparse feature processing")
            
            # Remove all-zero columns (fit on train, apply to both)
            zero_cols = X_train.columns[(X_train == 0).all()]
            if len(zero_cols) > 0:
                print(f"   Removing {len(zero_cols)} all-zero columns")
                X_train = X_train.drop(columns=zero_cols)
                X_test = X_test.drop(columns=zero_cols)
            
            # Remove low-variance columns (fit on train only)
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train)  # Fit on training set only
            
            selected_cols = X_train.columns[selector.get_support()]
            removed_cols = len(X_train.columns) - len(selected_cols)
            if removed_cols > 0:
                print(f"   Removing {removed_cols} low-variance columns")
                X_train = X_train[selected_cols]
                X_test = X_test[selected_cols]
        
        return X_train, X_test
    
    def preprocess(self, train_df, test_df, encode_categorical=True, apply_sampling=True):
        """Preprocess data with no data leakage"""
        # Identify label column
        label_col = self.identify_label_column(train_df)
        
        # Identify feature types (from training set only)
        feature_types = self.identify_feature_types(train_df, label_col)
        
        print(f"\nFeature Analysis:")
        print(f"  - Numerical: {len(feature_types['numerical'])} features")
        print(f"  - Categorical: {len(feature_types['categorical'])} features")
        print(f"  - ID: {len(feature_types['id'])} features (will be removed)")
        print(f"  - Timestamp: {len(feature_types['timestamp'])} features (will be removed)")
        
        # Separate features and labels
        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col]
        
        # Class imbalance analysis
        imbalance_ratio = ImbalanceHandler.get_imbalance_ratio(y_train)
        fraud_rate = (y_train == 1).sum() / len(y_train)
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"Fraud Rate: {fraud_rate*100:.4f}% ({(y_train == 1).sum():,} fraud samples)")
        
        # Remove ID and timestamp features
        drop_cols = feature_types['id'] + feature_types['timestamp']
        X_train = X_train.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.drop(columns=drop_cols, errors='ignore')
        
        # Encode categorical features (fit on train, apply to both)
        if encode_categorical and feature_types['categorical']:
            print(f"\nEncoding categorical features: {feature_types['categorical']}")
            for col in feature_types['categorical']:
                if col in X_train.columns:
                    le = LabelEncoder()
                    # Fit on training set
                    le.fit(X_train[col].astype(str))
                    X_train[col] = le.transform(X_train[col].astype(str))
                    # Handle unseen categories in test set
                    X_test[col] = X_test[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    self.label_encoders[col] = le
        
        # Handle sparse features (NO DATA LEAKAGE)
        X_train, X_test = self.handle_sparse_features(X_train, X_test)
        
        # Smart sampling (on training set only, after feature selection)
        if apply_sampling and self.max_samples and len(X_train) > self.max_samples:
            X_train, y_train = smart_sample(X_train, y_train, self.max_samples, min_fraud_samples=100)
        
        # Ensure y_train and y_test are 1D Series (not DataFrame)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]
        
        # Reset index to avoid alignment issues
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Standardize numerical features (fit on train, apply to both)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print(f"\n✅ Preprocessing complete! Final features: {X_train_scaled.shape[1]}")
        print(f"   Training set: {X_train_scaled.shape[0]:,} samples")
        print(f"   Test set: {X_test_scaled.shape[0]:,} samples")
        train_fraud_rate = (y_train == 1).sum() / len(y_train)
        test_fraud_rate = (y_test == 1).sum() / len(y_test)
        print(f"   Label distribution - Train: Normal={1-train_fraud_rate:.4f}, Fraud={train_fraud_rate:.4f}")
        print(f"   Label distribution - Test: Normal={1-test_fraud_rate:.4f}, Fraud={test_fraud_rate:.4f}")
        
        # Clear memory
        clear_memory()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_types

# Test dataset loader with FULL DATA (no sampling)
print("🧪 Testing dataset loader with FULL DATA...")
loader = DatasetLoader('creditCardPCA', max_samples=None)  # No sampling limit
train_df, test_df = loader.load_data()
X_train, X_test, y_train, y_test, feature_types = loader.preprocess(train_df, test_df)
print(f"\nSample data shape: X_train={X_train.shape}, y_train={y_train.shape}")
get_memory_usage()


# ## 3. Model Definitions

# In[5]:


# ==================== # Deep Learning Models (Multi-GPU Support) ====================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron Classifier (Optimized)"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
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
    """Autoencoder for Anomaly Detection (Optimized)"""
    def __init__(self, input_dim, encoding_dims=[128, 64, 32]):
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
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


# ==================== Training Functions (Optimized) ====================

def train_mlp(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=512, 
              lr=0.001, patience=10, use_multi_gpu=True):
    """Train MLP model (Multi-GPU and class weights support)"""
    # GPUConfiguration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # 多GPUData parallelism
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"   Using {torch.cuda.device_count()} GPUsfor training")
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    # Data preparation
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.values),
        torch.FloatTensor(y_train.values).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    
    X_val_tensor = torch.FloatTensor(X_val.values).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(device)
    
    # Calculate class weights
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # Using weighted BCE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # Remove final sigmoid, use logits
            if isinstance(model, nn.DataParallel):
                outputs = model.module.network[:-1](batch_X)  # Remove sigmoid
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
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping于epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    # Clear GPU cache
    clear_memory()
    
    return model


def train_autoencoder(model, X_train, epochs=50, batch_size=512, lr=0.001, use_multi_gpu=True):
    """Train Autoencoder (Multi-GPU support)"""
    # GPUConfiguration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # 多GPUData parallelism
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"   Using {torch.cuda.device_count()} GPUsfor training")
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        device = torch.device('cpu')
        model = model.to(device)
    
    # 只UsingNormalsamplesTraining
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    
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
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}")
    
    # Clear GPU cache
    clear_memory()
    
    return model


print("✅ Model Definitionscompleted！")
if torch.cuda.is_available():
    print(f"   Will use {torch.cuda.device_count()} GPUsfor accelerated training")


# ## 4. Unified Model Training and Evaluation Framework

# In[6]:


class PerformanceEvaluator:
    """Performance Evaluator (supports imbalance handling annotation)"""
    
    def evaluate_supervised(self, y_true, y_pred, y_pred_proba, model_name, dataset_name, 
                          train_time, inference_time, imbalance_strategy='none'):
        """Evaluate supervised learning model"""
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
        return result
    
    def evaluate_unsupervised(self, y_true, anomaly_scores, model_name, dataset_name,
                             train_time, inference_time, contamination, imbalance_strategy='none'):
        """Evaluate unsupervised learning model (anomaly detection)"""
        # Using最优阈Value
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
        return result
    
    def print_result(self, result):
        """Print evaluation results"""
        print(f"\n✅ {result['model']} completed!")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        print(f"   F1-Score: {result['f1_score']:.4f}")
        print(f"   ROC-AUC: {result['roc_auc']:.4f}")
        print(f"   TrainingTime: {result['train_time']:.2f}s")
        print(f"   Inference time: {result['inference_time']:.2f}s")


class ImbalanceHandler:
    """imbalance handling器"""
    
    @staticmethod
    def get_imbalance_ratio(y):
        """Calculate class imbalance ratio"""
        class_counts = pd.Series(y).value_counts()
        if len(class_counts) < 2:
            return 1.0
        return class_counts.max() / class_counts.min()
    
    @staticmethod
    def apply_smote(X, y):
        """Apply SMOTE oversampling"""
        smote = SMOTE(random_state=42)
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  SMOTEfailed: {e}，using original data")
            return X, y
    
    @staticmethod
    def apply_adasyn(X, y):
        """Apply ADASYN oversampling"""
        adasyn = ADASYN(random_state=42)
        try:
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  ADASYNfailed: {e}，using original data")
            return X, y
    
    @staticmethod
    def apply_smote_tomek(X, y):
        """Apply SMOTE+Tomek Links combined sampling"""
        smote_tomek = SMOTETomek(random_state=42)
        try:
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  SMOTE+Tomekfailed: {e}，using original data")
            return X, y
    
    @staticmethod
    def apply_undersampling(X, y):
        """Apply random undersampling"""
        rus = RandomUnderSampler(random_state=42)
        try:
            X_resampled, y_resampled = rus.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  Undersamplingfailed: {e}，using original data")
            return X, y


# Define imbalance handling strategies
IMBALANCE_STRATEGIES = {
    'none': {'name': 'None', 'func': None},
    'smote': {'name': 'SMOTE', 'func': ImbalanceHandler.apply_smote},
    'adasyn': {'name': 'ADASYN', 'func': ImbalanceHandler.apply_adasyn},
    'smote_tomek': {'name': 'SMOTE+Tomek', 'func': ImbalanceHandler.apply_smote_tomek},
    'undersampling': {'name': 'Undersampling', 'func': ImbalanceHandler.apply_undersampling},
}


print("✅ 评估andimbalanceprocessing模块构建completed！")


# %% [markdown]
# ## 🚀 Model Training Module (12 Models)

# %%
class ExperimentRunner:
    """Experiment Runner (supports imbalance handling comparison)"""
    
    def __init__(self, compare_imbalance=True, use_sampling_for_slow_models=True):
        self.evaluator = PerformanceEvaluator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compare_imbalance = compare_imbalance  # Whether to compare imbalance handling methods
        self.use_sampling_for_slow_models = use_sampling_for_slow_models  # Whether to sample for slow models
    
    def _apply_imbalance_strategy(self, X, y, strategy='none'):
        """Apply imbalance handling strategy"""
        if strategy == 'none' or IMBALANCE_STRATEGIES[strategy]['func'] is None:
            return X, y
        
        strategy_func = IMBALANCE_STRATEGIES[strategy]['func']
        return strategy_func(X, y)
    
    def run_logistic_regression(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """Logistic Regression"""
        print(f"\n🚀 Training Logistic Regression [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        start_time = time.time()
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1)
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'Logistic Regression', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_random_forest(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """Random Forest"""
        print(f"\n🚀 Training Random Forest [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, 
            class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'Random Forest', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_xgboost(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """XGBoost"""
        print(f"\n🚀 Training XGBoost [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        scale_pos_weight = (y_train_proc == 0).sum() / (y_train_proc == 1).sum()
        
        start_time = time.time()
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric='logloss', tree_method='gpu_hist',  # GPUacceleration
            gpu_id=0
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'XGBoost', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_lightgbm(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """LightGBM"""
        print(f"\n🚀 Training LightGBM [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        scale_pos_weight = (y_train_proc == 0).sum() / (y_train_proc == 1).sum()
        
        start_time = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            verbose=-1, device='cpu', n_jobs=-1  # ✅ Fixed: Use CPU instead of GPU
        )
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'LightGBM', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_knn(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none', max_samples=20000):
        """K-Nearest Neighbors（大Dataset采样）"""
        print(f"\n🚀 Training KNN [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        # for大Dataset采样
        if self.use_sampling_for_slow_models and len(X_train) > max_samples:
            X_train_sampled, y_train_sampled = smart_sample(X_train, y_train, max_samples)
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train_sampled, y_train_sampled, imbalance_strategy)
        
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'KNN', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_pca_svm(self, X_train, y_train, X_test, y_test, dataset_name, 
                    imbalance_strategy='none', n_components=0.95, max_samples=20000):
        """PCA + SVM（大Dataset采样）"""
        print(f"\n🚀 Training PCA + SVM [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        # for大Dataset采样
        if self.use_sampling_for_slow_models and len(X_train) > max_samples:
            X_train_sampled, y_train_sampled = smart_sample(X_train, y_train, max_samples)
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        start_time = time.time()
        
        # PCAdimension reduction
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_sampled)
        X_test_pca = pca.transform(X_test)
        print(f"   PCAdimension reduction: {X_train_sampled.shape[1]} -> {X_train_pca.shape[1]} dimensions")
        
        # 应withimbalanceprocessing
        X_train_pca_df = pd.DataFrame(X_train_pca)
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train_pca_df, y_train_sampled, imbalance_strategy)
        
        # SVM分类
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        model.fit(X_train_proc, y_train_proc)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test_pca)
        y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'PCA+SVM', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del pca, model, X_train_proc, y_train_proc, X_train_pca, X_test_pca
        clear_memory()
        return result
    
    def run_mlp(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """MLP (多层感知机)"""
        print(f"\n🚀 Training MLP [{IMBALANCE_STRATEGIES[imbalance_strategy]['name']}]...")
        
        X_train_proc, y_train_proc = self._apply_imbalance_strategy(X_train, y_train, imbalance_strategy)
        
        # 划分Validation集
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_proc, y_train_proc, test_size=0.2, random_state=42, stratify=y_train_proc
        )
        
        start_time = time.time()
        model = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64])
        model = train_mlp(model, X_train_split, y_train_split, X_val, y_val, epochs=50, use_multi_gpu=True)
        train_time = time.time() - start_time
        
        # 预测
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
            if isinstance(model, nn.DataParallel):
                y_pred_proba = model.module(X_test_tensor).cpu().numpy().flatten()
            else:
                y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_supervised(
            y_test, y_pred, y_pred_proba, 'MLP', dataset_name,
            train_time, inference_time, imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model, X_train_proc, y_train_proc
        clear_memory()
        return result
    
    def run_isolation_forest(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """Isolation Forest(Unsupervised, no resampling needed)"""
        print(f"\n🚀 Training Isolation Forest...")
        contamination = y_train.mean()
        
        start_time = time.time()
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        anomaly_scores = -model.score_samples(X_test)
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_unsupervised(
            y_test, anomaly_scores, 'Isolation Forest', dataset_name,
            train_time, inference_time, contamination=contamination, imbalance_strategy=imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model
        clear_memory()
        return result
    
    def run_autoencoder(self, X_train, y_train, X_test, y_test, dataset_name, imbalance_strategy='none'):
        """Autoencoder（无监督，只withNormalsamples）"""
        print(f"\n🚀 Training Autoencoder...")
        X_train_normal = X_train[y_train == 0]
        print(f"   Using {len(X_train_normal):,} NormalsamplesTraining")
        
        start_time = time.time()
        model = Autoencoder(input_dim=X_train.shape[1], encoding_dims=[128, 64, 32])
        model = train_autoencoder(model, X_train_normal, epochs=50, use_multi_gpu=True)
        train_time = time.time() - start_time
        
        # 计算重构误差
        model.eval()
        start_time = time.time()
        X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
        if isinstance(model, nn.DataParallel):
            anomaly_scores = model.module.get_reconstruction_error(X_test_tensor)
        else:
            anomaly_scores = model.get_reconstruction_error(X_test_tensor)
        inference_time = time.time() - start_time
        
        result = self.evaluator.evaluate_unsupervised(
            y_test, anomaly_scores, 'Autoencoder', dataset_name,
            train_time, inference_time, contamination=y_train.mean(), imbalance_strategy=imbalance_strategy
        )
        self.evaluator.print_result(result)
        
        del model
        clear_memory()
        return result
    
    def run_all_models(self, X_train, y_train, X_test, y_test, dataset_name, skip_slow=False):
        """running all models（支持class imbalancecomparison）"""
        print(f"\n{'='*80}")
        print(f"🔬 StartinginDataset '{dataset_name}' running all models")
        print(f"{'='*80}")
        
        # 确定要UsingofImbalance strategy
        if self.compare_imbalance:
            strategies = ['none', 'smote']  # comparison：None vs SMOTE
            print(f"📊 将comparisonimbalance handling方法: {[IMBALANCE_STRATEGIES[s]['name'] for s in strategies]}")
        else:
            strategies = ['none']
        
        models = {}
        
        for strategy in strategies:
            print(f"\n{'─'*80}")
            print(f"📌 Imbalance strategy: {IMBALANCE_STRATEGIES[strategy]['name']}")
            print(f"{'─'*80}")
            
            # 快速Model（支持大Dataset）
            models[f'lr_{strategy}'] = self.run_logistic_regression(X_train, y_train, X_test, y_test, dataset_name, strategy)
            models[f'rf_{strategy}'] = self.run_random_forest(X_train, y_train, X_test, y_test, dataset_name, strategy)
            models[f'xgb_{strategy}'] = self.run_xgboost(X_train, y_train, X_test, y_test, dataset_name, strategy)
            models[f'lgb_{strategy}'] = self.run_lightgbm(X_train, y_train, X_test, y_test, dataset_name, strategy)
            models[f'mlp_{strategy}'] = self.run_mlp(X_train, y_train, X_test, y_test, dataset_name, strategy)
            
            # 慢速Model（只in小Datasetor采样后运行）
            if not skip_slow or len(X_train) < 30000:
                models[f'knn_{strategy}'] = self.run_knn(X_train, y_train, X_test, y_test, dataset_name, strategy)
                models[f'pca_svm_{strategy}'] = self.run_pca_svm(X_train, y_train, X_test, y_test, dataset_name, strategy)
            else:
                print(f"\n⚠️  Large dataset, skipping KNN and PCA+SVM or using sampling")
                if self.use_sampling_for_slow_models:
                    models[f'knn_{strategy}'] = self.run_knn(X_train, y_train, X_test, y_test, dataset_name, strategy)
                    models[f'pca_svm_{strategy}'] = self.run_pca_svm(X_train, y_train, X_test, y_test, dataset_name, strategy)
        
        # # Unsupervised models (run once, not affected by imbalance handling)
        models['isolation_forest'] = self.run_isolation_forest(X_train, y_train, X_test, y_test, dataset_name, 'none')
        models['autoencoder'] = self.run_autoencoder(X_train, y_train, X_test, y_test, dataset_name, 'none')
        
        return models


print("✅ ExperimentRunner 构建completed（支持imbalancecomparison）！")


# ## 5. 结果可视化and分析

# In[7]:


class ResultsAnalyzer:
    """Results Analysis and Visualization (supports imbalance comparison)"""
    
    def __init__(self, results_df):
        self.results_df = results_df
    
    def plot_imbalance_comparison(self, metric='f1_score', figsize=(16, 10)):
        """comparison不同imbalance handling方法of效果"""
        if 'imbalance_strategy' not in self.results_df.columns:
            print("⚠️  No imbalance comparison experiment conducted")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        strategies = self.results_df['imbalance_strategy'].unique()
        datasets = self.results_df['dataset'].unique()
        
        # 1. 每Modelin不同strategy下of表现
        ax = axes[0]
        models = self.results_df['model'].unique()
        x = np.arange(len(models))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            strategy_data = self.results_df[self.results_df['imbalance_strategy'] == strategy]
            means = [strategy_data[strategy_data['model'] == m][metric].mean() for m in models]
            ax.bar(x + i * width, means, width, label=strategy, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(f'Average {metric.upper()}', fontsize=12)
        ax.set_title('不同Imbalance strategyfor各Modelof影响', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='strategy')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. 每Datasetin不同strategy下of表现
        ax = axes[1]
        x = np.arange(len(datasets))
        
        for i, strategy in enumerate(strategies):
            strategy_data = self.results_df[self.results_df['imbalance_strategy'] == strategy]
            means = [strategy_data[strategy_data['dataset'] == d][metric].mean() for d in datasets]
            ax.bar(x + i * width, means, width, label=strategy, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(f'Average {metric.upper()}', fontsize=12)
        ax.set_title('不同Imbalance strategyin各Dataset上of效果', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(title='strategy')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Strategy improvement effect (relative to none)
        ax = axes[2]
        if 'none' in strategies and len(strategies) > 1:
            improvements = []
            labels = []
            
            for strategy in strategies:
                if strategy != 'none':
                    none_scores = self.results_df[self.results_df['imbalance_strategy'] == 'none'][metric].mean()
                    strategy_scores = self.results_df[self.results_df['imbalance_strategy'] == strategy][metric].mean()
                    improvement = ((strategy_scores - none_scores) / none_scores) * 100
                    improvements.append(improvement)
                    labels.append(strategy)
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax.barh(labels, improvements, color=colors, alpha=0.7)
            ax.set_xlabel('F1-Score Improvement (%)', fontsize=12)
            ax.set_title('相for于Noneof性能Improvement', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Need to compare with none strategy', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # 4. Best Strategy Statistics
        ax = axes[3]
        best_strategies = []
        for dataset in datasets:
            dataset_data = self.results_df[self.results_df['dataset'] == dataset]
            for model in models:
                model_data = dataset_data[dataset_data['model'] == model]
                if len(model_data) > 0:
                    best_idx = model_data[metric].idxmax()
                    best_strategy = model_data.loc[best_idx, 'imbalance_strategy']
                    best_strategies.append(best_strategy)
        
        strategy_counts = pd.Series(best_strategies).value_counts()
        ax.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
              startangle=90, colors=plt.cm.Set3.colors)
        ax.set_title('Best Strategy Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metric_comparison(self, metric='f1_score', figsize=(14, 6)):
        """比较不同Modelin各Dataset上of指标"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 按Dataset分组
        datasets = self.results_df['dataset'].unique()
        
        # 左图: 各Modelin不同Dataset上of表现
        pivot_data = self.results_df.pivot_table(
            index='model', columns='dataset', values=metric, aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=axes[0])
        axes[0].set_title(f'{metric.upper()} - 按Model分组', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel(metric.upper(), fontsize=12)
        axes[0].legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 右图: 各Dataset上ModelofAverage表现
        avg_by_model = self.results_df.groupby('model')[metric].mean().sort_values(ascending=False)
        avg_by_model.plot(kind='barh', ax=axes[1], color='skyblue')
        axes[1].set_title(f'{metric.upper()} - ModelAverage表现', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'Average {metric.upper()}', fontsize=12)
        axes[1].set_ylabel('Model', fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_metrics_heatmap(self, figsize=(16, 10)):
        """绘制所有指标of热力图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            pivot_data = self.results_df.pivot_table(
                index='model', columns='dataset', values=metric, aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                       ax=axes[idx], cbar_kws={'label': metric.upper()})
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Dataset', fontsize=10)
            axes[idx].set_ylabel('Model', fontsize=10)
        
        # 隐藏多余of子图
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_comparison(self, figsize=(14, 5)):
        """比较TrainingTimeandInference time"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # TrainingTime
        avg_train_time = self.results_df.groupby('model')['train_time'].mean().sort_values(ascending=False)
        avg_train_time.plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('AverageTrainingTime', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel('Model', fontsize=12)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Inference time
        avg_inference_time = self.results_df.groupby('model')['inference_time'].mean().sort_values(ascending=False)
        avg_inference_time.plot(kind='barh', ax=axes[1], color='lightgreen')
        axes[1].set_title('AverageInference time', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_ylabel('Model', fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dataset_comparison(self, figsize=(14, 8)):
        """比较不同Datasetof表现"""
        datasets = self.results_df['dataset'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['f1_score', 'roc_auc', 'precision', 'recall']
        
        for idx, metric in enumerate(metrics):
            for dataset in datasets:
                dataset_data = self.results_df[self.results_df['dataset'] == dataset]
                dataset_data_avg = dataset_data.groupby('model')[metric].mean().sort_values(ascending=False)
                axes[idx].plot(range(len(dataset_data_avg)), dataset_data_avg.values, 
                             marker='o', label=dataset, linewidth=2)
            
            axes[idx].set_title(f'{metric.upper()} comparison', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Model排名', fontsize=10)
            axes[idx].set_ylabel(metric.upper(), fontsize=10)
            axes[idx].legend(title='Dataset', fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_table(self):
        """Generate summary table"""
        print("\n" + "="*120)
        print("📊 Experiment Results Summary")
        print("="*120 + "\n")
        
        # 按Dataset分组显示
        for dataset in self.results_df['dataset'].unique():
            dataset_results = self.results_df[self.results_df['dataset'] == dataset]
            dataset_results_sorted = dataset_results.sort_values('f1_score', ascending=False)
            
            print(f"\n🔹 Dataset: {dataset}")
            print("-" * 120)
            
            display_df = dataset_results_sorted[[
                'model', 'imbalance_strategy', 'accuracy', 'precision', 'recall', 'f1_score', 
                'roc_auc', 'train_time', 'inference_time'
            ]].copy()
            
            # 格式化数Value
            for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            for col in ['train_time', 'inference_time']:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}s' if pd.notna(x) else 'N/A')
            
            print(display_df.to_string(index=False))
            print()
        
        # 总体BestModel
        print("\n" + "="*120)
        print("🏆 BestModel总结")
        print("="*120 + "\n")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            best_result = self.results_df.loc[self.results_df[metric].idxmax()]
            print(f"Best {metric.upper():15s}: {best_result['model']:20s} "
                  f"(Dataset: {best_result['dataset']:20s}, "
                  f"strategy: {best_result['imbalance_strategy']:15s}, "
                  f"Value: {best_result[metric]:.4f})")
        
        # imbalanceprocessingcomparison
        if 'imbalance_strategy' in self.results_df.columns:
            print("\n" + "="*120)
            print("📊 imbalance handling效果comparison")
            print("="*120 + "\n")
            
            strategy_performance = self.results_df.groupby('imbalance_strategy')[['f1_score', 'precision', 'recall', 'roc_auc']].mean()
            print(strategy_performance.round(4))
        
        print("\n" + "="*120 + "\n")
    
    def export_results(self, output_path):
        """Export results to CSV"""
        self.results_df.to_csv(output_path, index=False)
        print(f"✅ Results saved to: {output_path}")


print("✅ 可视化模块构建completed（支持imbalancecomparison）！")


# ## 6. 运行完整实验
# 
# 现in我们将in所有Datasetrunning all models并分析结果。

# In[ ]:


# Dataset configuration: optimization strategies for different datasets
DATASET_CONFIGS = {
    'IEEE': {
        'max_samples': None,  # Use FULL dataset (no sampling)
        'handle_sparse': True,  # Handle sparse features
        'skip_slow': False,  # Run all models
        'size_category': 'large',  # 472K samples
    },
    'col14_behave': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'medium',  # 238K samples
    },
    'col16_raw': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'large',  # 1.47M samples
    },
    'creditCardPCA': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'medium',  # 228K samples
    },
    'creditCardTransaction': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'large',  # 1.30M samples
    },
    'counterfeit_products': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'small',  # 4K samples
    },
    'counterfeit_transactions': {
        'max_samples': None,
        'handle_sparse': False,
        'skip_slow': False,
        'size_category': 'small',  # 2.4K samples
    },
}

# Define datasets grouped by size
SMALL_DATASETS = ['counterfeit_products', 'counterfeit_transactions']
MEDIUM_DATASETS = ['creditCardPCA', 'col14_behave', 'IEEE']
LARGE_DATASETS = ['col16_raw', 'creditCardTransaction']

ALL_DATASETS = SMALL_DATASETS + MEDIUM_DATASETS + LARGE_DATASETS

# Create experiment runner
runner = ExperimentRunner(
    compare_imbalance=True,
    use_sampling_for_slow_models=False
)

print("🚀 Starting OPTIMIZED experiment with memory-efficient strategy...\n")
print("=" * 100)
print("Experiment Configuration:")
print(f"  - Compare imbalance handling: Yes")
print(f"  - GPU acceleration: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")
print(f"  - Total datasets: {len(ALL_DATASETS)}")
print(f"  - Small datasets (batch process): {len(SMALL_DATASETS)}")
print(f"  - Medium datasets (batch process): {len(MEDIUM_DATASETS)}")
print(f"  - Large datasets (one-by-one): {len(LARGE_DATASETS)}")
print("=" * 100)

all_results = {}


# Strategy 1: Process SMALL & MEDIUM datasets together (low memory overhead)
print(f"\n{'#'*100}")
print(f"# PHASE 1: Processing Small & Medium Datasets (Batch Mode)")
print(f"{'#'*100}\n")

for dataset_name in SMALL_DATASETS + MEDIUM_DATASETS:
    try:
        print(f"\n{'='*80}")
        print(f"📊 Loading dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        config = DATASET_CONFIGS[dataset_name]
        
        # Load and preprocess data
        loader = DatasetLoader(
            dataset_name,
            max_samples=config['max_samples'],
            handle_sparse=config['handle_sparse']
        )
        train_df, test_df = loader.load_data()
        X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
            train_df, test_df,
            apply_sampling=False
        )
        
        print(f"✅ Data loaded:")
        print(f"   Training: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"   Test: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        print(f"   Memory usage: ", end="")
        get_memory_usage()
        
        # Run all models on this dataset (CORRECT parameter order!)
        dataset_results = runner.run_all_models(
            X_train, y_train, X_test, y_test,  # ✅ Fixed: X_train, y_train, X_test, y_test
            dataset_name=dataset_name,
            skip_slow=config['skip_slow']
        )
        
        all_results[dataset_name] = dataset_results
        
        # IMMEDIATELY clear this dataset from memory
        del train_df, test_df, X_train, X_test, y_train, y_test, loader
        clear_memory()
        
        print(f"\n✅ {dataset_name} completed and cleared from memory\n")
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        continue


# Strategy 2: Process LARGE datasets ONE-BY-ONE with aggressive memory management
print(f"\n{'#'*100}")
print(f"# PHASE 2: Processing Large Datasets (One-by-One Mode)")
print(f"{'#'*100}\n")

for dataset_name in LARGE_DATASETS:
    try:
        print(f"\n{'='*80}")
        print(f"📊 Loading LARGE dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        config = DATASET_CONFIGS[dataset_name]
        
        # Pre-load check
        print(f"⚠️  Large dataset warning: {dataset_name}")
        print(f"   This dataset will be loaded and immediately cleared after EACH model")
        print(f"   to minimize memory usage.\n")
        
        # Load and preprocess data
        loader = DatasetLoader(
            dataset_name,
            max_samples=config['max_samples'],
            handle_sparse=config['handle_sparse']
        )
        train_df, test_df = loader.load_data()
        X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
            train_df, test_df,
            apply_sampling=False
        )
        
        print(f"✅ Data loaded:")
        print(f"   Training: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"   Test: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        print(f"   Memory usage: ", end="")
        get_memory_usage()
        
        # Run all models on this dataset (CORRECT parameter order!)
        dataset_results = runner.run_all_models(
            X_train, y_train, X_test, y_test,  # ✅ Fixed: X_train, y_train, X_test, y_test
            dataset_name=dataset_name,
            skip_slow=config['skip_slow']
        )
        
        all_results[dataset_name] = dataset_results
        
        # AGGRESSIVE memory cleanup for large datasets
        del train_df, test_df, X_train, X_test, y_train, y_test, loader
        clear_memory()
        
        # Extra GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"\n✅ {dataset_name} completed with aggressive memory cleanup\n")
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue


print(f"\n{'='*100}")
print(f"🎉 ALL EXPERIMENTS COMPLETED!")
print(f"{'='*100}\n")

# Save results
results_df = pd.DataFrame(all_results).T
results_df.to_csv(RESULTS_DIR / 'experiment_results.csv')
print(f"✅ Results saved to: {RESULTS_DIR / 'experiment_results.csv'}")


# ## 7. 结果分析and可视化

# In[ ]:


# 获取所有结果
results_df = runner.evaluator.get_results_df()

# 创建分析器
analyzer = ResultsAnalyzer(results_df)

# Generate summary table
analyzer.generate_summary_table()

# 保存结果
analyzer.export_results(RESULTS_DIR / 'experiment_results.csv')
print(f"\n完整结果已保存!")


# ### 7.0 imbalance handling效果comparison
# 
# 首先分析不同imbalance handling方法of效果。

# In[ ]:


analyzer.plot_imbalance_comparison('f1_score', figsize=(18, 12))


# ### 7.1 F1-Score comparison

# In[ ]:


analyzer.plot_metric_comparison('f1_score', figsize=(16, 6))


# ### 7.2 ROC-AUC comparison

# In[ ]:


analyzer.plot_metric_comparison('roc_auc', figsize=(16, 6))


# ### 7.3 所有指标热力图

# In[ ]:


analyzer.plot_all_metrics_heatmap(figsize=(18, 12))


# ### 7.4 Time性能comparison

# In[ ]:


analyzer.plot_time_comparison(figsize=(16, 5))


# ### 7.5 Datasetdimensions度comparison

# In[ ]:


analyzer.plot_dataset_comparison(figsize=(16, 10))


# ## 8. 深入分析：单Dataset示例
# 
# 以下演示如何in单Dataset上进行详细分析（可选择任意Dataset）。

# In[ ]:


# 选择一Dataset进行详细分析
selected_dataset = 'creditCardPCA'

# 重新加载数据
loader = DatasetLoader(selected_dataset)
train_df, test_df = loader.load_data()
X_train, X_test, y_train, y_test, feature_types = loader.preprocess(train_df, test_df)

print(f"\n📊 Dataset: {selected_dataset}")
print(f"Training集大小: {X_train.shape}")
print(f"Test set大小: {X_test.shape}")
print(f"features数量: {X_train.shape[1]}")
print(f"类别分布 (Training集): {dict(y_train.value_counts())}")
print(f"类别分布 (Test set): {dict(y_test.value_counts())}")


# In[ ]:


# 优化：自动读取最新samples量，支持分层采样and多次重复实验
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

selected_dataset = 'creditCardPCA'  # 可更改as任意Dataset
n_repeats = 5  # 重复实验次数，保证统计学严谨
split_ratio = 0.2  # Test setratio
random_seed = 42

# 重新加载数据
loader = DatasetLoader(selected_dataset)
train_df, test_df = loader.load_data()

# 合并全量数据（如有需要）
full_df = pd.concat([train_df, test_df], ignore_index=True)
X = full_df.drop(columns=['is_fraud'])
y = full_df['is_fraud']

print(f"\n📊 Dataset: {selected_dataset}")
print(f"全量samples: {len(full_df)}，类别分布: {dict(y.value_counts())}")

# 多次分层采样划分Training/Test set
for repeat in range(n_repeats):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=random_seed + repeat)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"\n第{repeat+1}次划分：")
    print(f"Training集: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training集类别分布: {dict(y_train.value_counts())}")
    print(f"Test set类别分布: {dict(y_test.value_counts())}")
    # 这里可插入ModelTrainingand评估代码，收集每次实验结果

# 后续可for n_repeats 次实验结果做均Value/方差统计and显著性检验


# ### 8.1 数据分布可视化

# In[ ]:


# 绘制Label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training集Label distribution
y_train.value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'coral'])
axes[0].set_title('Training集Label distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('标签 (0=Normal, 1=Fraud)', fontsize=12)
axes[0].set_ylabel('samples数量', fontsize=12)
axes[0].tick_params(axis='x', rotation=0)

# Test setLabel distribution
y_test.value_counts().plot(kind='bar', ax=axes[1], color=['skyblue', 'coral'])
axes[1].set_title('Test setLabel distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('标签 (0=Normal, 1=Fraud)', fontsize=12)
axes[1].set_ylabel('samples数量', fontsize=12)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# features相关性热力图（只显示前20features）
if X_train.shape[1] <= 20:
    plt.figure(figsize=(12, 10))
    corr = X_train.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title(f'{selected_dataset} - features相关性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print(f"features数量较多 ({X_train.shape[1]}), 跳过相关性热力图")


# ## 9. 总结and建议
# 
# 根据实验结果，我们可以得出以下结论and建议。

# In[ ]:


# 生成详细of分析报告
print("="*100)
print("📝 实验总结and分析")
print("="*100)

# 1. BestModel分析
print("\n### 1️⃣ BestModel分析\n")
best_models = {}
for metric in ['f1_score', 'roc_auc', 'precision', 'recall']:
    best_idx = results_df[metric].idxmax()
    best = results_df.loc[best_idx]
    best_models[metric] = best
    print(f"🏆 Best {metric.upper()}: {best['model']} (Dataset: {best['dataset']}, Value: {best[metric]:.4f})")

# 2. Model类别分析
print("\n### 2️⃣ Model类别性能分析\n")

# 监督学习方法
supervised = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'MLP', 'KNN']
supervised_results = results_df[results_df['model'].isin(supervised)]
print(f"📊 监督学习方法Average F1-Score: {supervised_results['f1_score'].mean():.4f}")

# dimension reduction+分类方法
dim_reduction = ['PCA+SVM', 'PCA+LR']
dim_reduction_results = results_df[results_df['model'].isin(dim_reduction)]
if not dim_reduction_results.empty:
    print(f"📊 dimension reduction+分类方法Average F1-Score: {dim_reduction_results['f1_score'].mean():.4f}")

# 无监督方法
unsupervised = ['Isolation Forest', 'One-Class SVM', 'Autoencoder']
unsupervised_results = results_df[results_df['model'].isin(unsupervised)]
if not unsupervised_results.empty:
    print(f"📊 无监督方法Average F1-Score: {unsupervised_results['f1_score'].mean():.4f}")

# 3. Dataset特性分析
print("\n### 3️⃣ Dataset特性forModel性能of影响\n")
for dataset in results_df['dataset'].unique():
    dataset_results = results_df[results_df['dataset'] == dataset]
    best_model = dataset_results.loc[dataset_results['f1_score'].idxmax()]
    avg_f1 = dataset_results['f1_score'].mean()
    print(f"📁 {dataset:25s} - Best: {best_model['model']:20s} (F1: {best_model['f1_score']:.4f}), "
          f"Average: {avg_f1:.4f}")

# 4. 效率分析
print("\n### 4️⃣ Trainingand推理效率分析\n")
time_analysis = results_df.groupby('model')[['train_time', 'inference_time']].mean()
time_analysis = time_analysis.sort_values('train_time')
print("⏱️  TrainingTime排名（快→慢）:")
for idx, (model, row) in enumerate(time_analysis.iterrows(), 1):
    print(f"   {idx}. {model:20s} - Training: {row['train_time']:6.2f}s, 推理: {row['inference_time']:.4f}s")

# 5. 综合推荐
print("\n### 5️⃣ Model推荐建议\n")
print("基于实验结果，针for不同场景of推荐：")
print("\n🎯 **高Accuracy场景**（追求Best性能）:")
best_f1_model = results_df.groupby('model')['f1_score'].mean().idxmax()
print(f"   推荐: {best_f1_model}")

print("\n⚡ **实时推理场景**（速度优先）:")
fast_models = results_df.groupby('model')['inference_time'].mean().nsmallest(3)
print(f"   推荐: {', '.join(fast_models.index.tolist())}")

print("\n💰 **资源受限场景**（低计算成本）:")
efficient_models = results_df.groupby('model')['train_time'].mean().nsmallest(3)
print(f"   推荐: {', '.join(efficient_models.index.tolist())}")

print("\n🔍 **class imbalance场景**（Fraud检测特性）:")
best_recall_model = results_df.groupby('model')['recall'].mean().idxmax()
print(f"   推荐: {best_recall_model} (高Recall)")

print("\n" + "="*100)


# ## 10. Dataset特性概览

# In[ ]:


# 读取Dataset元信息
with open(JSON_DIR / 'dataset_clean_summary.json', 'r') as f:
    dataset_info = json.load(f)

# 提取关键信息
dataset_summary = []
for dataset_name, info in dataset_info.items():
    if 'train' in info and info['train']:
        train_info = info['train'][0]
        test_info = info['test'][0] if 'test' in info and info['test'] else train_info
        
        # 获取Fraud Rate
        fraud_rate = 0
        if 'label_distribution' in train_info:
            label_dist = train_info['label_distribution']
            if 'is_fraud' in label_dist:
                fraud_rate = label_dist['is_fraud'].get('1', 0)
            elif 'isFraud' in label_dist:
                fraud_rate = label_dist['isFraud'].get('1', 0)
        
        dataset_summary.append({
            'Dataset': dataset_name,
            'Train Samples': train_info['n_rows'],
            'Test Samples': test_info['n_rows'],
            'Features': train_info['n_cols'] - 1,  # 减去标签列
            'Fraud Rate (%)': fraud_rate * 100
        })

summary_df = pd.DataFrame(dataset_summary)
summary_df = summary_df.sort_values('Train Samples', ascending=False)

# 显示表格
print("\n" + "="*100)
print("📊 Dataset特性总览")
print("="*100 + "\n")
print(summary_df.to_string(index=False))
print("\n")

# 可视化Dataset特性
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. samples量comparison
ax = axes[0, 0]
x = np.arange(len(summary_df))
width = 0.35
ax.bar(x - width/2, summary_df['Train Samples'], width, label='Training集', alpha=0.8)
ax.bar(x + width/2, summary_df['Test Samples'], width, label='Test set', alpha=0.8)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('samples数量', fontsize=12)
ax.set_title('Datasetsamples量comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Dataset'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. features数量comparison
ax = axes[0, 1]
ax.barh(summary_df['Dataset'], summary_df['Features'], color='skyblue')
ax.set_xlabel('features数量', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
ax.set_title('features数量comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 3. Fraud Ratecomparison
ax = axes[1, 0]
colors = ['green' if x < 5 else 'orange' if x < 20 else 'red' 
          for x in summary_df['Fraud Rate (%)']]
ax.barh(summary_df['Dataset'], summary_df['Fraud Rate (%)'], color=colors, alpha=0.7)
ax.set_xlabel('Fraud Rate (%)', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
ax.set_title('class imbalance程度 (Fraud Rate)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
# 添加阈Value线
ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='in度imbalance')
ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='轻度imbalance')
ax.legend()

# 4. Dataset规模散点图
ax = axes[1, 1]
scatter = ax.scatter(summary_df['Features'], summary_df['Train Samples'], 
                    s=summary_df['Fraud Rate (%)'] * 100, 
                    c=summary_df['Fraud Rate (%)'], 
                    cmap='RdYlGn_r', alpha=0.6, edgecolors='black')
for idx, row in summary_df.iterrows():
    ax.annotate(row['Dataset'], 
               (row['Features'], row['Train Samples']),
               fontsize=9, ha='center')
ax.set_xlabel('features数量', fontsize=12)
ax.set_ylabel('Trainingsamples数量', fontsize=12)
ax.set_title('Dataset规模分布 (气泡大小=Fraud Rate)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Fraud Rate (%)')

plt.tight_layout()
plt.show()

print("\n💡 Dataset特性分析:")
print("  - 最大Dataset:", summary_df.iloc[0]['Dataset'], 
      f"({summary_df.iloc[0]['Train Samples']:,} samples)")
print("  - 最小Dataset:", summary_df.iloc[-1]['Dataset'], 
      f"({summary_df.iloc[-1]['Train Samples']:,} samples)")
print("  - 最高dimensions度:", summary_df.loc[summary_df['Features'].idxmax()]['Dataset'],
      f"({summary_df['Features'].max()} features)")
print("  - 最imbalance:", summary_df.loc[summary_df['Fraud Rate (%)'].idxmin()]['Dataset'],
      f"({summary_df['Fraud Rate (%)'].min():.2f}% Fraud Rate)")
print("  - 最平衡:", summary_df.loc[summary_df['Fraud Rate (%)'].idxmax()]['Dataset'],
      f"({summary_df['Fraud Rate (%)'].max():.2f}% Fraud Rate)")
print()


# ---
# 
# ## 📝 实验completed清单
# 
# completed实验后，你应该得to：
# 
# ✅ **数据分析**
# - [ ] 7Datasetof详细描述
# - [ ] 数据分布可视化
# - [ ] class imbalance分析
# 
# ✅ **ModelTraining**
# - [ ] 至少8Modelin每Dataset上ofTraining
# - [ ] 完整ofTraining日志
# - [ ] Model保存（可选）
# 
# ✅ **评估结果**
# - [ ] `experiment_results.csv` 文件
# - [ ] 所有# Evaluation metrics（Accuracy, Precision, Recall, F1, AUC）
# - [ ] Time性能记录
# 
# ✅ **可视化分析**
# - [ ] F1-Scorecomparison图
# - [ ] ROC-AUCcomparison图
# - [ ] 指标热力图
# - [ ] Time性能图
# - [ ] Datasetdimensions度comparison
# 
# ✅ **分析报告**
# - [ ] BestModel总结
# - [ ] Model类别性能分析
# - [ ] Dataset特性影响分析
# - [ ] 效率分析
# - [ ] Using建议
# 
# ---
# 
# ## 🎯 下一步建议
# 
# ### 进阶实验
# 1. **features工程**: 尝试创建新featuresImprovementModel性能
# 2. **超参数调优**: UsingGridSearchorBayesian优化
# 3. **集成学习**: 组合多Modelof预测结果
# 4. **# Deep learning**: 尝试LSTM、Transformeretc架构
# 5. **解释性分析**: UsingSHAPValue分析features重要性
# 
# ### 论文撰写
# 1. **方法论**: 详细描述# Data preprocessingandModel选择理由
# 2. **实验# Settings**: 记录所有超参数and硬件Configuration
# 3. **结果分析**: 深入分析as什么某些Model表现更好
# 4. **讨论**: 比较你of结果and文献inof结果
# 5. **结论**: 总结关键发现and实践建议
# 
# ### 代码优化
# 1. **并行Training**: Usingjoblib并行Training多Model
# 2. **增量学习**: for于大DatasetUsingmini-batchTraining
# 3. **Model持久化**: 保存Training好ofModel以便复with
# 4. **日志记录**: Usinglogging模块记录详细日志
# 5. **Configuration管理**: UsingYAMLConfiguration文件管理参数
# 
# 祝你实验顺利！🚀

# In[ ]:




