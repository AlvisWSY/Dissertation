#!/usr/bin/env python3
"""
快速测试脚本 - 在单个小数据集上测试所有模型
用于验证环境配置是否正确
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

print("="*80)
print("🧪 快速测试 - 验证实验环境")
print("="*80)

# 测试导入
print("\n1️⃣ 检查依赖包...")
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost as xgb
    import lightgbm as lgb
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✅ 所有必要的包都已安装")
except ImportError as e:
    print(f"   ❌ 缺少依赖包: {e}")
    print("   请运行: pip install -r requirements.txt")
    sys.exit(1)

# 检查CUDA
print("\n2️⃣ 检查GPU支持...")
if torch.cuda.is_available():
    print(f"   ✅ CUDA可用, 设备: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA不可用，将使用CPU (MLP和Autoencoder会较慢)")

# 检查数据
print("\n3️⃣ 检查数据集...")
data_dir = Path(__file__).parent.parent / 'data'
if not data_dir.exists():
    print(f"   ❌ 数据目录不存在: {data_dir}")
    sys.exit(1)

test_dataset = 'counterfeit_products'
test_data_path = data_dir / test_dataset / 'train'
if not test_data_path.exists():
    print(f"   ❌ 测试数据集不存在: {test_data_path}")
    sys.exit(1)
print(f"   ✅ 数据目录正常: {data_dir}")

# 运行快速测试
print("\n4️⃣ 运行快速测试...")
print(f"   使用数据集: {test_dataset} (样本量小，训练快)")
print("   " + "-"*70)

try:
    # 这里需要从experiment.ipynb中复制必要的类定义
    # 简化版本，只测试基本功能
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    import time
    
    # 加载数据
    train_file = list(test_data_path.glob('*.csv'))[0]
    df = pd.read_csv(train_file)
    print(f"   📊 数据形状: {df.shape}")
    
    # 简单预处理
    label_col = 'is_fraud'
    X = df.drop(columns=[label_col, 'timestamp', 'seller_id'], errors='ignore')
    y = df[label_col]
    
    # 编码类别特征
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.factorize(X[col])[0]
    
    # 简单分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 测试逻辑回归
    print("\n   测试 Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = lr.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"   ✅ 训练时间: {train_time:.2f}s, F1-Score: {f1:.4f}")
    
    # 测试随机森林
    print("\n   测试 Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"   ✅ 训练时间: {train_time:.2f}s, F1-Score: {f1:.4f}")
    
    # 测试XGBoost
    print("\n   测试 XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"   ✅ 训练时间: {train_time:.2f}s, F1-Score: {f1:.4f}")
    
    print("\n" + "="*80)
    print("🎉 测试通过！环境配置正确")
    print("="*80)
    print("\n📝 下一步:")
    print("   1. 打开 src/experiment.ipynb")
    print("   2. 运行所有单元格开始完整实验")
    print("   3. 查看 src/quick_start.md 了解更多信息")
    print("\n")
    
except Exception as e:
    print(f"\n   ❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
