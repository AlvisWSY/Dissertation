"""
Results Analysis Script
=======================

Comprehensive analysis of experimental results.

Author: FYP Analysis
Date: 2025-11-12
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from experiment_refactored import RESULTS_DIR

def main():
    """Main analysis function"""
    
    results_csv = RESULTS_DIR / 'experiment_results.csv'
    
    if not results_csv.exists():
        print(f"❌ No results file found at: {results_csv}")
        return
    
    # Read results
    df = pd.read_csv(results_csv)
    
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*100)
    
    # =============================================================================
    # 1. OVERVIEW STATISTICS
    # =============================================================================
    print("\n" + "-"*100)
    print("1️⃣  OVERVIEW STATISTICS")
    print("-"*100)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total Experiments: {len(df)}")
    print(f"   Datasets: {df['dataset'].nunique()}")
    print(f"   Models: {df['model'].nunique()}")
    print(f"   Imbalance Strategies: {df['imbalance_strategy'].nunique()}")
    
    print(f"\n📁 Completed Datasets:")
    for ds in sorted(df['dataset'].unique()):
        count = len(df[df['dataset'] == ds])
        models = df[df['dataset'] == ds]['model'].nunique()
        strategies = df[df['dataset'] == ds]['imbalance_strategy'].nunique()
        print(f"   • {ds:30s}: {count:3d} experiments ({models} models × {strategies} strategies)")
    
    print(f"\n🤖 Models Evaluated:")
    for model in sorted(df['model'].unique()):
        count = len(df[df['model'] == model])
        print(f"   • {model:25s}: {count:3d} experiments")
    
    print(f"\n⚖️  Imbalance Strategies:")
    for strategy in sorted(df['imbalance_strategy'].unique()):
        count = len(df[df['imbalance_strategy'] == strategy])
        avg_f1 = df[df['imbalance_strategy'] == strategy]['f1_score'].mean()
        print(f"   • {strategy:20s}: {count:3d} experiments, Avg F1: {avg_f1:.4f}")
    
    # =============================================================================
    # 2. BEST MODELS PER DATASET
    # =============================================================================
    print("\n" + "-"*100)
    print("2️⃣  BEST MODELS PER DATASET (by F1-Score)")
    print("-"*100)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        best_row = dataset_df.loc[dataset_df['f1_score'].idxmax()]
        
        print(f"\n📊 {dataset}:")
        print(f"   🥇 Best Model: {best_row['model']}")
        print(f"   📋 Strategy: {best_row['imbalance_strategy']}")
        print(f"   📈 F1-Score: {best_row['f1_score']:.4f}")
        print(f"   📈 ROC-AUC: {best_row['roc_auc']:.4f}")
        print(f"   📈 Accuracy: {best_row['accuracy']:.4f}")
        print(f"   📈 Precision: {best_row['precision']:.4f}")
        print(f"   📈 Recall: {best_row['recall']:.4f}")
        print(f"   ⏱️  Training Time: {best_row['train_time']:.2f}s")
    
    # =============================================================================
    # 3. OVERALL BEST MODELS (across all datasets)
    # =============================================================================
    print("\n" + "-"*100)
    print("3️⃣  OVERALL BEST MODELS (Average Performance Across All Datasets)")
    print("-"*100)
    
    # Group by model and calculate average metrics
    model_performance = df.groupby('model').agg({
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'train_time': ['mean', 'sum']
    }).round(4)
    
    # Sort by F1 score
    model_performance_sorted = model_performance.sort_values(('f1_score', 'mean'), ascending=False)
    
    print("\n🏆 TOP 10 MODELS (by Average F1-Score):")
    for idx, (model, row) in enumerate(model_performance_sorted.head(10).iterrows(), 1):
        print(f"\n{idx}. {model}")
        print(f"   F1-Score: {row[('f1_score', 'mean')]:.4f} (±{row[('f1_score', 'std')]:.4f})")
        print(f"   ROC-AUC: {row[('roc_auc', 'mean')]:.4f} (±{row[('roc_auc', 'std')]:.4f})")
        print(f"   Accuracy: {row[('accuracy', 'mean')]:.4f} (±{row[('accuracy', 'std')]:.4f})")
        print(f"   Precision: {row[('precision', 'mean')]:.4f} (±{row[('precision', 'std')]:.4f})")
        print(f"   Recall: {row[('recall', 'mean')]:.4f} (±{row[('recall', 'std')]:.4f})")
        print(f"   Avg Training Time: {row[('train_time', 'mean')]:.2f}s")
        print(f"   Total Training Time: {row[('train_time', 'sum')]:.2f}s")
    
    # =============================================================================
    # 4. IMBALANCE STRATEGY COMPARISON
    # =============================================================================
    print("\n" + "-"*100)
    print("4️⃣  IMBALANCE STRATEGY EFFECTIVENESS")
    print("-"*100)
    
    print("\n⚖️  Strategy Rankings (by F1-Score):")
    strategy_f1 = df.groupby('imbalance_strategy')['f1_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for idx, (strategy, row) in enumerate(strategy_f1.iterrows(), 1):
        print(f"{idx}. {strategy:20s}: F1={row['mean']:.4f} (±{row['std']:.4f}), n={int(row['count'])}")
    
    # Best strategy per dataset
    print("\n📊 Best Strategy Per Dataset:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        best_strategy = dataset_df.groupby('imbalance_strategy')['f1_score'].mean().idxmax()
        best_f1 = dataset_df.groupby('imbalance_strategy')['f1_score'].mean().max()
        worst_strategy = dataset_df.groupby('imbalance_strategy')['f1_score'].mean().idxmin()
        worst_f1 = dataset_df.groupby('imbalance_strategy')['f1_score'].mean().min()
        print(f"   {dataset:30s}: Best={best_strategy:10s} (F1={best_f1:.4f}), Worst={worst_strategy:10s} (F1={worst_f1:.4f})")
    
    # =============================================================================
    # 5. SPEED vs PERFORMANCE ANALYSIS
    # =============================================================================
    print("\n" + "-"*100)
    print("5️⃣  SPEED vs PERFORMANCE ANALYSIS")
    print("-"*100)
    
    model_speed = df.groupby('model').agg({
        'train_time': 'mean',
        'inference_time': 'mean',
        'f1_score': 'mean'
    }).sort_values('train_time')
    
    print("\n⚡ Fastest Models (Training Time < 1 second):")
    fast_models = model_speed[model_speed['train_time'] < 1.0]
    if len(fast_models) > 0:
        for model, row in fast_models.iterrows():
            print(f"   • {model:25s}: Train={row['train_time']:.3f}s, Inference={row['inference_time']:.4f}s, F1={row['f1_score']:.4f}")
    else:
        print("   No models with training time < 1s")
    
    print("\n🐌 Slowest Models (Training Time > 100 seconds):")
    slow_models = model_speed[model_speed['train_time'] > 100]
    if len(slow_models) > 0:
        for model, row in slow_models.iterrows():
            print(f"   • {model:25s}: Train={row['train_time']:.2f}s, Inference={row['inference_time']:.4f}s, F1={row['f1_score']:.4f}")
    else:
        print("   No models with training time > 100s")
    
    print("\n🎯 Best Speed-Performance Trade-off (F1 > 0.70 and Train Time < 10s):")
    efficient = model_speed[(model_speed['f1_score'] > 0.70) & (model_speed['train_time'] < 10)]
    if len(efficient) > 0:
        efficient_sorted = efficient.sort_values('f1_score', ascending=False)
        for model, row in efficient_sorted.iterrows():
            print(f"   • {model:25s}: Train={row['train_time']:.2f}s, F1={row['f1_score']:.4f}")
    else:
        print("   No models meeting criteria (F1 > 0.70, Time < 10s)")
    
    # =============================================================================
    # 6. DATASET DIFFICULTY ANALYSIS
    # =============================================================================
    print("\n" + "-"*100)
    print("6️⃣  DATASET DIFFICULTY ANALYSIS")
    print("-"*100)
    
    print("\n📊 Dataset Rankings (by Average F1-Score):")
    dataset_stats = df.groupby('dataset').agg({
        'f1_score': ['mean', 'max', 'min', 'std'],
        'roc_auc': ['mean', 'max']
    }).round(4)
    
    dataset_f1 = df.groupby('dataset')['f1_score'].mean().sort_values(ascending=False)
    for idx, (dataset, avg_f1) in enumerate(dataset_f1.items(), 1):
        max_f1 = df[df['dataset'] == dataset]['f1_score'].max()
        min_f1 = df[df['dataset'] == dataset]['f1_score'].min()
        std_f1 = df[df['dataset'] == dataset]['f1_score'].std()
        avg_auc = df[df['dataset'] == dataset]['roc_auc'].mean()
        print(f"{idx}. {dataset:30s}: Avg F1={avg_f1:.4f}, Max={max_f1:.4f}, Min={min_f1:.4f}, Std={std_f1:.4f}, AUC={avg_auc:.4f}")
    
    # Identify easy vs hard datasets
    easy_datasets = dataset_f1[dataset_f1 > 0.85].index.tolist()
    medium_datasets = dataset_f1[(dataset_f1 >= 0.60) & (dataset_f1 <= 0.85)].index.tolist()
    hard_datasets = dataset_f1[dataset_f1 < 0.60].index.tolist()
    
    if easy_datasets:
        print(f"\n✅ Easy Datasets (Avg F1 > 0.85):")
        for ds in easy_datasets:
            print(f"   • {ds} (F1={dataset_f1[ds]:.4f})")
    
    if medium_datasets:
        print(f"\n⚠️  Medium Datasets (0.60 ≤ Avg F1 ≤ 0.85):")
        for ds in medium_datasets:
            print(f"   • {ds} (F1={dataset_f1[ds]:.4f})")
    
    if hard_datasets:
        print(f"\n❌ Hard Datasets (Avg F1 < 0.60):")
        for ds in hard_datasets:
            print(f"   • {ds} (F1={dataset_f1[ds]:.4f})")
    
    # =============================================================================
    # 7. MODEL CONSISTENCY ANALYSIS
    # =============================================================================
    print("\n" + "-"*100)
    print("7️⃣  MODEL CONSISTENCY ANALYSIS")
    print("-"*100)
    
    model_consistency = df.groupby('model')['f1_score'].std().sort_values()
    print("\n📊 Most Consistent Models (lowest F1 variance across datasets):")
    for idx, (model, std) in enumerate(model_consistency.head(5).items(), 1):
        mean_f1 = df[df['model'] == model]['f1_score'].mean()
        print(f"{idx}. {model:25s}: Std={std:.4f}, Mean F1={mean_f1:.4f}")
    
    print("\n📊 Least Consistent Models (highest F1 variance):")
    for idx, (model, std) in enumerate(model_consistency.tail(5).items(), 1):
        mean_f1 = df[df['model'] == model]['f1_score'].mean()
        print(f"{idx}. {model:25s}: Std={std:.4f}, Mean F1={mean_f1:.4f}")
    
    # =============================================================================
    # 8. KEY INSIGHTS & RECOMMENDATIONS
    # =============================================================================
    print("\n" + "="*100)
    print("8️⃣  KEY INSIGHTS & RECOMMENDATIONS")
    print("="*100)
    
    print("\n🎯 TOP INSIGHTS:")
    
    # Insight 1: Best overall model
    best_model = model_performance_sorted.index[0]
    best_model_f1 = model_performance_sorted.iloc[0][('f1_score', 'mean')]
    best_model_std = model_performance_sorted.iloc[0][('f1_score', 'std')]
    print(f"\n1️⃣  BEST OVERALL MODEL:")
    print(f"   → {best_model}")
    print(f"   → Average F1-Score: {best_model_f1:.4f} (±{best_model_std:.4f})")
    print(f"   → Recommendation: Use for general-purpose fraud detection")
    
    # Insight 2: Best imbalance strategy
    best_strategy = strategy_f1.index[0]
    best_strategy_f1 = strategy_f1.iloc[0]['mean']
    print(f"\n2️⃣  BEST IMBALANCE STRATEGY:")
    print(f"   → {best_strategy}")
    print(f"   → Average F1-Score: {best_strategy_f1:.4f}")
    
    # Insight 3: Speed-performance winner
    if len(efficient) > 0:
        best_efficient = efficient['f1_score'].idxmax()
        print(f"\n3️⃣  BEST SPEED-PERFORMANCE TRADE-OFF:")
        print(f"   → {best_efficient}")
        print(f"   → F1={efficient.loc[best_efficient, 'f1_score']:.4f}, Train Time={efficient.loc[best_efficient, 'train_time']:.2f}s")
        print(f"   → Recommendation: Ideal for production deployment")
    
    # Insight 4: Model consistency
    most_consistent = model_consistency.index[0]
    print(f"\n4️⃣  MOST CONSISTENT MODEL (lowest F1 variance):")
    print(f"   → {most_consistent} (Std={model_consistency.iloc[0]:.4f})")
    print(f"   → Recommendation: Use when predictable performance is critical")
    
    # Insight 5: Dataset-specific recommendations
    print(f"\n5️⃣  DATASET-SPECIFIC RECOMMENDATIONS:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        best_model = dataset_df.loc[dataset_df['f1_score'].idxmax(), 'model']
        best_f1 = dataset_df['f1_score'].max()
        avg_f1 = dataset_df['f1_score'].mean()
        
        if avg_f1 > 0.85:
            status = "✅ Excellent"
        elif avg_f1 > 0.70:
            status = "⚠️  Good"
        else:
            status = "❌ Challenging"
        
        print(f"   {dataset:30s}: {status}")
        print(f"      → Best: {best_model} (F1={best_f1:.4f})")
        print(f"      → Average: {avg_f1:.4f}")
    
    # =============================================================================
    # 9. RECOMMENDATIONS FOR PRODUCTION
    # =============================================================================
    print("\n" + "-"*100)
    print("9️⃣  RECOMMENDATIONS FOR PRODUCTION")
    print("-"*100)
    
    print("\n🚀 FOR DEPLOYMENT:")
    
    # Find models with good F1 and reasonable time
    production_candidates = model_speed[
        (model_speed['f1_score'] > 0.75) & 
        (model_speed['train_time'] < 50)
    ].sort_values('f1_score', ascending=False)
    
    if len(production_candidates) > 0:
        print(f"\n✅ Production-Ready Models (F1 > 0.75, Train Time < 50s):")
        for idx, (model, row) in enumerate(production_candidates.head(5).iterrows(), 1):
            print(f"{idx}. {model}")
            print(f"   F1-Score: {row['f1_score']:.4f}")
            print(f"   Training Time: {row['train_time']:.2f}s")
            print(f"   Inference Time: {row['inference_time']:.4f}s")
    
    # High-accuracy models
    high_accuracy = model_speed[model_speed['f1_score'] > 0.85].sort_values('f1_score', ascending=False)
    if len(high_accuracy) > 0:
        print(f"\n🎯 High-Performance Models (F1 > 0.85):")
        for idx, (model, row) in enumerate(high_accuracy.head(5).iterrows(), 1):
            print(f"{idx}. {model}")
            print(f"   F1-Score: {row['f1_score']:.4f}")
            print(f"   Training Time: {row['train_time']:.2f}s")
    
    # =============================================================================
    # 10. STATISTICAL SUMMARY
    # =============================================================================
    print("\n" + "-"*100)
    print("🔟 STATISTICAL SUMMARY")
    print("-"*100)
    
    print("\n📈 Overall Performance Metrics:")
    print(f"   F1-Score:   {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f} (range: {df['f1_score'].min():.4f} - {df['f1_score'].max():.4f})")
    print(f"   ROC-AUC:    {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f} (range: {df['roc_auc'].min():.4f} - {df['roc_auc'].max():.4f})")
    print(f"   Accuracy:   {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f} (range: {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f})")
    print(f"   Precision:  {df['precision'].mean():.4f} ± {df['precision'].std():.4f} (range: {df['precision'].min():.4f} - {df['precision'].max():.4f})")
    print(f"   Recall:     {df['recall'].mean():.4f} ± {df['recall'].std():.4f} (range: {df['recall'].min():.4f} - {df['recall'].max():.4f})")
    
    print(f"\n⏱️  Time Statistics:")
    print(f"   Training Time:   Mean={df['train_time'].mean():.2f}s, Median={df['train_time'].median():.2f}s, Max={df['train_time'].max():.2f}s")
    print(f"   Inference Time:  Mean={df['inference_time'].mean():.4f}s, Median={df['inference_time'].median():.4f}s, Max={df['inference_time'].max():.4f}s")
    
    print("\n" + "="*100)
    print("✅ ANALYSIS COMPLETE")
    print("="*100)
    
    # Save summary to file
    print(f"\n💾 Results saved to: {results_csv}")
    print(f"📊 Total experiments analyzed: {len(df)}")
    print(f"🎯 Key finding: {best_model} is the best overall model with F1={best_model_f1:.4f}")

if __name__ == "__main__":
    main()
