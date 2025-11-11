"""
Visualization and Main Execution Script
========================================

This module contains visualization functions and the main execution script
for running the complete experiment.

Author: Experiment Framework  
Date: 2025-11-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Import from other modules
from experiment_refactored import (
    logger, RESULTS_DIR, VIZ_DIR, DATASET_CONFIGS,
    DATA_DIR, DatasetLoader, log_memory_usage, clear_memory
)
from experiment_models import ExperimentRunner
from ieee_preprocessing import HighDimensionalPreprocessor


# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

class ResultsVisualizer:
    """Comprehensive results visualization"""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path = VIZ_DIR):
        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.comparison_dir = self.output_dir / 'comparisons'
        self.comparison_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_model_comparison(self, metric: str = 'f1_score'):
        """Compare models across all datasets"""
        logger.info(f"Plotting model comparison for {metric}")
        
        # Check if results_df is empty or missing required columns
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping model comparison plot")
            return
        
        if 'model' not in self.results_df.columns or metric not in self.results_df.columns:
            logger.warning(f"Missing required columns in results DataFrame, skipping plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Average performance across all datasets
        ax = axes[0, 0]
        avg_scores = self.results_df.groupby('model')[metric].mean().sort_values(ascending=False)
        avg_scores.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'Average {metric.upper()} Across All Datasets', fontsize=14, fontweight='bold')
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Performance by dataset
        ax = axes[0, 1]
        pivot_data = self.results_df.pivot_table(
            index='model', columns='dataset', values=metric, aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric.upper()} by Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Heatmap
        ax = axes[1, 0]
        pivot_data = self.results_df.pivot_table(
            index='model', columns='dataset', values=metric, aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                   cbar_kws={'label': metric.upper()})
        ax.set_title(f'{metric.upper()} Heatmap', fontsize=14, fontweight='bold')
        
        # 4. Box plot
        ax = axes[1, 1]
        models = self.results_df['model'].unique()
        data_to_plot = [self.results_df[self.results_df['model'] == model][metric].values 
                       for model in models]
        bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.comparison_dir / f'model_comparison_{metric}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def plot_imbalance_comparison(self):
        """Compare imbalance handling strategies"""
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping imbalance comparison plot")
            return
            
        if 'imbalance_strategy' not in self.results_df.columns:
            logger.warning("No imbalance strategy column found")
            return
        
        logger.info("Plotting imbalance strategy comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        strategies = self.results_df['imbalance_strategy'].unique()
        
        # 1. F1-Score comparison
        ax = axes[0, 0]
        for strategy in strategies:
            data = self.results_df[self.results_df['imbalance_strategy'] == strategy]
            avg_scores = data.groupby('model')['f1_score'].mean()
            ax.plot(range(len(avg_scores)), avg_scores.values, marker='o', label=strategy, linewidth=2)
        ax.set_title('F1-Score by Imbalance Strategy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Rank', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Average improvement over baseline
        ax = axes[0, 1]
        if 'none' in strategies:
            baseline = self.results_df[self.results_df['imbalance_strategy'] == 'none']['f1_score'].mean()
            improvements = []
            labels = []
            for strategy in strategies:
                if strategy != 'none':
                    score = self.results_df[self.results_df['imbalance_strategy'] == strategy]['f1_score'].mean()
                    improvement = ((score - baseline) / baseline) * 100
                    improvements.append(improvement)
                    labels.append(strategy)
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax.barh(labels, improvements, color=colors, alpha=0.7)
            ax.set_title('F1-Score Improvement over Baseline (%)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Improvement (%)', fontsize=12)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
        
        # 3. Strategy performance by dataset
        ax = axes[1, 0]
        datasets = self.results_df['dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            data = self.results_df[self.results_df['imbalance_strategy'] == strategy]
            means = [data[data['dataset'] == d]['f1_score'].mean() for d in datasets]
            ax.bar(x + i * width, means, width, label=strategy, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Average F1-Score', fontsize=12)
        ax.set_title('Performance by Dataset and Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Best strategy distribution
        ax = axes[1, 1]
        best_strategies = []
        for dataset in datasets:
            dataset_data = self.results_df[self.results_df['dataset'] == dataset]
            for model in dataset_data['model'].unique():
                model_data = dataset_data[dataset_data['model'] == model]
                if len(model_data) > 0:
                    best_idx = model_data['f1_score'].idxmax()
                    best_strategy = model_data.loc[best_idx, 'imbalance_strategy']
                    best_strategies.append(best_strategy)
        
        strategy_counts = pd.Series(best_strategies).value_counts()
        ax.pie(strategy_counts.values, labels=strategy_counts.index,
              autopct='%1.1f%%', startangle=90)
        ax.set_title('Best Strategy Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.comparison_dir / 'imbalance_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def plot_time_analysis(self):
        """Analyze training and inference time"""
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping time analysis plot")
            return
            
        logger.info("Plotting time analysis")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training time
        ax = axes[0]
        avg_train_time = self.results_df.groupby('model')['train_time'].mean().sort_values(ascending=False)
        avg_train_time.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Average Training Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        # Inference time
        ax = axes[1]
        avg_inference_time = self.results_df.groupby('model')['inference_time'].mean().sort_values(ascending=False)
        avg_inference_time.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title('Average Inference Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.comparison_dir / 'time_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def plot_all_metrics_heatmap(self):
        """Heatmap for all metrics"""
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping metrics heatmap")
            return
            
        logger.info("Plotting all metrics heatmap")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            pivot_data = self.results_df.pivot_table(
                index='model', columns='dataset', values=metric, aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu',
                       ax=axes[idx], cbar_kws={'label': metric.upper()})
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        
        # Hide extra subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        output_path = self.comparison_dir / 'all_metrics_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def generate_summary_report(self):
        """Generate text summary report"""
        logger.section("Generating Summary Report", level=1)
        
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping summary report")
            return
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("EXPERIMENT SUMMARY REPORT")
        report_lines.append("="*100)
        report_lines.append("")
        
        # Overall best models
        report_lines.append("BEST MODELS BY METRIC")
        report_lines.append("-"*100)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            best_idx = self.results_df[metric].idxmax()
            best = self.results_df.loc[best_idx]
            report_lines.append(f"Best {metric.upper():12s}: {best['model']:20s} "
                              f"(Dataset: {best['dataset']:20s}, Value: {best[metric]:.4f})")
        report_lines.append("")
        
        # Per-dataset summary
        report_lines.append("PER-DATASET SUMMARY")
        report_lines.append("-"*100)
        for dataset in sorted(self.results_df['dataset'].unique()):
            dataset_data = self.results_df[self.results_df['dataset'] == dataset]
            best_model = dataset_data.loc[dataset_data['f1_score'].idxmax()]
            
            report_lines.append(f"\nDataset: {dataset}")
            report_lines.append(f"  Best Model: {best_model['model']}")
            report_lines.append(f"  F1-Score: {best_model['f1_score']:.4f}")
            report_lines.append(f"  ROC-AUC: {best_model['roc_auc']:.4f}")
            report_lines.append(f"  Training Time: {best_model['train_time']:.2f}s")
        report_lines.append("")
        
        # Model rankings
        report_lines.append("OVERALL MODEL RANKINGS (by F1-Score)")
        report_lines.append("-"*100)
        avg_f1 = self.results_df.groupby('model')['f1_score'].mean().sort_values(ascending=False)
        for rank, (model, score) in enumerate(avg_f1.items(), 1):
            report_lines.append(f"  {rank:2d}. {model:20s} - F1: {score:.4f}")
        report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.comparison_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved: {report_path}")
        
        # Also log to console
        for line in report_lines:
            logger.info(line)
    
    def generate_all_visualizations(self):
        """Generate all visualization outputs"""
        logger.section("Generating All Visualizations", level=1)
        
        if self.results_df.empty:
            logger.warning("Results DataFrame is empty, skipping all visualizations")
            logger.warning("No experiments were completed successfully")
            return
        
        self.plot_model_comparison('f1_score')
        self.plot_model_comparison('roc_auc')
        self.plot_imbalance_comparison()
        self.plot_time_analysis()
        self.plot_all_metrics_heatmap()
        self.generate_summary_report()
        
        logger.info("All visualizations generated successfully!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.section("FRAUD DETECTION EXPERIMENT - MAIN EXECUTION", level=1)
    logger.info("Starting comprehensive fraud detection model comparison experiment")
    log_memory_usage()
    
    # Initialize experiment runner
    runner = ExperimentRunner(
        compare_imbalance=True,
        use_sampling_for_slow_models=True
    )
    
    # Select datasets to run (can be configured)
    #datasets_to_run = list(DATASET_CONFIGS.keys())
    datasets_to_run = ['counterfeit_products', 'creditCardPCA']  # For quick testing
    
    logger.info(f"Datasets to process: {datasets_to_run}")
    logger.info(f"Total datasets: {len(datasets_to_run)}")
    
    # Process each dataset
    all_results = {}
    for idx, dataset_name in enumerate(datasets_to_run, 1):
        try:
            logger.section(f"DATASET {idx}/{len(datasets_to_run)}: {dataset_name}", level=1)
            logger.progress(idx, len(datasets_to_run), "Dataset Processing")
            
            # Get dataset configuration
            config = DATASET_CONFIGS[dataset_name]
            dataset_size = config['size_category']
            
            # Load and preprocess data
            logger.section(f"Loading Dataset: {dataset_name}", level=2)
            loader = DatasetLoader(
                dataset_name,
                max_samples=config['max_samples'],
                handle_sparse=config['handle_sparse']
            )
            
            train_df, test_df = loader.load_data()
            
            # AUTOMATIC HIGH-DIMENSIONAL DATASET DETECTION
            n_features = train_df.shape[1] - 1  # Exclude target column
            n_samples = len(train_df)
            ratio = n_samples / n_features if n_features > 0 else 999
            
            logger.info(f"Dataset dimensions: {n_samples:,} samples × {n_features} features")
            logger.info(f"Samples-to-features ratio: {ratio:.2f}")
            
            # Use special preprocessing for high-dimensional datasets
            use_feature_selection = (ratio < 50 or n_features > 200)
            
            if use_feature_selection:
                logger.warning(f"⚠️  HIGH DIMENSIONALITY DETECTED!")
                logger.warning(f"   Ratio: {ratio:.2f} (< 50) or Features: {n_features} (> 200)")
                logger.info("Applying intelligent feature selection to avoid curse of dimensionality...")
                
                # Use high-dimensional preprocessor
                preprocessor = HighDimensionalPreprocessor(
                    target_n_features=min(150, n_features // 2),
                    variance_threshold=0.01,
                    max_missing_ratio=0.7
                )
                
                X_train, X_test, y_train, y_test = preprocessor.fit_transform(
                    train_df, test_df, target_col='is_fraud'
                )
                
                # Log feature importance
                logger.info(preprocessor.get_feature_importance_report())
                
                # Create dummy feature_types for compatibility
                feature_types = {
                    'numerical': list(range(X_train.shape[1])),
                    'categorical': [],
                    'id': [],
                    'timestamp': []
                }
                
                # Convert to DataFrame for report generation
                X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                y_train_series = pd.Series(y_train, name='is_fraud')
                y_test_series = pd.Series(y_test, name='is_fraud')
                
            else:
                logger.info("✓ Normal dimensionality, using standard preprocessing")
                X_train, X_test, y_train, y_test, feature_types = loader.preprocess(
                    train_df, test_df, apply_sampling=False
                )
                
                # For consistency, convert to DataFrame/Series if needed
                if isinstance(X_train, np.ndarray):
                    X_train_df = pd.DataFrame(X_train)
                    X_test_df = pd.DataFrame(X_test)
                else:
                    X_train_df = X_train
                    X_test_df = X_test
                
                if isinstance(y_train, np.ndarray):
                    y_train_series = pd.Series(y_train, name='is_fraud')
                    y_test_series = pd.Series(y_test, name='is_fraud')
                else:
                    y_train_series = y_train
                    y_test_series = y_test
            
            # Generate dataset report with visualizations (non-critical, won't stop execution)
            try:
                # Use appropriate data format for report
                if use_feature_selection:
                    loader.generate_dataset_report(
                        train_df, test_df, X_train_df, y_train_series, X_test_df, y_test_series, feature_types
                    )
                else:
                    loader.generate_dataset_report(
                        train_df, test_df, X_train, y_train, X_test, y_test, feature_types
                    )
            except Exception as e:
                logger.warning(f"Failed to generate dataset report (non-critical): {e}")
            
            log_memory_usage()
            
            # Run all models
            dataset_results = runner.run_all_models(
                X_train, y_train, X_test, y_test, dataset_name, dataset_size
            )
            
            all_results[dataset_name] = dataset_results
            
            # Clear memory immediately
            if use_feature_selection:
                del train_df, test_df, X_train, X_test, y_train, y_test
                del X_train_df, X_test_df, y_train_series, y_test_series, loader, preprocessor
            else:
                del train_df, test_df, X_train, X_test, y_train, y_test, loader
            clear_memory()
            
            logger.info(f"Dataset {dataset_name} completed and memory cleared")
            log_memory_usage()
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Get results DataFrame
    logger.section("COMPILING RESULTS", level=1)
    results_df = runner.evaluator.get_results_df()
    
    # Save results to CSV
    results_csv_path = RESULTS_DIR / 'experiment_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to: {results_csv_path}")
    
    # Generate visualizations
    visualizer = ResultsVisualizer(results_df)
    visualizer.generate_all_visualizations()
    
    # Final summary
    logger.section("EXPERIMENT COMPLETE", level=1)
    logger.info(f"Total datasets processed: {len(all_results)}")
    logger.info(f"Total experiments run: {len(results_df)}")
    logger.info(f"Results saved to: {results_csv_path}")
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info("All outputs are in English as requested")
    log_memory_usage()
    
    return results_df


if __name__ == "__main__":
    results = main()
    logger.info("Experiment finished successfully!")
