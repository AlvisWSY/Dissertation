"""
PaySim Python vs Java 对比分析脚本
比较 Python 版本和 Java 版本生成的数据
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp
import matplotlib.pyplot as plt
import sys


def load_data(python_path, java_path):
    """加载两个版本的数据"""
    print("加载数据...")
    df_python = pd.read_csv(python_path)
    df_java = pd.read_csv(java_path)
    
    # 统一列名（Java版本可能有不同的列名）
    if 'action' in df_java.columns:
        df_java = df_java.rename(columns={'action': 'type'})
    
    print(f"Python版本: {len(df_python)} 条交易")
    print(f"Java版本: {len(df_java)} 条交易")
    
    return df_python, df_java


def compare_transaction_types(df_python, df_java):
    """比较交易类型分布"""
    print("\n" + "=" * 60)
    print("交易类型分布对比")
    print("=" * 60)
    
    type_counts_py = df_python['type'].value_counts(normalize=True).sort_index()
    type_counts_java = df_java['type'].value_counts(normalize=True).sort_index()
    
    comparison = pd.DataFrame({
        'Python': type_counts_py,
        'Java': type_counts_java
    }).fillna(0)
    comparison['Difference (%)'] = (comparison['Python'] - comparison['Java']) * 100
    
    print(comparison)
    
    # 卡方检验
    all_types = list(set(type_counts_py.index) | set(type_counts_java.index))
    obs_py = [type_counts_py.get(t, 0) * len(df_python) for t in all_types]
    obs_java = [type_counts_java.get(t, 0) * len(df_java) for t in all_types]
    
    chi2, p_value, _, _ = chi2_contingency([obs_py, obs_java])
    print(f"\n卡方检验 p-value: {p_value:.6f}")
    if p_value > 0.05:
        print("✓ 交易类型分布无显著差异")
    else:
        print("✗ 交易类型分布存在显著差异")
    
    return comparison


def compare_amounts(df_python, df_java):
    """比较交易金额分布"""
    print("\n" + "=" * 60)
    print("交易金额统计对比")
    print("=" * 60)
    
    stats = pd.DataFrame({
        'Python': df_python['amount'].describe(),
        'Java': df_java['amount'].describe()
    })
    stats['Difference (%)'] = ((stats['Python'] - stats['Java']) / stats['Java'] * 100).round(2)
    
    print(stats)
    
    # KS检验
    ks_stat, p_value = ks_2samp(df_python['amount'], df_java['amount'])
    print(f"\nKS检验统计量: {ks_stat:.6f}")
    print(f"KS检验 p-value: {p_value:.6f}")
    if p_value > 0.05:
        print("✓ 交易金额分布无显著差异")
    else:
        print("✗ 交易金额分布存在显著差异")


def compare_fraud_rates(df_python, df_java):
    """比较欺诈率"""
    print("\n" + "=" * 60)
    print("欺诈统计对比")
    print("=" * 60)
    
    fraud_rate_py = df_python['isFraud'].mean()
    fraud_rate_java = df_java['isFraud'].mean()
    
    flagged_rate_py = df_python['isFlaggedFraud'].mean()
    flagged_rate_java = df_java['isFlaggedFraud'].mean()
    
    print(f"{'指标':<20} {'Python':>12} {'Java':>12} {'差异':>12}")
    print("-" * 60)
    print(f"{'欺诈率 (isFraud)':<20} {fraud_rate_py:>11.4%} {fraud_rate_java:>11.4%} {(fraud_rate_py-fraud_rate_java):>11.4%}")
    print(f"{'标记率 (Flagged)':<20} {flagged_rate_py:>11.4%} {flagged_rate_java:>11.4%} {(flagged_rate_py-flagged_rate_java):>11.4%}")
    
    # 按交易类型统计欺诈率
    print("\n按交易类型的欺诈率:")
    for tx_type in ['TRANSFER', 'CASH_OUT']:
        if tx_type in df_python['type'].values and tx_type in df_java['type'].values:
            fraud_py = df_python[df_python['type']==tx_type]['isFraud'].mean()
            fraud_java = df_java[df_java['type']==tx_type]['isFraud'].mean()
            print(f"  {tx_type:<15} Python: {fraud_py:>7.4%}  Java: {fraud_java:>7.4%}")


def plot_comparison(df_python, df_java, output_path='comparison.png'):
    """绘制对比图表"""
    print("\n生成对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 交易类型分布
    ax = axes[0, 0]
    type_counts_py = df_python['type'].value_counts().sort_index()
    type_counts_java = df_java['type'].value_counts().sort_index()
    
    x = np.arange(len(type_counts_py))
    width = 0.35
    
    ax.bar(x - width/2, type_counts_py.values, width, label='Python', alpha=0.8)
    ax.bar(x + width/2, type_counts_java.values, width, label='Java', alpha=0.8)
    ax.set_xlabel('Transaction Type')
    ax.set_ylabel('Count')
    ax.set_title('Transaction Type Distribution')
    ax.set_xticks(x)
    ax.set_xticks(x, type_counts_py.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 交易金额分布（对数）
    ax = axes[0, 1]
    ax.hist(np.log10(df_python['amount']+1), bins=50, alpha=0.5, label='Python', density=True)
    ax.hist(np.log10(df_java['amount']+1), bins=50, alpha=0.5, label='Java', density=True)
    ax.set_xlabel('log10(Amount + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Transaction Amount Distribution (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 时间步交易数量
    ax = axes[1, 0]
    step_counts_py = df_python.groupby('step').size()
    step_counts_java = df_java.groupby('step').size()
    
    ax.plot(step_counts_py.index, step_counts_py.values, label='Python', alpha=0.7)
    ax.plot(step_counts_java.index, step_counts_java.values, label='Java', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Transaction Count')
    ax.set_title('Transactions per Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 欺诈率对比
    ax = axes[1, 1]
    fraud_stats = pd.DataFrame({
        'Python': [
            df_python['isFraud'].mean(),
            df_python['isFlaggedFraud'].mean()
        ],
        'Java': [
            df_java['isFraud'].mean(),
            df_java['isFlaggedFraud'].mean()
        ]
    }, index=['Fraud Rate', 'Flagged Rate'])
    
    fraud_stats.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Fraud Statistics Comparison')
    ax.set_ylabel('Rate')
    ax.set_xticklabels(fraud_stats.index, rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")
    

def main():
    if len(sys.argv) < 3:
        print("使用方法: python compare_outputs.py <python_csv> <java_csv>")
        print("\n示例:")
        print("  python compare_outputs.py \\")
        print("    outputs/PS_xxx/PS_xxx_rawLog.csv \\")
        print("    ../PaySim/outputs/PS_yyy/PS_yyy_rawLog.csv")
        return
    
    python_path = sys.argv[1]
    java_path = sys.argv[2]
    
    print("=" * 60)
    print("PaySim Python vs Java 对比分析")
    print("=" * 60)
    print(f"Python数据: {python_path}")
    print(f"Java数据: {java_path}")
    
    # 加载数据
    df_python, df_java = load_data(python_path, java_path)
    
    # 执行对比
    compare_transaction_types(df_python, df_java)
    compare_amounts(df_python, df_java)
    compare_fraud_rates(df_python, df_java)
    
    # 绘图
    plot_comparison(df_python, df_java)
    
    print("\n" + "=" * 60)
    print("对比分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
