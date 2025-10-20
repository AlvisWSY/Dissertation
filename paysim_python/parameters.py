"""
PaySim Python 核心模块 - 参数加载器
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ParameterLoader:
    """从 paramFiles 加载仿真参数"""
    
    def __init__(self, param_dir: str):
        self.param_dir = param_dir
        self.transaction_types = None
        self.aggregated_transactions = None
        self.clients_profiles = None
        self.initial_balances_distribution = None
        self.overdraft_limits = None
        self.max_occurrences = None
        
        self._load_all()
    
    def _load_all(self):
        """加载所有参数文件"""
        try:
            self.transaction_types = pd.read_csv(
                os.path.join(self.param_dir, "transactionsTypes.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load transactionsTypes.csv: {e}")
            self.transaction_types = pd.DataFrame({
                'action': ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
            })
        
        try:
            self.aggregated_transactions = pd.read_csv(
                os.path.join(self.param_dir, "aggregatedTransactions.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load aggregatedTransactions.csv: {e}")
            self.aggregated_transactions = None
        
        try:
            self.clients_profiles = pd.read_csv(
                os.path.join(self.param_dir, "clientsProfiles.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load clientsProfiles.csv: {e}")
            self.clients_profiles = None
        
        try:
            self.initial_balances_distribution = pd.read_csv(
                os.path.join(self.param_dir, "initialBalancesDistribution.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load initialBalancesDistribution.csv: {e}")
            self.initial_balances_distribution = None
        
        try:
            self.overdraft_limits = pd.read_csv(
                os.path.join(self.param_dir, "overdraftLimits.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load overdraftLimits.csv: {e}")
            self.overdraft_limits = None
        
        try:
            self.max_occurrences = pd.read_csv(
                os.path.join(self.param_dir, "maxOccurrencesPerClient.csv")
            )
        except Exception as e:
            print(f"Warning: Could not load maxOccurrencesPerClient.csv: {e}")
            self.max_occurrences = None
    
    def get_action_types(self) -> List[str]:
        """获取交易类型列表"""
        if self.transaction_types is not None and 'action' in self.transaction_types.columns:
            return self.transaction_types['action'].tolist()
        return ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    
    def get_client_profiles(self, rng: np.random.Generator) -> List[Dict]:
        """获取客户配置文件列表"""
        profiles = []
        
        if self.clients_profiles is None:
            # 默认配置
            actions = self.get_action_types()
            n_profiles = 10
            for i in range(n_profiles):
                profile = {}
                probs = rng.dirichlet(np.ones(len(actions)))
                for action, prob in zip(actions, probs):
                    profile[action] = {
                        'probability': prob,
                        'avg_amount': float(rng.lognormal(5, 1)),
                        'std_amount': float(rng.lognormal(4, 0.5))
                    }
                profiles.append(profile)
        else:
            # 从文件加载
            df = self.clients_profiles
            grouped = df.groupby('clientId') if 'clientId' in df.columns else df.groupby(df.index // 5)
            
            for group_id, group in grouped:
                profile = {}
                for _, row in group.iterrows():
                    action = row['action'] if 'action' in row else 'PAYMENT'
                    profile[action] = {
                        'probability': float(row.get('freq', 0.2)),
                        'avg_amount': float(row.get('averageAmount', 100)),
                        'std_amount': float(row.get('stdAmount', 50))
                    }
                profiles.append(profile)
        
        return profiles
    
    def get_initial_balance_distribution(self) -> List[Tuple[float, float, float]]:
        """获取初始余额分布 (range_start, range_end, percentage)"""
        if self.initial_balances_distribution is None:
            # 默认分布
            return [
                (0, 1000, 0.3),
                (1000, 5000, 0.4),
                (5000, 50000, 0.2),
                (50000, 500000, 0.1),
            ]
        
        df = self.initial_balances_distribution
        ranges = []
        for _, row in df.iterrows():
            range_start = float(row.get('range_start', 0))
            range_end = float(row.get('range_end', 1000))
            percentage = float(row.get('percentage', 0.25))
            ranges.append((range_start, range_end, percentage))
        
        return ranges
    
    def sample_initial_balance(self, rng: np.random.Generator) -> float:
        """采样初始余额"""
        ranges = self.get_initial_balance_distribution()
        
        # 归一化百分比
        total_pct = sum(r[2] for r in ranges)
        if total_pct <= 0:
            return float(rng.uniform(0, 10000))
        
        probs = [r[2] / total_pct for r in ranges]
        idx = rng.choice(len(ranges), p=probs)
        range_start, range_end, _ = ranges[idx]
        
        return float(rng.uniform(range_start, range_end))
    
    def get_overdraft_limit(self, balance: float) -> float:
        """根据余额获取透支限额"""
        if self.overdraft_limits is None:
            # 默认规则：余额的 -10%
            return -0.1 * balance
        
        df = self.overdraft_limits
        for _, row in df.iterrows():
            lower = row.get('lowerbound', float('-inf'))
            upper = row.get('higherbound', float('inf'))
            if pd.isna(lower):
                lower = float('-inf')
            if pd.isna(upper):
                upper = float('inf')
            
            if lower <= balance <= upper:
                return float(row.get('overdraftLimit', 0))
        
        return 0.0
    
    def get_step_profiles(self) -> Dict[int, Dict[str, dict]]:
        """获取每个时间步的交易配置"""
        if self.aggregated_transactions is None:
            return {}
        
        df = self.aggregated_transactions
        step_profiles = {}
        
        for step in df['step'].unique():
            step_data = df[df['step'] == step]
            action_profiles = {}
            
            for _, row in step_data.iterrows():
                action = row.get('action', 'PAYMENT')
                action_profiles[action] = {
                    'count': int(row.get('count', 0)),
                    'avg_amount': float(row.get('average', 100)),
                    'std_amount': float(row.get('std', 50))
                }
            
            step_profiles[int(step)] = action_profiles
        
        return step_profiles
    
    def get_step_action_probabilities(self, step: int) -> Dict[str, float]:
        """获取指定时间步的交易类型概率分布"""
        if self.aggregated_transactions is None:
            actions = self.get_action_types()
            return {a: 1.0 / len(actions) for a in actions}
        
        df = self.aggregated_transactions
        step_data = df[df['step'] == step]
        
        if len(step_data) == 0:
            actions = self.get_action_types()
            return {a: 1.0 / len(actions) for a in actions}
        
        total_count = step_data['count'].sum()
        if total_count == 0:
            actions = self.get_action_types()
            return {a: 1.0 / len(actions) for a in actions}
        
        probs = {}
        for _, row in step_data.iterrows():
            action = row.get('action', 'PAYMENT')
            count = row.get('count', 0)
            probs[action] = count / total_count
        
        return probs
    
    def get_step_target_count(self, step: int) -> int:
        """获取指定时间步的目标交易总数"""
        if self.aggregated_transactions is None:
            return 1000
        
        df = self.aggregated_transactions
        step_data = df[df['step'] == step]
        
        if len(step_data) == 0:
            return 1000
        
        return int(step_data['count'].sum())
