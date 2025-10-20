"""
PaySim Python 核心模块 - 参与者（Actors）
"""
from typing import Optional
import numpy as np


class SuperActor:
    """所有参与者的基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.balance = 0.0
        self.overdraft_limit = 0.0
        self.is_fraud = False
    
    def deposit(self, amount: float):
        """存入资金"""
        self.balance += amount
    
    def withdraw(self, amount: float) -> bool:
        """取出资金，返回 True 表示发生未授权透支"""
        unauthorized_overdraft = False
        
        if self.balance - amount < self.overdraft_limit:
            unauthorized_overdraft = True
        else:
            self.balance -= amount
        
        return unauthorized_overdraft
    
    def get_balance(self) -> float:
        return self.balance
    
    def __str__(self) -> str:
        return self.name


class Bank(SuperActor):
    """银行"""
    
    def __init__(self, name: str):
        super().__init__(f"B{name}")
        self.balance = 1e10  # 银行初始余额


class Merchant(SuperActor):
    """商户"""
    
    def __init__(self, name: str, initial_balance: float = 0):
        super().__init__(f"M{name}")
        self.balance = initial_balance


class Client(SuperActor):
    """普通客户"""
    
    # 交易类型常量
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    DEPOSIT = "DEPOSIT"
    
    MIN_NB_TRANSFER_FOR_FRAUD = 3
    
    def __init__(self, 
                 name: str, 
                 bank: Bank,
                 initial_balance: float,
                 overdraft_limit: float,
                 client_profile: dict,
                 client_weight: float):
        super().__init__(f"C{name}")
        self.bank = bank
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.overdraft_limit = overdraft_limit
        self.client_profile = client_profile
        self.client_weight = client_weight
        
        self.balance_max = initial_balance
        self.count_transfer_transactions = 0
        self.expected_avg_transaction = 0.0
        
        # 计算期望平均交易额
        total_prob = 0.0
        weighted_amount = 0.0
        for action, prob in client_profile.items():
            if 'prob' in action:
                continue
            action_data = client_profile.get(action, {})
            if isinstance(action_data, dict):
                prob_val = action_data.get('probability', 0)
                avg_amt = action_data.get('avg_amount', 0)
                total_prob += prob_val
                weighted_amount += prob_val * avg_amt
        
        if total_prob > 0:
            self.expected_avg_transaction = weighted_amount / total_prob
    
    def pick_count(self, rng: np.random.Generator, target_step_count: int) -> int:
        """使用二项分布选择交易次数"""
        # B(n,p): n = targetStepCount & p = clientWeight
        return rng.binomial(target_step_count, min(self.client_weight, 1.0))
    
    def pick_action(self, rng: np.random.Generator, step_action_probs: dict) -> str:
        """选择交易类型"""
        client_probs = self.client_profile.copy()
        raw_probs = {}
        
        # 结合客户配置和步骤配置
        for action, client_prob in client_probs.items():
            if action.endswith('_prob'):
                continue
            if isinstance(client_prob, dict):
                client_prob = client_prob.get('probability', 0)
            
            if action in step_action_probs:
                step_prob = step_action_probs[action]
                raw_prob = (client_prob + step_prob) / 2
            else:
                raw_prob = client_prob
            raw_probs[action] = raw_prob
        
        # 根据余额调整流入流出概率（弹簧模型）
        prob_inflow = sum(raw_probs.get(a, 0) for a in [self.CASH_IN, self.DEPOSIT])
        prob_outflow = 1 - prob_inflow
        
        new_prob_inflow = self._compute_prob_with_spring(
            prob_inflow, prob_outflow, self.balance
        )
        new_prob_outflow = 1 - new_prob_inflow
        
        # 调整最终概率
        final_probs = {}
        for action, raw_prob in raw_probs.items():
            if self._is_inflow(action):
                if prob_inflow > 0:
                    final_probs[action] = raw_prob * new_prob_inflow / prob_inflow
                else:
                    final_probs[action] = raw_prob
            else:
                if prob_outflow > 0:
                    final_probs[action] = raw_prob * new_prob_outflow / prob_outflow
                else:
                    final_probs[action] = raw_prob
        
        # 归一化并选择
        total = sum(final_probs.values())
        if total > 0:
            probs = np.array([final_probs[a] for a in final_probs.keys()])
            probs = probs / probs.sum()
            return rng.choice(list(final_probs.keys()), p=probs)
        else:
            return rng.choice([self.PAYMENT, self.TRANSFER, self.CASH_OUT])
    
    def _is_inflow(self, action: str) -> bool:
        """判断是否为流入类型交易"""
        return action in [self.CASH_IN, self.DEPOSIT]
    
    def _compute_prob_with_spring(self, prob_up: float, prob_down: float, 
                                   current_balance: float) -> float:
        """使用弹簧模型计算调整后的概率"""
        equilibrium = 40 * self.expected_avg_transaction
        if equilibrium == 0:
            equilibrium = self.initial_balance
        
        correction_strength = 3e-5
        characteristic_length_spring = equilibrium
        k = 1 / characteristic_length_spring if characteristic_length_spring > 0 else 0
        spring_force = k * (equilibrium - current_balance)
        
        new_prob_up = 0.5 * (1.0 + (self.expected_avg_transaction * correction_strength) * 
                             spring_force + (prob_up - prob_down))
        
        return max(0.0, min(1.0, new_prob_up))
    
    def pick_amount(self, rng: np.random.Generator, action: str, 
                    step_profile: Optional[dict] = None) -> float:
        """选择交易金额"""
        # 优先使用步骤配置
        if step_profile and 'avg_amount' in step_profile:
            avg = step_profile['avg_amount']
            std = step_profile.get('std_amount', max(1.0, 0.5 * avg))
        elif action in self.client_profile and isinstance(self.client_profile[action], dict):
            profile = self.client_profile[action]
            avg = profile.get('avg_amount', 100)
            std = profile.get('std_amount', max(1.0, 0.5 * avg))
        else:
            avg = 100.0
            std = 50.0
        
        amount = abs(rng.normal(avg, std))
        return max(1.0, amount)


class Fraudster(Client):
    """欺诈者"""
    
    def __init__(self, 
                 name: str, 
                 bank: Bank,
                 initial_balance: float = 0,
                 overdraft_limit: float = 0):
        # 简化配置
        profile = {
            self.TRANSFER: {'probability': 0.6, 'avg_amount': 5000, 'std_amount': 2000},
            self.CASH_OUT: {'probability': 0.4, 'avg_amount': 3000, 'std_amount': 1000},
        }
        super().__init__(name, bank, initial_balance, overdraft_limit, profile, 0.01)
        self.is_fraud = True
        self.fraud_strategy_active = True
