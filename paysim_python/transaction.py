"""
PaySim Python 核心模块 - 交易基础类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Transaction:
    """交易记录"""
    step: int
    action: str  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER, DEPOSIT
    amount: float
    name_orig: str
    old_balance_orig: float
    new_balance_orig: float
    name_dest: str
    old_balance_dest: float
    new_balance_dest: float
    is_fraud: bool = False
    is_flagged_fraud: bool = False
    is_unauthorized_overdraft: bool = False
    
    def is_failed_transaction(self) -> bool:
        """判断交易是否失败"""
        return self.is_flagged_fraud or self.is_unauthorized_overdraft
    
    def to_dict(self) -> dict:
        """转换为字典格式用于输出"""
        return {
            'step': self.step,
            'type': self.action,
            'amount': round(self.amount, 2),
            'nameOrig': self.name_orig,
            'oldbalanceOrg': round(self.old_balance_orig, 2),
            'newbalanceOrig': round(self.new_balance_orig, 2),
            'nameDest': self.name_dest,
            'oldbalanceDest': round(self.old_balance_dest, 2),
            'newbalanceDest': round(self.new_balance_dest, 2),
            'isFraud': int(self.is_fraud),
            'isFlaggedFraud': int(self.is_flagged_fraud),
        }
    
    def __str__(self) -> str:
        return f"{self.step},{self.action},{self.amount:.2f},{self.name_orig}," \
               f"{self.old_balance_orig:.2f},{self.new_balance_orig:.2f}," \
               f"{self.name_dest},{self.old_balance_dest:.2f},{self.new_balance_dest:.2f}," \
               f"{int(self.is_fraud)},{int(self.is_flagged_fraud)}"


@dataclass
class ClientActionProfile:
    """客户行为配置"""
    action: str
    probability: float
    avg_amount: float
    std_amount: float
    
    
@dataclass
class StepActionProfile:
    """时间步行为配置"""
    step: int
    action: str
    count: int
    avg_amount: float
    std_amount: float
