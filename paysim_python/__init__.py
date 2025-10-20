"""
PaySim Python - 金融交易仿真器

这是 PaySim (Java) 的 Python 重构版本，用于生成移动支付交易数据集。

主要模块:
- config: 配置管理
- parameters: 参数加载
- actors: 参与者类（客户、商户、银行、欺诈者）
- transaction: 交易数据结构
- simulator: 主仿真引擎

使用示例:
    from paysim_python import PaySimPython, SimulationConfig
    
    config = SimulationConfig(
        nb_steps=720,
        nb_clients=10000,
        nb_fraudsters=500
    )
    
    sim = PaySimPython(config)
    sim.run()
"""

from config import SimulationConfig
from simulator import PaySimPython
from transaction import Transaction
from actors import Bank, Merchant, Client, Fraudster

__version__ = "2.0.0"
__all__ = [
    'SimulationConfig',
    'PaySimPython',
    'Transaction',
    'Bank',
    'Merchant',
    'Client',
    'Fraudster',
]
