"""
PaySim Python 配置模块
从配置文件或环境变量加载参数
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """仿真配置参数"""
    seed: Optional[int] = None  # None 表示使用时间戳
    nb_steps: int = 720
    multiplier: float = 1.0
    nb_clients: int = 20000
    nb_fraudsters: int = 1000
    nb_merchants: int = 34749
    nb_banks: int = 5
    fraud_probability: float = 0.001
    transfer_limit: float = 20000000000.0
    
    # 文件路径
    param_dir: str = "../PaySim/paramFiles"
    output_path: str = "./outputs"
    
    # 数据库配置（可选）
    save_to_db: bool = False
    db_url: str = "jdbc:mysql://localhost:3306/paysim"
    db_user: str = "none"
    db_password: str = "none"
    
    @classmethod
    def from_properties_file(cls, filepath: str) -> 'SimulationConfig':
        """从 .properties 文件加载配置"""
        config = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        
        return cls(
            seed=None if config.get('seed', 'time') == 'time' else int(config.get('seed', 0)),
            nb_steps=int(config.get('nbSteps', 720)),
            multiplier=float(config.get('multiplier', 1.0)),
            nb_clients=int(config.get('nbClients', 20000)),
            nb_fraudsters=int(config.get('nbFraudsters', 1000)),
            nb_merchants=int(config.get('nbMerchants', 34749)),
            nb_banks=int(config.get('nbBanks', 5)),
            fraud_probability=float(config.get('fraudProbability', 0.001)),
            transfer_limit=float(config.get('transferLimit', 20000000000)),
            param_dir=config.get('transactionsTypes', './paramFiles').replace('./paramFiles/transactionsTypes.csv', './paramFiles'),
            output_path=config.get('outputPath', './outputs'),
            save_to_db=bool(int(config.get('saveToDB', 0))),
        )
    
    def get_scaled_value(self, base_value: int) -> int:
        """根据 multiplier 缩放值"""
        return int(base_value * self.multiplier)
