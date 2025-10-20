"""
PaySim Python 核心模块 - 主仿真器
"""
import os
import time
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd

from config import SimulationConfig
from parameters import ParameterLoader
from actors import Bank, Merchant, Client, Fraudster
from transaction import Transaction


class PaySimPython:
    """PaySim 仿真器主类"""
    
    VERSION = "2.0-Python"
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 初始化随机数生成器
        if config.seed is None:
            self.seed = int(time.time() * 1000) % (2**32)
        else:
            self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)
        
        # 加载参数
        self.param_loader = ParameterLoader(config.param_dir)
        
        # 初始化参与者列表
        self.banks: List[Bank] = []
        self.merchants: List[Merchant] = []
        self.clients: List[Client] = []
        self.fraudsters: List[Fraudster] = []
        
        # 交易记录
        self.transactions: List[Transaction] = []
        
        # 仿真状态
        self.current_step = 0
        self.simulation_name = self._generate_simulation_name()
        self.output_dir = os.path.join(config.output_path, self.simulation_name)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"PaySim Python v{self.VERSION}")
        print(f"Simulation: {self.simulation_name}")
        print(f"Seed: {self.seed}")
    
    def _generate_simulation_name(self) -> str:
        """生成仿真名称"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"PS_{timestamp}_{self.seed}"
    
    def _init_actors(self):
        """初始化所有参与者"""
        print("\n=== Initializing Actors ===")
        
        # 创建银行
        n_banks = self.config.nb_banks
        print(f"Creating {n_banks} banks...")
        for i in range(n_banks):
            bank = Bank(str(i))
            self.banks.append(bank)
        
        # 创建商户
        n_merchants = self.config.get_scaled_value(self.config.nb_merchants)
        print(f"Creating {n_merchants} merchants...")
        for i in range(n_merchants):
            initial_balance = float(self.rng.lognormal(9.0, 1.0))
            merchant = Merchant(str(i), initial_balance)
            self.merchants.append(merchant)
        
        # 加载客户配置文件
        client_profiles = self.param_loader.get_client_profiles(self.rng)
        total_target_count = sum(
            self.param_loader.get_step_target_count(s) 
            for s in range(self.config.nb_steps)
        )
        
        # 创建普通客户
        n_clients = self.config.get_scaled_value(self.config.nb_clients)
        print(f"Creating {n_clients} clients...")
        for i in range(n_clients):
            bank = self.rng.choice(self.banks)
            initial_balance = self.param_loader.sample_initial_balance(self.rng)
            overdraft_limit = self.param_loader.get_overdraft_limit(initial_balance)
            
            # 随机选择客户配置
            profile = self.rng.choice(client_profiles)
            client_weight = 1.0 / n_clients if n_clients > 0 else 0
            
            client = Client(
                name=str(i),
                bank=bank,
                initial_balance=initial_balance,
                overdraft_limit=overdraft_limit,
                client_profile=profile,
                client_weight=client_weight
            )
            self.clients.append(client)
        
        # 创建欺诈者
        n_fraudsters = self.config.get_scaled_value(self.config.nb_fraudsters)
        print(f"Creating {n_fraudsters} fraudsters...")
        for i in range(n_fraudsters):
            bank = self.rng.choice(self.banks)
            initial_balance = self.param_loader.sample_initial_balance(self.rng)
            overdraft_limit = self.param_loader.get_overdraft_limit(initial_balance)
            
            fraudster = Fraudster(
                name=f"F{i}",
                bank=bank,
                initial_balance=initial_balance,
                overdraft_limit=overdraft_limit
            )
            self.fraudsters.append(fraudster)
        
        print(f"Total actors: {len(self.banks) + len(self.merchants) + len(self.clients) + len(self.fraudsters)}")
    
    def _pick_random_bank(self) -> Bank:
        """随机选择一个银行"""
        return self.rng.choice(self.banks) if self.banks else None
    
    def _pick_random_merchant(self) -> Merchant:
        """随机选择一个商户"""
        return self.rng.choice(self.merchants) if self.merchants else None
    
    def _pick_random_client(self, exclude: Client = None) -> Client:
        """随机选择一个客户（可排除指定客户）"""
        candidates = [c for c in self.clients if c != exclude]
        return self.rng.choice(candidates) if candidates else None
    
    def _make_transaction(self, step: int, orig: Client, dest, 
                          action: str, amount: float) -> Transaction:
        """执行一笔交易"""
        old_balance_orig = orig.balance
        old_balance_dest = dest.balance if dest else 0
        
        # 执行转账
        is_unauthorized = orig.withdraw(amount)
        if dest:
            dest.deposit(amount)
        
        new_balance_orig = orig.balance
        new_balance_dest = dest.balance if dest else old_balance_dest
        
        # 创建交易记录
        tx = Transaction(
            step=step,
            action=action,
            amount=amount,
            name_orig=orig.name,
            old_balance_orig=old_balance_orig,
            new_balance_orig=new_balance_orig,
            name_dest=dest.name if dest else "EXTERNAL",
            old_balance_dest=old_balance_dest,
            new_balance_dest=new_balance_dest,
            is_fraud=orig.is_fraud,
            is_flagged_fraud=False,
            is_unauthorized_overdraft=is_unauthorized
        )
        
        # 欺诈标记逻辑（简化版）
        if orig.is_fraud and action in [Client.TRANSFER, Client.CASH_OUT]:
            if amount > 10000:  # 大额交易更容易被标记
                if self.rng.random() < 0.5:
                    tx.is_flagged_fraud = True
        
        return tx
    
    def _step_clients(self, step: int):
        """执行一个时间步的所有客户交易"""
        step_target_count = self.param_loader.get_step_target_count(step)
        step_action_probs = self.param_loader.get_step_action_probabilities(step)
        
        for client in self.clients:
            count = client.pick_count(self.rng, step_target_count)
            
            for _ in range(count):
                action = client.pick_action(self.rng, step_action_probs)
                amount = client.pick_amount(self.rng, action)
                
                # 根据交易类型选择目标
                dest = None
                if action == Client.PAYMENT:
                    dest = self._pick_random_merchant()
                elif action == Client.TRANSFER:
                    dest = self._pick_random_client(exclude=client)
                elif action == Client.DEBIT:
                    dest = client.bank
                elif action == Client.CASH_IN:
                    # CASH_IN: 外部资金流入，客户余额增加
                    client.deposit(amount)
                    continue
                elif action == Client.CASH_OUT:
                    # CASH_OUT: 资金流出网络
                    dest = None
                
                if dest or action == Client.CASH_OUT:
                    tx = self._make_transaction(step, client, dest, action, amount)
                    self.transactions.append(tx)
    
    def _step_fraudsters(self, step: int):
        """执行一个时间步的所有欺诈者交易"""
        for fraudster in self.fraudsters:
            if not fraudster.fraud_strategy_active:
                continue
            
            # 欺诈者以更高频率发起交易
            if self.rng.random() < self.config.fraud_probability * 10:
                count = self.rng.integers(1, 5)
                
                for _ in range(count):
                    # 随机选择 TRANSFER 或 CASH_OUT
                    action = self.rng.choice([Client.TRANSFER, Client.CASH_OUT])
                    amount = float(self.rng.lognormal(8, 1))  # 较大金额
                    amount = min(amount, fraudster.balance * 0.9)  # 不要转空账户
                    
                    if amount < 1:
                        continue
                    
                    dest = None
                    if action == Client.TRANSFER:
                        # 转给随机客户（可能是同伙）
                        dest = self._pick_random_client()
                    else:
                        # CASH_OUT
                        dest = None
                    
                    if dest or action == Client.CASH_OUT:
                        tx = self._make_transaction(step, fraudster, dest, action, amount)
                        self.transactions.append(tx)
    
    def run(self):
        """运行仿真"""
        print(f"\n=== Starting Simulation for {self.config.nb_steps} steps ===")
        start_time = time.time()
        
        # 初始化参与者
        self._init_actors()
        
        # 主循环
        for step in range(self.config.nb_steps):
            self.current_step = step
            
            # 执行交易
            self._step_clients(step)
            self._step_fraudsters(step)
            
            # 进度显示
            if step % 100 == 99:
                print(f"Step {step + 1}/{self.config.nb_steps} - "
                      f"Transactions: {len(self.transactions)}")
        
        # 保存结果
        self._save_results()
        
        elapsed_time = (time.time() - start_time) / 60
        print(f"\n=== Simulation Complete ===")
        print(f"Total steps: {self.current_step + 1}")
        print(f"Total transactions: {len(self.transactions)}")
        print(f"Execution time: {elapsed_time:.2f} minutes")
        print(f"Output directory: {self.output_dir}")
    
    def _save_results(self):
        """保存仿真结果"""
        print("\n=== Saving Results ===")
        
        # 转换为 DataFrame
        df = pd.DataFrame([tx.to_dict() for tx in self.transactions])
        
        # 保存原始日志
        output_file = os.path.join(self.output_dir, f"{self.simulation_name}_rawLog.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved raw log: {output_file}")
        
        # 保存摘要
        summary_file = os.path.join(self.output_dir, f"{self.simulation_name}_Summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"PaySim Python Simulation Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Simulation Name: {self.simulation_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Steps: {self.config.nb_steps}\n")
            f.write(f"Clients: {len(self.clients)}\n")
            f.write(f"Fraudsters: {len(self.fraudsters)}\n")
            f.write(f"Merchants: {len(self.merchants)}\n")
            f.write(f"Banks: {len(self.banks)}\n\n")
            f.write(f"Total Transactions: {len(self.transactions)}\n")
            f.write(f"Fraud Transactions: {sum(1 for tx in self.transactions if tx.is_fraud)}\n")
            f.write(f"Flagged Fraud: {sum(1 for tx in self.transactions if tx.is_flagged_fraud)}\n\n")
            
            # 交易类型统计
            f.write("Transaction Type Distribution:\n")
            type_counts = df['type'].value_counts()
            for tx_type, count in type_counts.items():
                f.write(f"  {tx_type}: {count} ({count/len(df)*100:.2f}%)\n")
        
        print(f"Saved summary: {summary_file}")
        
        # 保存配置
        config_file = os.path.join(self.output_dir, f"{self.simulation_name}_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"seed={self.seed}\n")
            f.write(f"nbSteps={self.config.nb_steps}\n")
            f.write(f"multiplier={self.config.multiplier}\n")
            f.write(f"nbClients={self.config.nb_clients}\n")
            f.write(f"nbFraudsters={self.config.nb_fraudsters}\n")
            f.write(f"nbMerchants={self.config.nb_merchants}\n")
            f.write(f"nbBanks={self.config.nb_banks}\n")
        
        print(f"Saved config: {config_file}")
