"""
最小化测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig
from simulator import PaySimPython


def main():
    print("PaySim Python - 最小化测试")
    print("=" * 50)
    
    # 超小规模配置
    config = SimulationConfig(
        seed=42,
        nb_steps=10,           # 仅10步
        nb_clients=50,         # 50个客户
        nb_fraudsters=5,       # 5个欺诈者
        nb_merchants=10,       # 10个商户
        nb_banks=2,            # 2个银行
        fraud_probability=0.1,
        multiplier=1.0,
        param_dir="../PaySim/paramFiles",
        output_path="./outputs"
    )
    
    print(f"\n配置: {config.nb_steps} 步, {config.nb_clients} 客户")
    
    # 运行仿真
    simulator = PaySimPython(config)
    simulator.run()
    
    print("\n测试完成！")
    print(f"生成交易数: {len(simulator.transactions)}")
    print(f"欺诈交易数: {sum(1 for tx in simulator.transactions if tx.is_fraud)}")


if __name__ == '__main__':
    main()
