"""
示例脚本：快速生成小规模数据集用于测试
"""
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入模块
from config import SimulationConfig
from simulator import PaySimPython


def main():
    print("=" * 60)
    print("PaySim Python - 快速测试示例")
    print("=" * 60)
    
    # 创建一个小规模配置用于快速测试
    config = SimulationConfig(
        seed=42,
        nb_steps=100,          # 100个时间步（约4天）
        nb_clients=1000,       # 1000个客户
        nb_fraudsters=50,      # 50个欺诈者
        nb_merchants=200,      # 200个商户
        nb_banks=3,            # 3个银行
        fraud_probability=0.01,
        multiplier=1.0,
        param_dir="../PaySim/paramFiles",
        output_path="./outputs"
    )
    
    print("\n配置参数:")
    print(f"  步数: {config.nb_steps}")
    print(f"  客户: {config.nb_clients}")
    print(f"  欺诈者: {config.nb_fraudsters}")
    print(f"  商户: {config.nb_merchants}")
    print(f"  银行: {config.nb_banks}")
    print(f"  随机种子: {config.seed}")
    
    # 运行仿真
    simulator = PaySimPython(config)
    simulator.run()
    
    print("\n" + "=" * 60)
    print("测试完成！查看 outputs/ 目录获取结果")
    print("=" * 60)


if __name__ == '__main__':
    main()
