"""
PaySim Python - 主运行脚本

运行方式:
    python run_paysim.py
    或
    python run_paysim.py --config custom_config.properties
"""
import argparse
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig
from simulator import PaySimPython


def main():
    parser = argparse.ArgumentParser(description='PaySim Python - Financial Transaction Simulator')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (.properties)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of simulation steps')
    parser.add_argument('--clients', type=int, default=None,
                        help='Number of clients')
    parser.add_argument('--fraudsters', type=int, default=None,
                        help='Number of fraudsters')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        config = SimulationConfig.from_properties_file(args.config)
    else:
        print("Using default configuration")
        config = SimulationConfig()
    
    # 命令行参数覆盖
    if args.steps:
        config.nb_steps = args.steps
    if args.clients:
        config.nb_clients = args.clients
    if args.fraudsters:
        config.nb_fraudsters = args.fraudsters
    if args.seed:
        config.seed = args.seed
    if args.output:
        config.output_path = args.output
    
    # 运行仿真
    simulator = PaySimPython(config)
    simulator.run()


if __name__ == '__main__':
    main()
