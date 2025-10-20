# PaySim Python

这是 PaySim (Java版本) 的 Python 完整重构版本，用于生成逼真的移动支付金融交易数据集，包含正常交易和欺诈交易。

## 项目结构

```
paysim_python/
├── __init__.py          # 包初始化
├── config.py            # 配置管理模块
├── parameters.py        # 参数加载模块
├── actors.py            # 参与者类（客户、商户、银行、欺诈者）
├── transaction.py       # 交易数据结构
├── simulator.py         # 主仿真引擎
├── run_paysim.py        # 主运行脚本
└── README.md            # 本文件
```

## 主要特性

### ✅ 已实现的核心功能

1. **多类型参与者**
   - 普通客户 (Client)
   - 欺诈者 (Fraudster)
   - 商户 (Merchant)
   - 银行 (Bank)

2. **五种交易类型**
   - CASH_IN: 现金流入
   - CASH_OUT: 现金流出
   - TRANSFER: 客户间转账
   - PAYMENT: 客户向商户付款
   - DEBIT: 客户与银行交易

3. **智能行为模拟**
   - 基于客户配置文件的个性化交易行为
   - 弹簧模型平衡账户余额（避免余额偏离过大）
   - 二项分布控制交易频率
   - 正态分布采样交易金额

4. **欺诈检测特性**
   - 欺诈交易标记 (isFraud)
   - 系统标记 (isFlaggedFraud)
   - 未授权透支检测

5. **参数化配置**
   - 从 PaySim.properties 加载配置
   - 从 paramFiles 目录加载参数文件
   - 支持命令行参数覆盖

## 安装依赖

```bash
pip install numpy pandas
```

## 使用方法

### 方法1: 使用默认配置运行

```bash
cd /usr1/home/s124mdg53_07/wang/FYP/paysim_python
python run_paysim.py
```

### 方法2: 指定配置文件

```bash
python run_paysim.py --config ../PaySim/PaySim.properties
```

### 方法3: 命令行参数

```bash
python run_paysim.py --steps 100 --clients 5000 --fraudsters 200 --seed 42
```

### 方法4: Python 代码调用

```python
from paysim_python import PaySimPython, SimulationConfig

# 创建配置
config = SimulationConfig(
    seed=42,
    nb_steps=720,           # 仿真步数（例如720=30天*24小时）
    nb_clients=10000,       # 客户数量
    nb_fraudsters=500,      # 欺诈者数量
    nb_merchants=5000,      # 商户数量
    nb_banks=5,             # 银行数量
    fraud_probability=0.001, # 欺诈概率
    param_dir="../PaySim/paramFiles",  # 参数文件目录
    output_path="./outputs"            # 输出目录
)

# 运行仿真
simulator = PaySimPython(config)
simulator.run()
```

## 输出结果

仿真完成后，会在 `outputs/` 目录下生成以下文件：

```
outputs/
└── PS_<timestamp>_<seed>/
    ├── PS_<timestamp>_<seed>_rawLog.csv      # 原始交易日志
    ├── PS_<timestamp>_<seed>_Summary.txt     # 仿真摘要
    └── PS_<timestamp>_<seed>_config.txt      # 运行配置
```

### 输出CSV格式

rawLog.csv 包含以下列：

| 列名 | 说明 |
|------|------|
| step | 时间步 |
| type | 交易类型 |
| amount | 交易金额 |
| nameOrig | 源账户名称 |
| oldbalanceOrg | 源账户旧余额 |
| newbalanceOrig | 源账户新余额 |
| nameDest | 目标账户名称 |
| oldbalanceDest | 目标账户旧余额 |
| newbalanceDest | 目标账户新余额 |
| isFraud | 是否欺诈(0/1) |
| isFlaggedFraud | 是否被系统标记(0/1) |

## 与 Java 版本的对比

### 优势

1. **更简洁**: Python 代码量约为 Java 版本的 1/3
2. **易维护**: 面向对象设计，模块清晰
3. **易扩展**: 可快速添加新的交易类型或欺诈模式
4. **数据分析友好**: 直接输出 Pandas DataFrame，便于后续分析
5. **无需编译**: 即改即用，调试方便

### 保持一致

1. **相同的参数文件**: 兼容原 PaySim 的 paramFiles
2. **相同的输出格式**: CSV 格式与原版一致
3. **相同的业务逻辑**: 交易类型、余额管理、欺诈标记
4. **相同的随机机制**: 使用相同的统计分布

## 高级用法

### 自定义欺诈策略

```python
from paysim_python.actors import Fraudster

class CustomFraudster(Fraudster):
    def custom_fraud_behavior(self, step, amount):
        # 实现自定义欺诈逻辑
        pass
```

### 批量运行

```python
for seed in range(10):
    config = SimulationConfig(seed=seed)
    sim = PaySimPython(config)
    sim.run()
```

## 性能

在典型配置下（10000 客户，720 步）：
- 运行时间: ~2-5 分钟
- 内存占用: ~500MB
- 生成交易: ~100,000 条

## 参数文件说明

PaySim 依赖以下参数文件（位于 `paramFiles/` 目录）：

1. **transactionsTypes.csv**: 交易类型定义
2. **clientsProfiles.csv**: 客户行为配置文件
3. **aggregatedTransactions.csv**: 历史交易聚合数据
4. **initialBalancesDistribution.csv**: 初始余额分布
5. **overdraftLimits.csv**: 透支限额规则
6. **maxOccurrencesPerClient.csv**: 客户最大交易次数

## 故障排除

### 问题: 无法加载参数文件
```
Warning: Could not load xxx.csv
```
**解决**: 检查 `param_dir` 配置是否正确指向 PaySim/paramFiles 目录

### 问题: 交易数量太少
**解决**: 增加 `nb_steps` 或 `nb_clients` 参数

### 问题: 欺诈率太低
**解决**: 增加 `fraud_probability` 或 `nb_fraudsters` 参数

## 开发计划

- [ ] 添加更复杂的欺诈模式（洗钱网络、木马账户等）
- [ ] 支持实时可视化
- [ ] 支持数据库输出
- [ ] 添加更多统计分析工具
- [ ] 性能优化（多进程/GPU加速）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

与原 PaySim 项目保持一致

## 致谢

基于 EURECOM 的 PaySim 项目：
- 原始论文: https://www.researchgate.net/publication/313138956
- GitHub: https://github.com/EdgarLopezPhD/PaySim

## 联系方式

如有问题，请联系项目维护者。
