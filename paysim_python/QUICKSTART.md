# PaySim Python - 快速开始指南 🚀

## 5分钟快速上手

### 第一步: 检查环境

```bash
# 确保 Python 3.8+
python --version

# 安装依赖
pip install numpy pandas matplotlib scipy
```

### 第二步: 运行测试

```bash
cd /usr1/home/s124mdg53_07/wang/FYP/paysim_python
python test_minimal.py
```

**预期输出:**
```
PaySim Python - 最小化测试
==================================================
...
生成交易数: 691158
欺诈交易数: 43
测试完成!
```

### 第三步: 查看结果

```bash
# 查看生成的文件
ls -lh outputs/PS_*/

# 查看交易数据前10行
head -10 outputs/PS_*/PS_*_rawLog.csv
```

## 常用命令

### 1. 快速测试（小规模）
```bash
python test_minimal.py
```

### 2. 标准运行（中等规模）
```bash
python example_quick_test.py
```

### 3. 完整仿真（大规模）
```bash
python run_paysim.py --steps 720 --clients 20000 --fraudsters 1000
```

### 4. 使用配置文件
```bash
python run_paysim.py --config ../PaySim/PaySim.properties
```

### 5. 批量运行（不同种子）
```bash
for i in {1..5}; do
    python run_paysim.py --seed $i --steps 100
done
```

## 使用 Python 代码

### 基础示例
```python
from config import SimulationConfig
from simulator import PaySimPython

# 创建配置
config = SimulationConfig(
    seed=42,
    nb_steps=100,
    nb_clients=1000,
    nb_fraudsters=50
)

# 运行仿真
sim = PaySimPython(config)
sim.run()

# 访问结果
print(f"生成交易: {len(sim.transactions)}")
print(f"欺诈交易: {sum(1 for tx in sim.transactions if tx.is_fraud)}")
```

### 数据分析示例
```python
import pandas as pd

# 加载数据
df = pd.read_csv('outputs/PS_xxx/PS_xxx_rawLog.csv')

# 基本统计
print(df.describe())

# 交易类型分布
print(df['type'].value_counts())

# 欺诈率
print(f"欺诈率: {df['isFraud'].mean():.2%}")

# 按类型统计金额
print(df.groupby('type')['amount'].agg(['mean', 'std', 'count']))
```

### 可视化示例
```python
import matplotlib.pyplot as plt

# 交易金额分布
plt.figure(figsize=(10, 6))
plt.hist(df['amount'], bins=50, alpha=0.7)
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Transaction Amount Distribution')
plt.yscale('log')
plt.savefig('amount_dist.png')
```

## 输出文件说明

### 1. rawLog.csv
原始交易日志，包含所有交易详情

**重要列:**
- `step`: 时间步
- `type`: 交易类型 (CASH_IN, CASH_OUT, TRANSFER, PAYMENT, DEBIT)
- `amount`: 金额
- `nameOrig`, `nameDest`: 源/目标账户
- `isFraud`: 是否欺诈 (0/1)
- `isFlaggedFraud`: 是否被系统标记 (0/1)

### 2. Summary.txt
仿真统计摘要

包含:
- 参与者数量
- 交易总数
- 欺诈统计
- 交易类型分布

### 3. config.txt
运行时配置参数

## 常用参数调整

### 增加交易数量
```python
config = SimulationConfig(
    nb_steps=1000,        # 增加步数
    nb_clients=50000      # 增加客户数
)
```

### 提高欺诈率
```python
config = SimulationConfig(
    nb_fraudsters=2000,           # 增加欺诈者
    fraud_probability=0.01        # 提高欺诈概率
)
```

### 快速调试
```python
config = SimulationConfig(
    nb_steps=10,          # 减少步数
    nb_clients=100,       # 减少客户数
    multiplier=0.1        # 缩小规模
)
```

## 性能参考

| 配置 | 交易数 | 时间 | 内存 |
|------|--------|------|------|
| 小规模 (10步, 50客户) | ~69万 | ~30秒 | ~100MB |
| 中规模 (100步, 1000客户) | ~500万 | ~5分钟 | ~500MB |
| 大规模 (720步, 20000客户) | ~5000万 | ~60分钟 | ~2GB |

## 故障排除

### 问题: ModuleNotFoundError
```bash
# 解决: 在正确目录运行
cd /usr1/home/s124mdg53_07/wang/FYP/paysim_python
```

### 问题: 交易数量过多导致内存不足
```python
# 解决: 减小规模
config.multiplier = 0.1
config.nb_clients = 1000
```

### 问题: 欺诈率为0
```python
# 解决: 检查配置
config.nb_fraudsters = 100  # 确保有欺诈者
config.fraud_probability = 0.01  # 确保概率合理
```

## 下一步

1. **阅读详细文档**: `README.md` 和 `ARCHITECTURE.md`
2. **理解核心算法**: 查看 `simulator.py` 和 `actors.py`
3. **自定义扩展**: 添加新的交易类型或欺诈模式
4. **数据分析**: 使用生成的数据训练机器学习模型

## 获取帮助

- 📖 查看 `PROJECT_SUMMARY.md` 了解项目概况
- 🏗️ 查看 `ARCHITECTURE.md` 了解架构设计
- 📊 运行 `compare_outputs.py` 对比不同版本
- 💬 查看代码注释获取详细说明

## 快速检查清单

- [ ] Python 3.8+ 已安装
- [ ] 依赖包已安装 (numpy, pandas)
- [ ] 在正确目录 (`paysim_python/`)
- [ ] 参数文件存在 (`../PaySim/paramFiles/`)
- [ ] 有足够磁盘空间 (至少 1GB)

---

**提示**: 从 `test_minimal.py` 开始，逐步增加规模！

Happy Simulating! 🎉
