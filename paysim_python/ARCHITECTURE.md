# PaySim Python 架构设计文档

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PaySim Python 仿真器                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌────────────────┐                         ┌────────────────┐
│  配置管理层      │                         │  参数加载层      │
│  (config.py)   │                         │(parameters.py) │
└────────────────┘                         └────────────────┘
        │                                           │
        │  SimulationConfig                         │  ParameterLoader
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   仿真引擎层      │
                    │  (simulator.py)  │
                    │   PaySimPython   │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌────────────┐      ┌────────────────┐      ┌──────────────┐
│  参与者层   │      │   交易数据层    │      │   输出层      │
│(actors.py) │      │(transaction.py)│      │  (CSV/TXT)   │
└────────────┘      └────────────────┘      └──────────────┘
```

## 类层次结构

### 参与者类 (actors.py)

```
SuperActor (基类)
│
├── 属性:
│   ├── name: str              # 参与者名称
│   ├── balance: float         # 账户余额
│   ├── overdraft_limit: float # 透支限额
│   └── is_fraud: bool         # 是否欺诈者
│
├── 方法:
│   ├── deposit(amount)        # 存入资金
│   ├── withdraw(amount)       # 取出资金
│   └── get_balance()          # 获取余额
│
├─── Bank (银行)
│    └── 初始余额: 100亿
│
├─── Merchant (商户)
│    └── 初始余额: 对数正态分布
│
└─── Client (客户)
     │
     ├── 属性:
     │   ├── bank: Bank                    # 关联银行
     │   ├── client_profile: dict          # 行为配置
     │   ├── client_weight: float          # 交易权重
     │   ├── expected_avg_transaction      # 期望交易额
     │   └── count_transfer_transactions   # 转账次数
     │
     ├── 方法:
     │   ├── pick_count()                  # 选择交易次数
     │   ├── pick_action()                 # 选择交易类型
     │   ├── pick_amount()                 # 选择交易金额
     │   └── _compute_prob_with_spring()   # 弹簧模型
     │
     └─── Fraudster (欺诈者)
          └── fraud_strategy_active: bool
```

### 交易数据结构 (transaction.py)

```
Transaction
│
├── 时间信息:
│   └── step: int                    # 时间步
│
├── 交易信息:
│   ├── action: str                  # 交易类型
│   └── amount: float                # 交易金额
│
├── 源账户:
│   ├── name_orig: str               # 源账户名
│   ├── old_balance_orig: float      # 旧余额
│   └── new_balance_orig: float      # 新余额
│
├── 目标账户:
│   ├── name_dest: str               # 目标账户名
│   ├── old_balance_dest: float      # 旧余额
│   └── new_balance_dest: float      # 新余额
│
└── 标记信息:
    ├── is_fraud: bool               # 是否欺诈
    ├── is_flagged_fraud: bool       # 是否被标记
    └── is_unauthorized_overdraft    # 是否未授权透支
```

## 数据流图

```
                    ┌──────────────┐
                    │ PaySim.properties│
                    │  配置文件     │
                    └───────┬──────┘
                            │
                            ▼
                    ┌──────────────┐
                    │SimulationConfig│
                    │  加载配置     │
                    └───────┬──────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌────────────────┐                    ┌─────────────────┐
│ paramFiles/    │                    │ PaySimPython    │
│ ├─ clientsProfiles.csv              │ 初始化仿真器     │
│ ├─ aggregatedTransactions.csv  ◄───│                 │
│ ├─ initialBalances...csv           └─────────┬───────┘
│ └─ ...                                       │
└────────────────┘                             │
                                               ▼
                                    ┌──────────────────┐
                                    │  初始化参与者     │
                                    │  ├─ Banks        │
                                    │  ├─ Merchants    │
                                    │  ├─ Clients      │
                                    │  └─ Fraudsters   │
                                    └─────────┬────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │        主仿真循环 (每步)            │
                            └─────────────────┬─────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
            ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
            │ 客户交易循环   │        │ 欺诈者交易循环 │        │ 收集交易记录   │
            │               │        │               │        │               │
            │ 对每个客户:    │        │ 对每个欺诈者:  │        │ transactions  │
            │ 1.选择次数    │        │ 1.激活策略    │        │ .append(tx)   │
            │ 2.选择类型    │        │ 2.大额转账    │        │               │
            │ 3.选择金额    │        │ 3.快速套现    │        │               │
            │ 4.执行交易    │        │ 4.执行交易    │        │               │
            └───────────────┘        └───────────────┘        └───────┬───────┘
                    │                         │                        │
                    └─────────────────────────┼────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  输出结果         │
                                    │  ├─ rawLog.csv   │
                                    │  ├─ Summary.txt  │
                                    │  └─ config.txt   │
                                    └──────────────────┘
```

## 核心算法流程

### 1. 客户交易决策流程

```
┌─────────────────┐
│ 开始新的时间步   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ 1. 计算本步目标交易总数      │
│    target_count = sum(agg)  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2. 为当前客户采样交易次数    │
│    count ~ B(target, weight)│
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 对每笔交易:                  │
└────────┬────────────────────┘
         │
         ├──► ┌──────────────────────┐
         │    │ 3a. 选择交易类型      │
         │    │ - 结合客户/步骤配置   │
         │    │ - 应用弹簧模型调整    │
         │    └──────────┬───────────┘
         │               │
         ├──► ┌──────────▼───────────┐
         │    │ 3b. 选择交易金额      │
         │    │ amount ~ N(μ, σ²)    │
         │    └──────────┬───────────┘
         │               │
         └──► ┌──────────▼───────────┐
              │ 3c. 执行交易          │
              │ - 选择目标            │
              │ - 更新余额            │
              │ - 记录交易            │
              └──────────────────────┘
```

### 2. 弹簧模型算法

```python
# 目标: 防止账户余额过度偏离平衡点

输入: 
  - prob_inflow: 流入概率
  - prob_outflow: 流出概率  
  - current_balance: 当前余额

equilibrium = 40 * expected_avg_transaction
k = 1 / equilibrium  # 弹簧系数
spring_force = k * (equilibrium - current_balance)

# 调整后的流入概率
new_prob_inflow = 0.5 * (
    1.0 
    + correction_strength * spring_force 
    + (prob_inflow - prob_outflow)
)

# 限制在 [0, 1] 范围内
new_prob_inflow = max(0, min(1, new_prob_inflow))

输出: new_prob_inflow
```

**物理解释**:
- 当余额 < 平衡点: spring_force > 0 → 增加流入概率
- 当余额 > 平衡点: spring_force < 0 → 减少流入概率
- 类似弹簧恢复力，使余额趋向平衡

### 3. 欺诈检测逻辑

```
┌────────────────────┐
│ 检查交易类型        │
└─────────┬──────────┘
          │
          ▼
    ┌─────────────┐
    │ TRANSFER or │  No
    │ CASH_OUT?   ├────► 标记为正常交易
    └─────┬───────┘
          │ Yes
          ▼
    ┌─────────────┐
    │ 发起者是     │  No
    │ 欺诈者?     ├────► 标记为正常交易
    └─────┬───────┘
          │ Yes
          ▼
    ┌─────────────┐
    │ isFraud = 1 │
    └─────┬───────┘
          │
          ▼
    ┌─────────────┐
    │ 金额 > 10K? │  No
    └─────┬───────┘       ├────► isFlaggedFraud = 0
          │ Yes           │
          ▼               │
    ┌─────────────┐       │
    │ 随机 < 0.5? │       │
    └─────┬───────┘       │
          │ Yes           │
          ▼               │
    ┌─────────────┐       │
    │ isFlaggedFraud=1│◄──┘
    └─────────────┘
```

## 配置参数说明

### SimulationConfig 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| seed | int | None | 随机种子 (None=时间戳) |
| nb_steps | int | 720 | 仿真步数 (例如: 30天×24小时) |
| multiplier | float | 1.0 | 规模缩放因子 |
| nb_clients | int | 20000 | 普通客户数量 |
| nb_fraudsters | int | 1000 | 欺诈者数量 |
| nb_merchants | int | 34749 | 商户数量 |
| nb_banks | int | 5 | 银行数量 |
| fraud_probability | float | 0.001 | 欺诈概率 |
| transfer_limit | float | 2e10 | 转账限额 |
| param_dir | str | ./paramFiles | 参数文件目录 |
| output_path | str | ./outputs | 输出目录 |

### 参数文件格式

#### 1. clientsProfiles.csv
```csv
clientId,action,freq,averageAmount,stdAmount
1,CASH_IN,0.1,5000,1000
1,PAYMENT,0.5,200,50
1,TRANSFER,0.3,1000,500
...
```

#### 2. aggregatedTransactions.csv
```csv
step,action,count,average,std
0,PAYMENT,150,200,50
0,TRANSFER,80,1000,300
1,CASH_OUT,50,500,100
...
```

#### 3. initialBalancesDistribution.csv
```csv
range_start,range_end,percentage
0,1000,0.3
1000,5000,0.4
5000,50000,0.2
50000,500000,0.1
```

## 性能优化策略

### 1. 向量化操作
```python
# ✗ 慢: 循环
for i in range(n):
    result[i] = func(data[i])

# ✓ 快: NumPy 向量化
result = np.vectorize(func)(data)
```

### 2. 预分配内存
```python
# ✓ 预分配列表大小
transactions = [None] * expected_count
```

### 3. 使用 Generator
```python
# ✓ 节省内存
def generate_transactions():
    for i in range(n):
        yield create_transaction(i)
```

### 4. 缓存计算结果
```python
# ✓ 缓存概率分布
@lru_cache(maxsize=128)
def get_step_probabilities(step):
    return compute_probabilities(step)
```

## 扩展开发指南

### 添加新的交易类型

1. 在 `Client` 类中添加常量:
```python
class Client:
    NEW_TYPE = "NEW_TYPE"
```

2. 在 `_step_clients` 中添加处理逻辑:
```python
elif action == Client.NEW_TYPE:
    dest = self._pick_custom_dest()
    # ... 处理逻辑
```

3. 更新参数文件:
```csv
# transactionsTypes.csv
action
...
NEW_TYPE
```

### 添加新的欺诈模式

1. 继承 `Fraudster` 类:
```python
class AdvancedFraudster(Fraudster):
    def custom_fraud_strategy(self):
        # 实现复杂欺诈逻辑
        pass
```

2. 在仿真器中使用:
```python
fraudster = AdvancedFraudster(...)
```

### 添加自定义输出

1. 在 `_save_results` 中添加:
```python
def _save_results(self):
    # ... 现有代码
    
    # 添加自定义输出
    custom_df = self._generate_custom_report()
    custom_df.to_csv(f"{self.output_dir}/custom_report.csv")
```

## 常见问题排查

### 问题 1: 交易数量异常
**症状**: 生成的交易数远超预期  
**原因**: `multiplier` 设置过大或 `client_weight` 计算错误  
**解决**: 检查配置参数，确保 `multiplier=1.0`

### 问题 2: 欺诈率过低
**症状**: `isFraud=1` 的交易占比很小  
**原因**: `fraud_probability` 或 `nb_fraudsters` 设置过小  
**解决**: 增加 `fraud_probability` 至 0.01 或增加欺诈者数量

### 问题 3: 余额异常
**症状**: 账户余额出现大幅负值  
**原因**: 透支限额未正确设置  
**解决**: 检查 `overdraftLimits.csv` 配置

### 问题 4: 内存不足
**症状**: 程序崩溃或运行缓慢  
**原因**: 一次性生成过多交易  
**解决**: 减少 `nb_steps` 或 `nb_clients`，或分批处理

## 开发工具推荐

### 调试工具
- **VS Code** + Python 扩展
- **PyCharm** Professional
- **Jupyter Notebook** (用于交互式探索)

### 性能分析
```python
# 使用 cProfile
python -m cProfile -o output.prof test_minimal.py

# 使用 line_profiler
@profile
def expensive_function():
    pass
```

### 代码质量
```bash
# 类型检查
mypy paysim_python/

# 代码风格
pylint paysim_python/
black paysim_python/

# 单元测试
pytest tests/
```

## 相关论文与资源

1. **PaySim原始论文** (2016)
   - "PaySim: A financial mobile money simulator for fraud detection"
   - Edgar Alonso Lopez-Rojas et al.

2. **欺诈检测综述**
   - "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy"
   - IEEE Transactions on Neural Networks and Learning Systems

3. **多智能体仿真**
   - MASON框架文档
   - NetLogo 案例研究

---

**文档版本**: 1.0  
**最后更新**: 2025年10月20日  
**维护者**: PaySim Python Team
