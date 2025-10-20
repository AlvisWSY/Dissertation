# PaySim Python 项目文件索引

## 📁 项目结构

```
paysim_python/
├── 核心代码模块 (Python)
│   ├── __init__.py                   (892 字节)   - 包初始化文件
│   ├── config.py                     (2.3 KB)     - 配置管理模块
│   ├── transaction.py                (2.0 KB)     - 交易数据结构
│   ├── actors.py                     (7.4 KB)     - 参与者类实现
│   ├── parameters.py                 (8.6 KB)     - 参数加载器
│   └── simulator.py                  (13 KB)      - 主仿真引擎
│
├── 运行脚本 (Python)
│   ├── run_paysim.py                 (1.9 KB)     - 主运行脚本
│   ├── example_quick_test.py         (1.4 KB)     - 快速测试示例
│   ├── test_minimal.py               (1.1 KB)     - 最小化测试
│   └── compare_outputs.py            (7.2 KB)     - Python/Java对比工具
│
├── 文档 (Markdown)
│   ├── README.md                     (5.7 KB)     - 用户手册
│   ├── QUICKSTART.md                 (4.8 KB)     - 快速开始指南
│   ├── PROJECT_SUMMARY.md            (7.9 KB)     - 项目总结
│   ├── ARCHITECTURE.md               (19 KB)      - 架构设计文档
│   └── INDEX.md                      (本文件)     - 文件索引
│
└── 输出目录
    └── outputs/                                   - 仿真结果输出
        └── PS_<timestamp>_<seed>/
            ├── PS_xxx_rawLog.csv                  - 原始交易日志
            ├── PS_xxx_Summary.txt                 - 仿真摘要
            └── PS_xxx_config.txt                  - 运行配置
```

**总代码量**: 约 2550 行（含文档）  
**核心代码**: 约 1100 行（纯Python）

---

## 📚 文档导航

### 🚀 新手入门
1. **首先阅读**: [QUICKSTART.md](QUICKSTART.md) - 5分钟快速上手
2. **然后阅读**: [README.md](README.md) - 完整使用手册
3. **深入理解**: [ARCHITECTURE.md](ARCHITECTURE.md) - 架构设计

### 📖 按主题查找

#### 安装与配置
- **安装依赖**: README.md → "安装依赖"
- **配置参数**: ARCHITECTURE.md → "配置参数说明"
- **配置文件格式**: config.py (源代码注释)

#### 快速使用
- **5分钟测试**: QUICKSTART.md → "第一步"
- **命令行用法**: QUICKSTART.md → "常用命令"
- **Python API**: QUICKSTART.md → "使用 Python 代码"

#### 核心概念
- **系统架构**: ARCHITECTURE.md → "系统架构"
- **类层次结构**: ARCHITECTURE.md → "类层次结构"
- **核心算法**: ARCHITECTURE.md → "核心算法流程"
- **弹簧模型**: ARCHITECTURE.md → "弹簧模型算法"

#### 开发扩展
- **添加交易类型**: ARCHITECTURE.md → "扩展开发指南"
- **添加欺诈模式**: ARCHITECTURE.md → "添加新的欺诈模式"
- **性能优化**: ARCHITECTURE.md → "性能优化策略"

#### 对比分析
- **Python vs Java**: PROJECT_SUMMARY.md → "与 Java 版本的对比"
- **测试结果**: PROJECT_SUMMARY.md → "测试结果"
- **运行对比**: compare_outputs.py (脚本)

---

## 🔍 代码模块速查

### config.py
**主要类**: `SimulationConfig`  
**作用**: 管理仿真配置参数  
**关键方法**:
- `from_properties_file()` - 从配置文件加载
- `get_scaled_value()` - 参数缩放

### transaction.py
**主要类**: `Transaction`  
**作用**: 定义交易数据结构  
**关键属性**:
- 交易信息 (step, action, amount)
- 账户信息 (orig/dest balances)
- 欺诈标记 (isFraud, isFlaggedFraud)

### actors.py
**类层次**:
```
SuperActor
├── Bank
├── Merchant
└── Client
    └── Fraudster
```
**关键方法**:
- `deposit()` / `withdraw()` - 资金操作
- `pick_count()` - 二项分布采样
- `pick_action()` - 交易类型选择（含弹簧模型）
- `pick_amount()` - 正态分布采样

### parameters.py
**主要类**: `ParameterLoader`  
**作用**: 加载 paramFiles 目录的参数  
**支持文件**:
- clientsProfiles.csv
- aggregatedTransactions.csv
- initialBalancesDistribution.csv
- overdraftLimits.csv
- transactionsTypes.csv

### simulator.py
**主要类**: `PaySimPython`  
**作用**: 仿真主引擎  
**核心流程**:
1. 初始化参与者
2. 主循环（每个时间步）
3. 收集交易记录
4. 输出结果

---

## 🎯 快速定位代码

### 想要...
- **修改交易类型**: → `actors.py:Client` 类常量
- **调整欺诈概率**: → `config.py:SimulationConfig.fraud_probability`
- **改变余额分布**: → `parameters.py:sample_initial_balance()`
- **自定义输出格式**: → `simulator.py:_save_results()`
- **增加新参数**: → `config.py:SimulationConfig`
- **修改弹簧模型**: → `actors.py:_compute_prob_with_spring()`
- **调整欺诈策略**: → `simulator.py:_step_fraudsters()`

### 想要理解...
- **如何选择交易次数**: → `actors.py:pick_count()`
- **如何选择交易类型**: → `actors.py:pick_action()`
- **如何选择交易金额**: → `actors.py:pick_amount()`
- **如何执行交易**: → `simulator.py:_make_transaction()`
- **如何标记欺诈**: → `simulator.py:_make_transaction()` 末尾
- **余额如何平衡**: → `actors.py:_compute_prob_with_spring()`

---

## 📊 输出文件说明

### rawLog.csv 列定义

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| step | int | 时间步 | 0, 1, 2, ... |
| type | str | 交易类型 | CASH_IN, TRANSFER, ... |
| amount | float | 交易金额 | 12525.99 |
| nameOrig | str | 源账户名 | C1234 |
| oldbalanceOrg | float | 源账户旧余额 | 158794.14 |
| newbalanceOrig | float | 源账户新余额 | 146268.16 |
| nameDest | str | 目标账户名 | M5678 |
| oldbalanceDest | float | 目标账户旧余额 | 0.0 |
| newbalanceDest | float | 目标账户新余额 | 12525.99 |
| isFraud | int | 是否欺诈 | 0 或 1 |
| isFlaggedFraud | int | 是否被标记 | 0 或 1 |

### 账户名称前缀
- `C` - Client (客户)
- `M` - Merchant (商户)
- `B` - Bank (银行)
- `F` - Fraudster (欺诈者，也是客户类型)

---

## 🔧 开发工具

### 运行测试
```bash
# 最小化测试（推荐首次运行）
python test_minimal.py

# 快速测试
python example_quick_test.py

# 完整测试
python run_paysim.py
```

### 代码质量检查
```bash
# 类型检查
mypy *.py

# 代码风格
pylint *.py
black *.py

# 代码复杂度
radon cc *.py -a
```

### 性能分析
```bash
# 性能剖析
python -m cProfile -o profile.stats test_minimal.py

# 内存分析
python -m memory_profiler test_minimal.py
```

---

## 📈 版本历史

### v2.0.0 (2025-10-20)
- ✅ 完成 Java 到 Python 完整重构
- ✅ 实现所有核心功能
- ✅ 添加完整文档
- ✅ 性能优化
- ✅ 测试通过

---

## 🤝 贡献指南

### 提交代码前检查
- [ ] 代码符合 PEP 8 规范
- [ ] 添加必要的注释和文档字符串
- [ ] 运行 `test_minimal.py` 确保基本功能正常
- [ ] 更新相关文档

### 报告问题
提供以下信息:
1. Python 版本
2. 依赖包版本 (numpy, pandas)
3. 完整的错误信息
4. 最小可复现示例

---

## 📞 联系方式

- **项目位置**: `/usr1/home/s124mdg53_07/wang/FYP/paysim_python/`
- **原始项目**: PaySim (Java) - Edgar Alonso Lopez-Rojas
- **重构版本**: PaySim Python v2.0

---

## 🎓 学习路径

### 初学者 (1-2小时)
1. 阅读 QUICKSTART.md
2. 运行 test_minimal.py
3. 查看生成的输出文件
4. 修改简单参数重新运行

### 进阶用户 (3-5小时)
1. 阅读 README.md 和 ARCHITECTURE.md
2. 理解核心算法（弹簧模型、概率采样）
3. 运行 compare_outputs.py 对比不同配置
4. 自定义客户行为配置

### 开发者 (5-10小时)
1. 深入研究源代码
2. 添加新的交易类型
3. 实现自定义欺诈策略
4. 性能优化和扩展开发

---

## 🔗 相关资源

### 外部链接
- [PaySim 原始论文](https://www.researchgate.net/publication/313138956)
- [MASON 框架文档](https://cs.gmu.edu/~eclab/projects/mason/)
- [NumPy 文档](https://numpy.org/doc/)
- [Pandas 文档](https://pandas.pydata.org/docs/)

### 内部参考
- Java 原始代码: `../PaySim/src/`
- 参数文件: `../PaySim/paramFiles/`
- Java 输出示例: `../PaySim/outputs/`

---

**最后更新**: 2025-10-20  
**文档版本**: 1.0  
**项目状态**: ✅ 生产就绪

---

💡 **提示**: 将此文件作为项目导航的起点！
