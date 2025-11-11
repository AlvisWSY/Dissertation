# 🔧 实验错误修复指南

## ❌ 错误: ModuleNotFoundError: No module named 'xgboost'

### 📋 错误分析

**日志文件**: `experiment_log.txt`

**错误信息**:
```
ModuleNotFoundError: No module named 'xgboost'
```

**发生位置**: notebook第一个单元格导入库时

---

## 🔍 根本原因

✅ xgboost **已经安装** 在FYP环境中（版本3.0.5）  
❌ 但运行实验时 **没有激活FYP环境**

### 详细说明

当你运行：
```bash
jupyter nbconvert --execute src/experiment.ipynb
```

这会使用**当前激活的Python环境**。如果没有激活FYP，就会使用base环境，而base环境中没有安装xgboost。

---

## ✅ 解决方案

### 方案1: 使用提供的脚本（最简单）✨

所有脚本都已经配置好自动激活FYP环境：

```bash
# 选择任一方式
./run_with_tmux.sh        # tmux方式（推荐）
./run_with_screen.sh      # screen方式
./run_with_nohup.sh       # nohup方式
./run_experiment_background.sh  # 统一入口
```

这些脚本内部都包含了 `FYP` 命令来激活环境。

---

### 方案2: 使用Jupyter Notebook（推荐）

```bash
# 启动Jupyter（自动激活FYP）
./start_jupyter.sh

# 或手动：
FYP
cd /usr1/home/s124mdg53_07/wang/FYP
jupyter notebook

# 然后在浏览器中：
# 1. 打开 src/experiment.ipynb
# 2. 点击 Cell -> Run All
```

---

### 方案3: 手动命令行运行

```bash
# ✅ 正确方式（先激活FYP）
cd /usr1/home/s124mdg53_07/wang/FYP
FYP  # 激活虚拟环境！！！
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=72000 \
    --output=experiment_executed.ipynb \
    src/experiment.ipynb 2>&1 | tee experiment_log.txt
```

```bash
# ❌ 错误方式（忘记激活环境）
cd /usr1/home/s124mdg53_07/wang/FYP
jupyter nbconvert --execute src/experiment.ipynb  # ❌ 会失败！
```

---

## 🧪 验证环境

在运行实验前，先验证环境配置：

```bash
# 激活环境
FYP

# 验证Python路径
which python
# 应该输出: /usr1/home/s124mdg53_07/anaconda3/envs/FYP/bin/python

# 验证包安装
python -c "import xgboost; print('xgboost版本:', xgboost.__version__)"
python -c "import lightgbm; print('lightgbm版本:', lightgbm.__version__)"
python -c "import torch; print('torch版本:', torch.__version__)"

# 或运行测试脚本
python src/test_environment.py
```

---

## 📊 重新运行实验

现在环境已经正确配置，你可以：

### 选项A: 使用tmux后台运行（推荐）

```bash
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_tmux.sh

# 选择实验模式
# 按 Ctrl+B, D 断开（实验继续运行）
# 稍后重新连接: tmux attach -t experiment
```

### 选项B: 直接在Jupyter中运行

```bash
./start_jupyter.sh
# 在浏览器中打开notebook并运行
```

### 选项C: 快速测试

```bash
FYP
python src/test_environment.py
```

---

## 🔍 检查点清单

运行实验前，确认以下几点：

- [ ] ✅ 已激活FYP环境（命令提示符显示 `(FYP)`）
- [ ] ✅ Python路径正确（`which python` 指向FYP环境）
- [ ] ✅ xgboost可以导入（`python -c "import xgboost"`）
- [ ] ✅ GPU可用（`nvidia-smi` 显示GPU）
- [ ] ✅ 使用提供的脚本或手动激活环境后运行

---

## 🐛 其他可能的问题

### 问题1: 环境激活命令不工作

```bash
# 如果FYP命令不存在，使用完整命令
conda activate FYP

# 或
source /usr1/home/s124mdg53_07/anaconda3/bin/activate FYP
```

### 问题2: Jupyter找不到kernel

```bash
# 安装ipykernel到FYP环境
FYP
pip install ipykernel
python -m ipykernel install --user --name=FYP --display-name="Python (FYP)"

# 重启Jupyter
```

### 问题3: 包版本冲突

```bash
# 重新安装依赖
FYP
pip install -r requirements.txt --force-reinstall
```

### 问题4: 权限问题

```bash
# 使用--user安装
FYP
pip install -r requirements.txt --user
```

---

## 📝 错误预防

### 创建快捷命令（可选）

在 `~/.bashrc` 中添加：

```bash
# 快捷实验命令
alias exp-test='cd /usr1/home/s124mdg53_07/wang/FYP && FYP && python src/test_environment.py'
alias exp-start='cd /usr1/home/s124mdg53_07/wang/FYP && ./run_with_tmux.sh'
alias exp-jupyter='cd /usr1/home/s124mdg53_07/wang/FYP && ./start_jupyter.sh'
alias exp-status='tmux attach -t experiment'
```

然后：
```bash
source ~/.bashrc

# 之后可以直接使用
exp-test      # 快速测试
exp-start     # 启动实验
exp-jupyter   # 启动Jupyter
exp-status    # 查看实验状态
```

---

## 🎯 现在开始吧！

### 最简单的开始方式

```bash
cd /usr1/home/s124mdg53_07/wang/FYP

# 方式1: tmux后台运行（推荐）
./run_with_tmux.sh

# 方式2: Jupyter Notebook
./start_jupyter.sh

# 方式3: 快速测试
FYP
python src/test_environment.py
```

---

## 💡 关键要记住的

**永远记住**: 在运行任何Python命令前，先执行 `FYP` 激活环境！

```bash
# ✅ 正确流程
FYP                    # 1. 激活环境
cd .../FYP             # 2. 进入目录
python xxx.py          # 3. 运行脚本

# ❌ 错误流程
cd .../FYP             # 忘记激活环境
python xxx.py          # 会使用错误的Python环境
```

---

## 📚 相关文档

- `START_HERE.md` - 完整开始指南
- `BACKGROUND_RUNNING_GUIDE.md` - 后台运行指南
- `OPTIMIZATION_GUIDE.md` - 性能优化指南
- `QUICK_REFERENCE_BACKGROUND.md` - 快速参考

---

**问题已解决！现在可以正常运行实验了！** ✅
