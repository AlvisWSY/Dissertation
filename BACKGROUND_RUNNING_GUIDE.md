# 🛡️ SSH断开保护 - 后台运行实验指南

当你通过SSH连接到服务器运行长时间实验时，如果网络断开或关闭终端，正在运行的进程会被终止。本指南提供多种方案来保持实验在后台持续运行。

---

## 📋 方案对比

| 方案 | 难度 | 推荐度 | 优点 | 缺点 |
|------|------|--------|------|------|
| **tmux** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 功能强大，可分屏，易用 | 需要学习快捷键 |
| **screen** | ⭐ | ⭐⭐⭐⭐ | 简单易用，广泛支持 | 功能较tmux少 |
| **nohup** | ⭐ | ⭐⭐⭐ | 最简单，无需额外安装 | 无法交互查看进度 |
| **systemd** | ⭐⭐⭐ | ⭐⭐ | 开机自启，管理规范 | 配置复杂，需root |

---

## 🚀 方案1: tmux（最推荐）

### 为什么选择tmux？
- ✅ 断开SSH后会话持续运行
- ✅ 随时重新连接查看实时输出
- ✅ 支持多窗口和分屏
- ✅ 可以在会话间切换

### 快速开始

```bash
# 一键启动（自动使用FYP虚拟环境）
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_tmux.sh
```

### 手动使用tmux

```bash
# 1. 创建新会话
tmux new -s experiment

# 2. 激活虚拟环境并运行实验
FYP
cd /usr1/home/s124mdg53_07/wang/FYP
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=72000 \
    --output=experiment_executed.ipynb \
    src/experiment.ipynb

# 3. 断开会话（实验继续运行）
# 按 Ctrl+B，然后按 D

# 4. 重新连接（随时查看进度）
tmux attach -t experiment

# 5. 列出所有会话
tmux ls

# 6. 终止会话
tmux kill-session -t experiment
```

### tmux核心快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+B, D` | 断开会话（detach） |
| `Ctrl+B, C` | 创建新窗口 |
| `Ctrl+B, N` | 切换到下一个窗口 |
| `Ctrl+B, P` | 切换到上一个窗口 |
| `Ctrl+B, %` | 垂直分屏 |
| `Ctrl+B, "` | 水平分屏 |
| `Ctrl+B, ←/→/↑/↓` | 切换分屏 |
| `Ctrl+B, [` | 进入滚动模式（按Q退出） |

### 高级技巧

```bash
# 创建会话时指定名称
tmux new -s experiment_1

# 查看所有运行中的会话
tmux ls

# 连接到指定会话
tmux attach -t experiment_1

# 在会话中运行监控
tmux new -s monitor
watch -n 1 nvidia-smi

# 同时查看实验和GPU（分屏）
# 1. 连接到实验会话: tmux attach -t experiment
# 2. 按 Ctrl+B, % 垂直分屏
# 3. 在新分屏运行: watch -n 1 nvidia-smi
```

---

## 🚀 方案2: screen（简单易用）

### 快速开始

```bash
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_screen.sh
```

### 手动使用screen

```bash
# 1. 创建新会话
screen -S experiment

# 2. 激活虚拟环境并运行实验
FYP
cd /usr1/home/s124mdg53_07/wang/FYP
jupyter nbconvert --to notebook --execute src/experiment.ipynb

# 3. 断开会话（按 Ctrl+A，然后按 D）

# 4. 重新连接
screen -r experiment

# 5. 列出所有会话
screen -ls

# 6. 终止会话
screen -S experiment -X quit
```

### screen核心快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+A, D` | 断开会话 |
| `Ctrl+A, C` | 创建新窗口 |
| `Ctrl+A, N` | 下一个窗口 |
| `Ctrl+A, P` | 上一个窗口 |
| `Ctrl+A, [` | 进入滚动模式 |

---

## 🚀 方案3: nohup（最简单）

### 特点
- ✅ 最简单，不需要额外安装
- ✅ 直接后台运行
- ❌ 无法实时查看进度
- ❌ 只能通过日志文件查看输出

### 快速开始

```bash
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_nohup.sh
```

### 手动使用nohup

```bash
# 1. 后台运行实验
FYP
cd /usr1/home/s124mdg53_07/wang/FYP
nohup jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=72000 \
    --output=experiment_executed.ipynb \
    src/experiment.ipynb > experiment.log 2>&1 &

# 记录进程ID
echo $! > experiment.pid

# 2. 查看日志
tail -f experiment.log

# 3. 检查进程状态
ps aux | grep $(cat experiment.pid)

# 4. 终止进程
kill $(cat experiment.pid)
```

---

## 📊 实际使用场景

### 场景1: 首次测试（推荐tmux）

```bash
# 使用tmux进行快速测试
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_tmux.sh

# 选择 1 (快速验证)
# 可以实时看到输出
# 按 Ctrl+B, D 断开测试
```

### 场景2: 标准实验（1-2小时）

```bash
# 使用tmux运行
./run_with_tmux.sh
# 选择 2 (标准实验)

# 断开会话
# 按 Ctrl+B, D

# 1小时后重新连接查看进度
tmux attach -t experiment
```

### 场景3: 完整实验（3-4小时，夜间运行）

```bash
# 使用nohup在后台运行
./run_with_nohup.sh
# 选择 3 (完整实验)

# 可以安全关闭终端和断开网络
# 第二天查看结果
tail -100 experiment_nohup.log
cat results/results.csv
```

### 场景4: 同时监控多项指标

```bash
# 使用tmux分屏
tmux new -s monitor

# 第一个窗口：运行实验
FYP
cd /usr1/home/s124mdg53_07/wang/FYP
jupyter nbconvert --to notebook --execute src/experiment.ipynb

# 按 Ctrl+B, % 垂直分屏

# 第二个窗口：监控GPU
watch -n 1 nvidia-smi

# 按 Ctrl+B, " 再次水平分屏

# 第三个窗口：监控日志
tail -f experiment_log.txt

# 按 Ctrl+B, D 断开（所有监控继续运行）
```

---

## 🔍 监控和管理

### 查看实验进度

```bash
# tmux/screen方式
tmux attach -t experiment  # 或 screen -r experiment

# nohup方式
tail -f experiment_nohup.log
tail -n 100 experiment_nohup.log  # 查看最后100行
```

### 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或在tmux/screen中持续运行
tmux new -s gpu_monitor
watch -n 1 nvidia-smi
# 按 Ctrl+B, D 断开
```

### 监控系统资源

```bash
# CPU和内存
htop

# 或在tmux分屏中
tmux split-window -h htop
```

### 查看实验输出文件

```bash
# 查看结果
cat results/results.csv
head -20 results/results.csv

# 查看执行后的notebook
jupyter notebook src/experiment_executed.ipynb

# 查看日志
less experiment_log.txt
```

---

## 🐛 故障排查

### 问题1: tmux/screen未安装

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tmux screen

# CentOS/RHEL
sudo yum install tmux screen

# 使用conda（不需要sudo）
conda install -c conda-forge tmux

# 如果都不行，使用nohup
./run_with_nohup.sh
```

### 问题2: 无法重新连接tmux会话

```bash
# 列出所有会话
tmux ls

# 如果显示 "no server running"
# 说明会话已经结束，检查日志：
cat experiment_log.txt

# 如果显示会话列表但无法连接
# 尝试强制连接
tmux attach -t experiment -d
```

### 问题3: 后台进程被终止

```bash
# 检查进程是否还在运行
ps aux | grep jupyter
ps aux | grep python

# 如果进程消失了，可能原因：
# 1. 服务器重启
# 2. 内存不足被OOM killer杀死
# 3. 手动终止了

# 查看系统日志
dmesg | tail -50
journalctl -xe

# 查看内存使用
free -h
```

### 问题4: 找不到实验输出

```bash
# 检查工作目录
pwd

# 搜索输出文件
find /usr1/home/s124mdg53_07/wang/FYP -name "experiment_executed.ipynb"
find /usr1/home/s124mdg53_07/wang/FYP -name "*.log"

# 检查results目录
ls -lh results/
```

---

## 💡 最佳实践

### 1. 实验前准备

```bash
# 确认虚拟环境
FYP
which python
python --version

# 确认GPU可用
nvidia-smi

# 确认磁盘空间
df -h /usr1/home/s124mdg53_07/wang/FYP

# 测试环境
python src/test_environment.py
```

### 2. 使用描述性会话名称

```bash
# 好的命名
tmux new -s exp_creditcard_smote_2024
tmux new -s exp_ieee_standard_v1

# 不好的命名
tmux new -s test
tmux new -s tmp
```

### 3. 保存重要信息

```bash
# 在启动实验时记录
cat > experiment_info.txt << EOF
实验开始时间: $(date)
会话名称: experiment
虚拟环境: FYP
配置: 标准实验，3个数据集
预计完成时间: $(date -d '+2 hours')
EOF

# 保存PID
echo $! > experiment.pid
```

### 4. 设置告警

```bash
# 实验完成后发送邮件（如果配置了邮件）
jupyter nbconvert --execute src/experiment.ipynb && \
    echo "实验完成" | mail -s "实验完成通知" your@email.com

# 或保存完成标记
jupyter nbconvert --execute src/experiment.ipynb && \
    touch EXPERIMENT_COMPLETED
```

### 5. 定期检查

```bash
# 创建检查脚本
cat > check_experiment.sh << 'EOF'
#!/bin/bash
echo "实验状态检查: $(date)"
echo "================================"
echo "进程状态:"
ps aux | grep jupyter | grep -v grep

echo ""
echo "GPU使用:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo ""
echo "最新日志 (最后10行):"
tail -10 experiment_log.txt
EOF

chmod +x check_experiment.sh

# 定期运行
watch -n 300 ./check_experiment.sh  # 每5分钟检查一次
```

---

## 📚 快速参考

### tmux速查表

```bash
# 会话管理
tmux new -s NAME       # 创建会话
tmux ls               # 列出会话
tmux attach -t NAME   # 连接会话
tmux kill-session -t NAME  # 终止会话

# 会话内快捷键（先按Ctrl+B，再按对应键）
D     断开会话
C     新窗口
N     下一窗口
P     上一窗口
%     垂直分屏
"     水平分屏
方向键  切换分屏
[     滚动模式（Q退出）
```

### screen速查表

```bash
# 会话管理
screen -S NAME        # 创建会话
screen -ls           # 列出会话
screen -r NAME       # 连接会话
screen -S NAME -X quit  # 终止会话

# 会话内快捷键（先按Ctrl+A，再按对应键）
D     断开会话
C     新窗口
N     下一窗口
P     上一窗口
[     滚动模式
```

### nohup速查表

```bash
# 运行
nohup COMMAND > output.log 2>&1 &
echo $! > process.pid

# 监控
tail -f output.log
ps aux | grep $(cat process.pid)

# 终止
kill $(cat process.pid)
```

---

## 🎓 推荐工作流

### 完整实验流程

```bash
# 1. 准备阶段
cd /usr1/home/s124mdg53_07/wang/FYP
FYP
python src/test_environment.py

# 2. 创建tmux会话
tmux new -s experiment_$(date +%Y%m%d)

# 3. 分屏设置
# 主屏：运行实验
# 右侧：监控GPU
# 按 Ctrl+B, % 分屏
watch -n 1 nvidia-smi

# 切回主屏: Ctrl+B, 方向键左

# 4. 启动实验
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=72000 \
    --output=experiment_executed.ipynb \
    src/experiment.ipynb 2>&1 | tee experiment_log.txt

# 5. 断开会话
# 按 Ctrl+B, D

# 6. 定期检查（新终端）
tmux attach -t experiment_$(date +%Y%m%d)

# 7. 实验完成后
# 在tmux会话中按 Ctrl+D 关闭会话
# 或 tmux kill-session -t experiment_$(date +%Y%m%d)
```

---

## ✅ 检查清单

实验开始前：
- [ ] 激活FYP虚拟环境
- [ ] 确认GPU可用
- [ ] 确认磁盘空间充足
- [ ] 选择合适的后台运行方案
- [ ] 创建描述性会话名称
- [ ] 准备监控脚本

实验运行中：
- [ ] 成功断开SSH连接（测试）
- [ ] 能重新连接查看进度
- [ ] GPU使用率正常（60-90%）
- [ ] 内存使用正常（<80%）
- [ ] 日志文件正常增长

实验完成后：
- [ ] 检查结果文件
- [ ] 保存执行后的notebook
- [ ] 备份重要日志
- [ ] 关闭后台会话
- [ ] 清理临时文件

---

**祝实验顺利！** 🚀

记住：**tmux是你最好的朋友**，随时按 `Ctrl+B, D` 断开，随时用 `tmux attach` 重连！
