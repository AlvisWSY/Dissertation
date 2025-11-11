# 🛡️ SSH断开保护 - 快速参考卡

## 🚀 一键启动（最简单）

```bash
cd /usr1/home/s124mdg53_07/wang/FYP
./run_experiment_background.sh
```

---

## 📋 三种方案速览

### 方案1: tmux（最推荐）⭐⭐⭐⭐⭐

```bash
# 快速启动
./run_with_tmux.sh

# 核心操作
tmux new -s experiment     # 创建会话
Ctrl+B, D                  # 断开（实验继续运行）
tmux attach -t experiment  # 重新连接
tmux ls                    # 查看所有会话
```

**优点**: 可实时查看进度，支持分屏  
**缺点**: 需要记几个快捷键

---

### 方案2: screen（简单易用）⭐⭐⭐⭐

```bash
# 快速启动
./run_with_screen.sh

# 核心操作
screen -S experiment       # 创建会话
Ctrl+A, D                  # 断开
screen -r experiment       # 重新连接
screen -ls                 # 查看所有会话
```

**优点**: 操作简单  
**缺点**: 功能比tmux少

---

### 方案3: nohup（最简单）⭐⭐⭐

```bash
# 快速启动
./run_with_nohup.sh

# 查看进度
tail -f experiment_nohup.log
```

**优点**: 最简单，无需学习  
**缺点**: 无法实时交互

---

## 🎯 推荐使用场景

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 首次测试 | tmux | 可以实时看到输出 |
| 标准实验（1-2h） | tmux | 可以随时检查进度 |
| 长时间实验（过夜） | nohup | 简单可靠 |
| 需要监控GPU | tmux | 可以分屏同时查看 |

---

## 💡 核心命令记忆法

### tmux
- **创建**: `tmux new -s 名字`
- **断开**: `Ctrl+B, D` (B=Before, D=Detach)
- **重连**: `tmux attach -t 名字`

### screen  
- **创建**: `screen -S 名字`
- **断开**: `Ctrl+A, D` (A=Activate, D=Detach)
- **重连**: `screen -r 名字`

### nohup
- **启动**: `nohup 命令 > log.txt 2>&1 &`
- **查看**: `tail -f log.txt`

---

## 🔍 常用监控命令

```bash
# 查看GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f experiment_log.txt

# 查看进程
ps aux | grep python

# 查看结果
cat results/results.csv
```

---

## ⚠️ 重要提醒

1. **断开前确认**: 实验已经在运行中
2. **记住会话名**: 方便重新连接
3. **定期检查**: 确保实验正常运行
4. **保存日志**: 便于问题排查

---

## 🆘 紧急情况

```bash
# 找不到会话？
tmux ls          # 列出所有tmux会话
screen -ls       # 列出所有screen会话

# 实验卡住了？
# 重新连接到会话，按 Ctrl+C 终止

# 完全不知道状态？
ps aux | grep jupyter    # 查看是否在运行
tail -100 experiment_log.txt  # 查看最新日志
```

---

## 📚 完整文档

```bash
# 查看完整指南
less BACKGROUND_RUNNING_GUIDE.md

# 查看开始指南
less START_HERE.md

# 查看优化指南
less OPTIMIZATION_GUIDE.md
```

---

## ✅ 最简工作流

```bash
# 1. 启动后台运行
cd /usr1/home/s124mdg53_07/wang/FYP
./run_with_tmux.sh

# 2. 选择实验模式（如：标准实验）

# 3. 断开会话
# 按 Ctrl+B, D

# 4. 可以安全关闭终端/断开SSH

# 5. 稍后重新连接查看进度
tmux attach -t experiment

# 6. 实验完成后查看结果
cat results/results.csv
```

---

**记住**: 所有脚本都已经配置好FYP虚拟环境，直接运行即可！🚀
