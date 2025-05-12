#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os

# --- 配置参数 ---
PYTHON_EXECUTABLE = "python"  # 或者 "python3"，根据你的环境
TRAIN_SCRIPT = "train.py"
COMMON_ARGS = ["--use_gpu", "--epochs=30"] # 其他固定参数
LEARNING_RATES = [1e-2, 1e-3, 1e-4]

# 检查 train.py 是否存在
if not os.path.exists(TRAIN_SCRIPT):
    print(f"错误: 训练脚本 '{TRAIN_SCRIPT}' 未在当前目录找到。")
    exit(1)

# --- 主逻辑 ---
def run_training_with_lr(lr_value):
    """
    使用指定的学习率运行训练脚本。
    """
    lr_arg = f"--lr={lr_value}"
    command = [PYTHON_EXECUTABLE, TRAIN_SCRIPT, lr_arg] + COMMON_ARGS

    print("-" * 50)
    print(f"🚀 正在执行命令: {' '.join(command)}")
    print("-" * 50)

    try:
        # shell=False 更安全，命令和参数作为列表传递
        # check=True 会在命令返回非零退出码时抛出 CalledProcessError
        process = subprocess.run(command, check=True, text=True)
        print(f"✅ 学习率 {lr_value} 训练完成。")
    except subprocess.CalledProcessError as e:
        print(f"❌ 学习率 {lr_value} 训练失败。")
        print(f"错误码: {e.returncode}")
        if e.stdout:
            print(f"标准输出:\n{e.stdout}")
        if e.stderr:
            print(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        print(f"错误: Python 可执行文件 '{PYTHON_EXECUTABLE}' 或训练脚本 '{TRAIN_SCRIPT}' 未找到。")
        print("请确保它们在你的 PATH 环境变量中，或者提供正确的路径。")
        exit(1)
    except Exception as e:
        print(f"❌ 执行命令时发生未知错误: {e}")

if __name__ == "__main__":
    print("自动化训练脚本已启动...")
    print(f"将使用以下学习率进行训练: {LEARNING_RATES}")
    print(f"通用参数: {' '.join(COMMON_ARGS)}")
    print(f"训练脚本: {TRAIN_SCRIPT}")
    print("\n")

    for lr in LEARNING_RATES:
        run_training_with_lr(lr)
        print("\n") # 在两次训练之间添加一些间隔

    print("所有训练任务已尝试执行完毕。")