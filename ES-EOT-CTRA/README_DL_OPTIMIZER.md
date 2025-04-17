# 深度学习参数优化器

本模块使用深度强化学习方法自动优化ES-EOT-CTRA模型的噪声参数和系统参数，以提高跟踪性能。

## 功能特点

- 使用深度强化学习自动寻找最优参数组合
- 支持与传统蒙特卡洛方法的性能比较
- 可视化训练过程和优化结果
- 自动应用最佳参数到系统配置

## 参数说明

优化的参数包括：

- `sigma_v`：速度噪声参数
- `sigma_a`：加速度噪声参数
- `sigma_omega`：角速度噪声参数
- `sigma_ext`：扩展参数噪声
- `epsilon`：弹性系数
- `rho`：阻尼系数

## 使用方法

### 1. 训练深度学习模型

```bash
python dl_param_optimizer.py --mode train --episodes 100 --scenario turn_around
```

参数说明：
- `--episodes`：训练轮数，默认为100
- `--scenario`：场景名称，默认为'turn_around'
- `--apply`：添加此参数将自动应用优化后的参数到系统配置

### 2. 使用训练好的模型优化参数

```bash
python dl_param_optimizer.py --mode optimize --model_path dl_param_policy.pth --scenario turn_around --apply
```

参数说明：
- `--model_path`：训练好的模型路径，默认为'dl_param_policy.pth'

### 3. 比较深度学习方法与蒙特卡洛方法

```bash
python dl_param_optimizer.py --mode compare --model_path dl_param_policy.pth --mc_params_path best_noise_params.npy --scenario turn_around --apply
```

参数说明：
- `--mc_params_path`：蒙特卡洛方法优化的最佳参数路径，默认为'best_noise_params.npy'

## 输出文件

- `dl_param_policy.pth`：训练好的深度学习模型
- `dl_best_params.npy`：深度学习方法找到的最佳参数
- `dl_training_rewards.png`：训练过程中奖励值的变化曲线
- `methods_comparison.png`：不同方法性能比较的可视化结果

## 技术实现

本模块使用深度强化学习中的策略梯度方法，通过神经网络学习参数与性能指标之间的映射关系。主要评估指标包括：

- IOU（交并比）：评估目标形状估计的准确性
- 速度RMSE：评估速度估计的准确性
- 加速度RMSE：评估加速度估计的准确性

综合评分计算公式：
```
score = iou_xy * 10 - v_rmse * 2 - a_rmse * 3
```

## 依赖库

- PyTorch
- NumPy
- Matplotlib
- tqdm

## 安装依赖

```bash
pip install torch numpy matplotlib tqdm
```