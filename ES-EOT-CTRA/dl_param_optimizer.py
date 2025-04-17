import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque

# 导入项目模块
from config import *
from FuncTools import *
from monte_carlo_noise_eval import run_simulation

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from metrics import *

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义参数范围
PARAM_RANGES = {
    'sigma_v': (0.08, 0.12),      # 速度噪声
    'sigma_a': (0.04, 0.06),        # 加速度噪声
    'sigma_omega': (0.9, 1.1),     # 角速度噪声
    'sigma_ext': (0.01, 0.1),      # 扩展参数噪声
    'epsilon': (100, 200),          # 弹性系数
    'rho': (18, 22)                # 阻尼系数
}

# 定义神经网络模型
class ParamPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ParamPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # 使用tanh输出[-1,1]范围的值，后续会映射到参数范围
        x = self.tanh(self.fc3(x))
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=200):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# 将神经网络输出的动作（-1到1范围）映射到实际参数范围
def map_action_to_params(action):
    params = {}
    for i, param_name in enumerate(PARAM_RANGES.keys()):
        min_val, max_val = PARAM_RANGES[param_name]
        # 将[-1,1]映射到[min_val, max_val]
        params[param_name] = ((action[i] + 1) / 2) * (max_val - min_val) + min_val
    return params

# 从仿真结果中提取状态特征
def extract_state_features(metrics, prev_params=None):
    # 基本特征：IOU和RMSE指标
    features = [
        metrics['iou_avg'][0],  # xy平面IOU
        metrics['iou_avg'][1],  # yz平面IOU
        metrics['iou_avg'][2],  # xz平面IOU
        metrics['v_rmse'],      # 速度RMSE
        metrics['a_rmse']       # 加速度RMSE
    ]
    
    # 如果有前一组参数，添加参数变化率作为特征
    if prev_params is not None:
        for param_name in PARAM_RANGES.keys():
            features.append(prev_params[param_name])
    else:
        # 首次运行，添加默认参数作为特征
        features.extend([sigma_v, sigma_a, sigma_omega, sigma_ext, epsilon, rho])
    
    return np.array(features, dtype=np.float32)

# 计算奖励函数
def calculate_reward(metrics):
    # 奖励函数：高IOU奖励，低RMSE奖励
    iou_xy = metrics['iou_avg'][0]
    v_rmse = metrics['v_rmse']
    
    # 综合评分 (高IOU更好，低RMSE更好)
    reward = iou_xy * 10
    return reward

# 运行单次仿真并获取评估指标
def run_single_simulation(params, scenario='turn_around'):
    metrics = run_simulation(
        params['sigma_v'], 
        params['sigma_a'], 
        params['sigma_omega'], 
        params['sigma_ext'],
        params['epsilon'], 
        params['rho'], 
        scenario
    )
    return metrics

# 深度强化学习训练函数
def train_rl_optimizer(episodes=100, scenario='turn_around'):
    set_seed(42)
    
    # 定义状态和动作维度
    state_dim = 11  # 5个指标 + 6个参数
    action_dim = 6  # 6个参数
    
    # 初始化策略网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ParamPolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # 初始化经验回FFER缓冲区
    replay_buffer = ReplayBuffer(capacity=200)
    batch_size = 32
    
    # 训练记录
    rewards_history = []
    best_reward = float('-inf')
    best_params = None
    
    # 初始状态：使用当前配置参数运行仿真
    current_params = {
        'sigma_v': sigma_v,
        'sigma_a': sigma_a,
        'sigma_omega': sigma_omega,
        'sigma_ext': sigma_ext,
        'epsilon': epsilon,
        'rho': rho
    }
    
    metrics = run_single_simulation(current_params, scenario)
    state = extract_state_features(metrics)
    
    print("开始深度强化学习训练...")
    for episode in tqdm(range(episodes), desc="训练进度"):
        # 使用策略网络选择动作（参数调整）
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(state_tensor).cpu().numpy()[0]
        
        # 添加探索噪声
        exploration_noise = np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action + exploration_noise, -1, 1)
        
        # 将动作映射到实际参数
        next_params = map_action_to_params(action)
        
        # 使用新参数运行仿真
        metrics = run_single_simulation(next_params, scenario)
        reward = calculate_reward(metrics)
        next_state = extract_state_features(metrics, next_params)
        
        # 存储经验
        replay_buffer.push(state, action, reward, next_state, False)
        
        # 更新状态和参数
        state = next_state
        current_params = next_params
        
        # 记录奖励
        rewards_history.append(reward)
        
        # 更新最佳参数
        if reward > best_reward:
            best_reward = reward
            best_params = current_params.copy()
            print(f"\n第{episode+1}轮：发现更好的参数组合，奖励值：{best_reward:.4f}")
            for param_name, value in best_params.items():
                print(f"{param_name}: {value:.4f}")
        
        # 当经验回FFER缓冲区足够大时，进行批量学习
        if len(replay_buffer) >= batch_size:
            # 从经验回FFER缓冲区采样
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # 转换为张量
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            
            # 计算策略梯度
            optimizer.zero_grad()
            pred_actions = policy_net(states)
            loss = nn.MSELoss()(pred_actions, actions)
            loss.backward()
            optimizer.step()
    
    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel('训练轮次')
    plt.ylabel('奖励值')
    plt.title('深度强化学习训练过程')
    plt.savefig('dl_training_rewards.png')
    plt.show()
    
    # 保存最佳参数和模型
    np.save('dl_best_params.npy', best_params)
    torch.save(policy_net.state_dict(), 'dl_param_policy.pth')
    
    print("\n训练完成！")
    print(f"最佳参数组合 (奖励值: {best_reward:.4f}):")
    for param_name, value in best_params.items():
        print(f"{param_name}: {value:.4f}")
    
    return best_params, rewards_history

# 使用训练好的模型进行参数优化
def optimize_params_with_dl(model_path='dl_param_policy.pth', scenario='turn_around'):
    # 定义状态和动作维度
    state_dim = 11  # 5个指标 + 6个参数
    action_dim = 6  # 6个参数
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ParamPolicyNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()
    
    # 初始状态：使用当前配置参数运行仿真
    current_params = {
        'sigma_v': sigma_v,
        'sigma_a': sigma_a,
        'sigma_omega': sigma_omega,
        'sigma_ext': sigma_ext,
        'epsilon': epsilon,
        'rho': rho
    }
    
    metrics = run_single_simulation(current_params, scenario)
    state = extract_state_features(metrics)
    
    # 使用策略网络选择最优参数
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = policy_net(state_tensor).cpu().numpy()[0]
    
    # 将动作映射到实际参数
    optimized_params = map_action_to_params(action)
    
    # 使用优化后的参数运行仿真
    metrics = run_single_simulation(optimized_params, scenario)
    reward = calculate_reward(metrics)
    
    print("\n深度学习参数优化结果:")
    print(f"奖励值: {reward:.4f}")
    print(f"IOU (xy平面): {metrics['iou_avg'][0]:.4f}")
    print(f"速度RMSE: {metrics['v_rmse']:.4f}")
    print(f"加速度RMSE: {metrics['a_rmse']:.4f}")
    
    for param_name, value in optimized_params.items():
        print(f"{param_name}: {value:.4f}")
    
    return optimized_params, metrics

# 比较深度学习方法与蒙特卡洛方法
def compare_methods(dl_params, mc_params, scenario='turn_around'):
    # 运行深度学习优化的参数
    dl_metrics = run_single_simulation(dl_params, scenario)
    dl_reward = calculate_reward(dl_metrics)
    
    # 运行蒙特卡洛优化的参数
    mc_metrics = run_single_simulation(mc_params, scenario)
    mc_reward = calculate_reward(mc_metrics)
    
    # 运行默认参数
    default_params = {
        'sigma_v': sigma_v,
        'sigma_a': sigma_a,
        'sigma_omega': sigma_omega,
        'sigma_ext': sigma_ext,
        'epsilon': epsilon,
        'rho': rho
    }
    default_metrics = run_single_simulation(default_params, scenario)
    default_reward = calculate_reward(default_metrics)
    
    # 打印比较结果
    print("\n方法比较结果:")
    print("-" * 80)
    print(f"{'方法':<15}{'奖励值':<10}{'IOU (xy)':<10}{'速度RMSE':<12}{'加速度RMSE':<12}")
    print("-" * 80)
    print(f"{'默认参数':<15}{default_reward:<10.4f}{default_metrics['iou_avg'][0]:<10.4f}{default_metrics['v_rmse']:<12.4f}{default_metrics['a_rmse']:<12.4f}")
    print(f"{'蒙特卡洛':<15}{mc_reward:<10.4f}{mc_metrics['iou_avg'][0]:<10.4f}{mc_metrics['v_rmse']:<12.4f}{mc_metrics['a_rmse']:<12.4f}")
    print(f"{'深度学习':<15}{dl_reward:<10.4f}{dl_metrics['iou_avg'][0]:<10.4f}{dl_metrics['v_rmse']:<12.4f}{dl_metrics['a_rmse']:<12.4f}")
    
    # 可视化比较结果
    methods = ['默认参数', '蒙特卡洛', '深度学习']
    rewards = [default_reward, mc_reward, dl_reward]
    ious = [default_metrics['iou_avg'][0], mc_metrics['iou_avg'][0], dl_metrics['iou_avg'][0]]
    v_rmses = [default_metrics['v_rmse'], mc_metrics['v_rmse'], dl_metrics['v_rmse']]
    a_rmses = [default_metrics['a_rmse'], mc_metrics['a_rmse'], dl_metrics['a_rmse']]
    
    # 创建子图
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制奖励值比较
    axs[0, 0].bar(methods, rewards, color=['gray', 'blue', 'red'])
    axs[0, 0].set_title('奖励值比较')
    axs[0, 0].set_ylabel('奖励值')
    
    # 绘制IOU比较
    axs[0, 1].bar(methods, ious, color=['gray', 'blue', 'red'])
    axs[0, 1].set_title('IOU (xy平面) 比较')
    axs[0, 1].set_ylabel('IOU')
    
    # 绘制速度RMSE比较
    axs[1, 0].bar(methods, v_rmses, color=['gray', 'blue', 'red'])
    axs[1, 0].set_title('速度RMSE比较')
    axs[1, 0].set_ylabel('RMSE')
    
    # 绘制加速度RMSE比较
    axs[1, 1].bar(methods, a_rmses, color=['gray', 'blue', 'red'])
    axs[1, 1].set_title('加速度RMSE比较')
    axs[1, 1].set_ylabel('RMSE')
    
    plt.tight_layout()
    plt.savefig('methods_comparison.png')
    plt.show()
    
    return {
        'default': {'params': default_params, 'metrics': default_metrics, 'reward': default_reward},
        'monte_carlo': {'params': mc_params, 'metrics': mc_metrics, 'reward': mc_reward},
        'deep_learning': {'params': dl_params, 'metrics': dl_metrics, 'reward': dl_reward}
    }

# 应用最佳参数到配置
def apply_best_params(params):
    global sigma_v, sigma_a, sigma_omega, sigma_ext, epsilon, rho, W
    
    # 更新全局参数
    sigma_v = params['sigma_v']
    sigma_a = params['sigma_a']
    sigma_omega = params['sigma_omega']
    sigma_ext = params['sigma_ext']
    epsilon = params['epsilon']
    rho = params['rho']
    
    # 更新过程噪声矩阵W
    W = block_diag(
        np.zeros((3, 3)),  # 位置噪声
        np.eye(1) * sigma_v,  # 速度噪声
        np.eye(1) * sigma_a,  # 加速度噪声
        np.zeros((3, 3)),  # 角度噪声
        np.eye(3) * sigma_omega,  # 角速度噪声
        np.eye(2) * sigma_ext  # 扩展参数噪声
    )
    
    print("\n已应用最佳参数到系统配置")
    print(f"速度噪声 (sigma_v): {sigma_v:.4f}")
    print(f"加速度噪声 (sigma_a): {sigma_a:.4f}")
    print(f"角速度噪声 (sigma_omega): {sigma_omega:.4f}")
    print(f"扩展参数噪声 (sigma_ext): {sigma_ext:.4f}")
    print(f"弹性系数 (epsilon): {epsilon:.4f}")
    print(f"阻尼系数 (rho): {rho:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='深度学习参数优化器')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'optimize', 'compare'],
                        help='运行模式: train (训练模型), optimize (使用模型优化参数), compare (比较方法)')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--scenario', type=str, default='turn_around', help='场景名称')
    parser.add_argument('--model_path', type=str, default='dl_param_policy.pth', help='模型路径')
    parser.add_argument('--mc_params_path', type=str, default='best_noise_params.npy', help='蒙特卡洛最佳参数路径')
    parser.add_argument('--apply', action='store_true', help='是否应用优化后的参数到系统配置')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练深度强化学习模型
        print(f"开始训练深度强化学习模型 (训练轮数: {args.episodes})...")
        best_params, _ = train_rl_optimizer(episodes=args.episodes, scenario=args.scenario)
        
        if args.apply:
            apply_best_params(best_params)
            
    elif args.mode == 'optimize':
        # 使用训练好的模型优化参数
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件 '{args.model_path}' 不存在，请先训练模型")
            sys.exit(1)
            
        print(f"使用深度学习模型 '{args.model_path}' 优化参数...")
        optimized_params, _ = optimize_params_with_dl(model_path=args.model_path, scenario=args.scenario)
        
        if args.apply:
            apply_best_params(optimized_params)
            
    elif args.mode == 'compare':
        # 比较深度学习方法与蒙特卡洛方法
        if not os.path.exists(args.model_path):
            print(f"错误: 深度学习模型文件 '{args.model_path}' 不存在，请先训练模型")
            sys.exit(1)
            
        if not os.path.exists(args.mc_params_path):
            print(f"错误: 蒙特卡洛参数文件 '{args.mc_params_path}' 不存在，请先运行蒙特卡洛优化")
            sys.exit(1)
        
        # 加载蒙特卡洛最佳参数
        mc_params = np.load(args.mc_params_path, allow_pickle=True).item()
        
        # 使用深度学习模型优化参数
        dl_params, _ = optimize_params_with_dl(model_path=args.model_path, scenario=args.scenario)
        
        # 比较两种方法
        results = compare_methods(dl_params, mc_params, scenario=args.scenario)
        
        # 确定最佳方法
        best_method = max(results.items(), key=lambda x: x[1]['reward'])[0]
        best_params = results[best_method]['params']
        
        print(f"\n最佳方法: {best_method}")
        
        if args.apply:
            apply_best_params(best_params)