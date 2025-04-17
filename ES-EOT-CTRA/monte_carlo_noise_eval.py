import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rt
from scipy.linalg import block_diag

# 导入项目模块
from config import *
from FuncTools import *

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from metrics import *

def run_simulation(sigma_v_val, sigma_a_val, sigma_omega_val, sigma_ext_val, epsilon_val, rho_val, scenario='turn_around', noise_id=0):
    """
    使用指定的噪声参数和系统参数运行一次仿真
    
    参数:
    sigma_v_val: 速度噪声参数
    sigma_a_val: 加速度噪声参数
    sigma_omega_val: 角速度噪声参数
    sigma_ext_val: 扩展参数噪声
    epsilon_val: 弹性系数
    rho_val: 阻尼系数
    scenario: 场景名称
    noise_id: 噪声参数组合的ID
    
    返回:
    metrics_avg: 包含平均IOU、速度RMSE和加速度RMSE的字典
    """
    # 设置路径
    data_root_path = os.path.join(os.path.dirname(__file__), f'../data/{scenario}/')
    label_path = os.path.join(data_root_path, 'labels.npy')
    radar_dir_path = os.path.join(data_root_path, 'radar')
    keypoints_det_path = os.path.join(data_root_path, 'vision/output-keypoints.npy')
    
    # 加载测量数据
    labels = np.load(label_path, allow_pickle=True)
    keypoints_det = np.load(keypoints_det_path, allow_pickle=True)
    
    # 选择一个雷达文件进行测试
    radar_file = '10.0-1.npy'  # 可以根据需要更改
    file_path = os.path.join(radar_dir_path, radar_file)
    radar_point = np.load(file_path, allow_pickle=True)
    
    # 设置噪声参数和系统参数
    global sigma_v, sigma_a, sigma_omega, sigma_ext, W, epsilon, rho
    sigma_v = sigma_v_val
    sigma_a = sigma_a_val
    sigma_omega = sigma_omega_val
    sigma_ext = sigma_ext_val
    epsilon = epsilon_val
    rho = rho_val
    
    # 更新过程噪声矩阵W - 适用于CTRA模型
    W = block_diag(
        np.zeros((3, 3)),  # 位置噪声
        np.eye(1) * sigma_v,  # 速度噪声
        np.eye(1) * sigma_a,  # 加速度噪声
        np.zeros((3, 3)),  # 角度噪声
        np.eye(3) * sigma_omega,  # 角速度噪声
        np.eye(2) * sigma_ext  # 扩展参数噪声
    )
    
    # 初始化状态
    pos = labels[0]['vehicle_pos'][0]
    v = np.linalg.norm(labels[0]['velocity'][0])
    speed = labels[0]['velocity'][0]
    v1 = speed / v
    a1 = labels[0]['acceleration'][0]
    a2 = np.dot(a1, v1)
    a = np.linalg.norm(a2)
    quat = labels[0]['vehicle_quats'][0]
    theta = Rt.from_quat(quat).as_rotvec()
    
    x_ref = np.array([pos[0], pos[1], pos[2], v, a, theta[0], theta[1], theta[2], 0, 0, 0, 1, 1])
    
    dx = np.zeros(len(x_ref))
    P = np.eye(len(x_ref))
    P[3, 4] = P[4, 3] = 0.005
    
    m = np.ones(N_T)
    
    mu = np.random.normal(0, 1, (N_T, 9))
    Sigma = np.tile(np.identity(mu.shape[1]), (N_T, 1, 1))
    
    # 初始化状态
    Theta = State(x_ref, dx, P, m, mu, Sigma)
    
    res = []
    
    # 运行跟踪算法
    for i in range(len(radar_point)):
        z_r = np.array(radar_point[i])
        z_c = np.array(keypoints_det[i]['keypoints'][0], dtype=np.float64)
        
        Theta = update(Theta, z_r, z_c)
        
        now = dict()
        now['x_ref'] = Theta.x_ref.copy()
        now['P'] = Theta.P.copy()
        now['m'] = Theta.m.copy()
        now['mu'] = Theta.mu.copy()
        now['Sigma'] = Theta.Sigma.copy()
        res.append(now)
        
        Theta = predict(Theta)
    
    # 计算评估指标
    metrics = {'iou': [], 'e_v': [], 'e_a': []}
    
    for frame in range(len(res)):
        x_ref = res[frame]['x_ref']
        pos = x_ref[:3]
        theta = x_ref[5:8]
        mu = res[frame]['mu']
        u = mu[:, 3:6]
        base = mu[:, :3]
        
        # 检测形状
        R = Rt.from_rotvec(theta).as_matrix()
        u = (R @ u.T).T + pos
        base = (R @ base.T).T + pos
        v = x_ref[3]
        a = x_ref[4]  # CTRA模型中的加速度
        
        # 真实形状
        verts = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
        vel = np.linalg.norm(labels[frame]['velocity'][0])
        gt_quats = np.array(labels[frame]['vehicle_quats'][0])
        R_gt = Rt.from_quat(gt_quats).as_matrix()
        gt_pos = np.array(labels[frame]['vehicle_pos'][0])
        
        # 计算IOU
        iou = iou_of_convex_hulls((R_gt.T @ (u - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
        diff_v = difference_between_velocity(v, vel)
        
        # 计算加速度误差
        if frame > 0:
            v_gt_curr = np.linalg.norm(np.array(labels[frame]['velocity'][0]))
            v_gt_prev = np.linalg.norm(np.array(labels[frame - 1]['velocity'][0]))
            gt_a = (v_gt_curr - v_gt_prev) / dt
        else:
            gt_a = 0.0
        
        diff_a = np.abs(a - gt_a)
        
        metrics['iou'].append(iou)
        metrics['e_v'].append(diff_v)
        metrics['e_a'].append(diff_a)
    
    # 计算平均指标
    metrics_avg = {
        'iou_avg': np.mean(metrics['iou'], axis=0),
        'v_rmse': np.sqrt(np.mean(np.array(metrics['e_v'])**2)),
        'a_rmse': np.sqrt(np.mean(np.array(metrics['e_a'])**2))
    }
    
    return metrics_avg

def monte_carlo_noise_evaluation(n_samples=20, scenario='turn_around'):
    """
    使用蒙特卡洛方法评估不同噪声参数和系统参数组合的性能
    
    参数:
    n_samples: 每个参数范围内的采样数量
    scenario: 场景名称
    
    返回:
    best_params: 最佳参数组合
    all_results: 所有参数组合的结果
    """
    # 定义参数范围 - 针对CTRA模型调整
    sigma_v_range = np.linspace(0.05, 0.1, n_samples)  # 速度噪声
    sigma_a_range = np.linspace(0.01, 0.2, n_samples)    # 加速度噪声
    sigma_omega_range = np.linspace(0.5, 2.0, n_samples) # 角速度噪声
    sigma_ext_range = np.linspace(0.05, 0.5, n_samples)  # 扩展参数噪声
    epsilon_range = np.linspace(100, 200, n_samples)      # 弹性系数
    rho_range = np.linspace(10, 30, n_samples)          # 阻尼系数
    
    # 随机采样参数组合
    np.random.seed(42)  # 设置随机种子以确保可重复性
    n_combinations = 50  # 总共测试的参数组合数量
    
    param_combinations = []
    for _ in range(n_combinations):
        sigma_v_val = np.random.choice(sigma_v_range)
        sigma_a_val = np.random.choice(sigma_a_range)
        sigma_omega_val = np.random.choice(sigma_omega_range)
        sigma_ext_val = np.random.choice(sigma_ext_range)
        epsilon_val = np.random.choice(epsilon_range)
        rho_val = np.random.choice(rho_range)
        param_combinations.append((sigma_v_val, sigma_a_val, sigma_omega_val, sigma_ext_val, epsilon_val, rho_val))
    
    # 添加当前使用的参数组合
    param_combinations.append((0.01, 0.01, 1.0, 0.1, 100, 20))  # 当前配置
    
    # 存储结果
    all_results = []
    
    # 运行蒙特卡洛仿真
    for i, params in enumerate(tqdm(param_combinations, desc="Monte Carlo Simulation")):
        sigma_v_val, sigma_a_val, sigma_omega_val, sigma_ext_val, epsilon_val, rho_val = params
        
        # 运行仿真
        metrics = run_simulation(sigma_v_val, sigma_a_val, sigma_omega_val, sigma_ext_val, epsilon_val, rho_val, scenario, i)
        
        # 存储结果
        result = {
            'params': {
                'sigma_v': sigma_v_val,
                'sigma_a': sigma_a_val,
                'sigma_omega': sigma_omega_val,
                'sigma_ext': sigma_ext_val,
                'epsilon': epsilon_val,
                'rho': rho_val
            },
            'metrics': metrics
        }
        all_results.append(result)
    
    # 计算综合评分 - 针对CTRA模型调整权重
    for result in all_results:
        # 归一化指标
        iou_xy = result['metrics']['iou_avg'][0]  # xy平面的IOU
        v_rmse = result['metrics']['v_rmse']
        a_rmse = result['metrics']['a_rmse']
        
        # 综合评分 (高IOU更好，低RMSE更好)
        # 对于CTRA模型，我们更关注加速度的精度，因此增加其权重
        score = iou_xy - 0.2 * v_rmse - 0.03 * a_rmse
        result['score'] = score
    
    # 按评分排序
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best_params = all_results[0]['params']
    
    return best_params, all_results

def visualize_results(all_results):
    """
    可视化蒙特卡洛仿真结果
    
    参数:
    all_results: 所有参数组合的结果
    """
    # 提取数据
    sigma_v_values = [result['params']['sigma_v'] for result in all_results]
    sigma_a_values = [result['params']['sigma_a'] for result in all_results]
    sigma_omega_values = [result['params']['sigma_omega'] for result in all_results]
    sigma_ext_values = [result['params']['sigma_ext'] for result in all_results]
    
    iou_xy_values = [result['metrics']['iou_avg'][0] for result in all_results]
    v_rmse_values = [result['metrics']['v_rmse'] for result in all_results]
    a_rmse_values = [result['metrics']['a_rmse'] for result in all_results]
    scores = [result['score'] for result in all_results]
    
    # 创建图表
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # 速度噪声与IOU的关系
    axs[0, 0].scatter(sigma_v_values, iou_xy_values, c=scores, cmap='viridis')
    axs[0, 0].set_xlabel('速度噪声 (sigma_v)')
    axs[0, 0].set_ylabel('IOU (xy平面)')
    axs[0, 0].set_title('速度噪声与IOU的关系')
    
    # 加速度噪声与速度RMSE的关系
    axs[0, 1].scatter(sigma_a_values, v_rmse_values, c=scores, cmap='viridis')
    axs[0, 1].set_xlabel('加速度噪声 (sigma_a)')
    axs[0, 1].set_ylabel('速度RMSE')
    axs[0, 1].set_title('加速度噪声与速度RMSE的关系')
    
    # 角速度噪声与加速度RMSE的关系
    axs[1, 0].scatter(sigma_omega_values, a_rmse_values, c=scores, cmap='viridis')
    axs[1, 0].set_xlabel('角速度噪声 (sigma_omega)')
    axs[1, 0].set_ylabel('加速度RMSE')
    axs[1, 0].set_title('角速度噪声与加速度RMSE的关系')
    
    # 扩展参数噪声与综合评分的关系
    axs[1, 1].scatter(sigma_ext_values, scores, c=scores, cmap='viridis')
    axs[1, 1].set_xlabel('扩展参数噪声 (sigma_ext)')
    axs[1, 1].set_ylabel('综合评分')
    axs[1, 1].set_title('扩展参数噪声与综合评分的关系')
    
    # 弹性系数与IOU的关系
    epsilon_values = [result['params']['epsilon'] for result in all_results]
    axs[2, 0].scatter(epsilon_values, iou_xy_values, c=scores, cmap='viridis')
    axs[2, 0].set_xlabel('弹性系数 (epsilon)')
    axs[2, 0].set_ylabel('IOU (xy平面)')
    axs[2, 0].set_title('弹性系数与IOU的关系')
    
    # 阻尼系数与速度RMSE的关系
    rho_values = [result['params']['rho'] for result in all_results]
    axs[2, 1].scatter(rho_values, v_rmse_values, c=scores, cmap='viridis')
    axs[2, 1].set_xlabel('阻尼系数 (rho)')
    axs[2, 1].set_ylabel('速度RMSE')
    axs[2, 1].set_title('阻尼系数与速度RMSE的关系')
    
    # 添加颜色条
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axs.ravel().tolist())
    cbar.set_label('综合评分')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png')
    plt.show()
    
    # 打印最佳参数组合的表格
    print("\n最佳参数组合 (前5名):")
    print("-" * 80)
    print(f"{'排名':<6}{'sigma_v':<10}{'sigma_a':<10}{'sigma_omega':<15}{'sigma_ext':<15}{'epsilon':<10}{'rho':<10}{'IOU (xy)':<10}{'速度RMSE':<12}{'加速度RMSE':<12}{'评分':<8}")
    print("-" * 100)
    
    for i, result in enumerate(all_results[:5]):
        params = result['params']
        metrics = result['metrics']
        print(f"{i+1:<6}{params['sigma_v']:<10.4f}{params['sigma_a']:<10.4f}{params['sigma_omega']:<15.4f}{params['sigma_ext']:<15.4f}{params['epsilon']:<10.4f}{params['rho']:<10.4f}{metrics['iou_avg'][0]:<10.4f}{metrics['v_rmse']:<12.4f}{metrics['a_rmse']:<12.4f}{result['score']:<8.4f}")

def evaluate_current_params():
    """
    评估当前配置中使用的噪声参数和系统参数
    """
    # 当前配置的参数
    current_params = {
        'sigma_v': sigma_v,
        'sigma_a': sigma_a,
        'sigma_omega': sigma_omega,
        'sigma_ext': sigma_ext,
        'epsilon': epsilon,
        'rho': rho
    }
    
    print("\n当前配置的参数:")
    print(f"速度噪声 (sigma_v): {current_params['sigma_v']}")
    print(f"加速度噪声 (sigma_a): {current_params['sigma_a']}")
    print(f"角速度噪声 (sigma_omega): {current_params['sigma_omega']}")
    print(f"扩展参数噪声 (sigma_ext): {current_params['sigma_ext']}")
    print(f"弹性系数 (epsilon): {current_params['epsilon']}")
    print(f"阻尼系数 (rho): {current_params['rho']}")
    
    # 运行仿真评估当前参数
    metrics = run_simulation(current_params['sigma_v'], current_params['sigma_a'], 
                           current_params['sigma_omega'], current_params['sigma_ext'],
                           current_params['epsilon'], current_params['rho'])
    
    print("\n当前参数的性能指标:")
    print(f"IOU (xy平面): {metrics['iou_avg'][0]:.4f}")
    print(f"IOU (yz平面): {metrics['iou_avg'][1]:.4f}")
    print(f"IOU (xz平面): {metrics['iou_avg'][2]:.4f}")
    print(f"速度RMSE: {metrics['v_rmse']:.4f}")
    print(f"加速度RMSE: {metrics['a_rmse']:.4f}")
    
    # 计算综合评分
    score = metrics['iou_avg'][0] - 0.2 * metrics['v_rmse'] - 0.3 * metrics['a_rmse']
    print(f"综合评分: {score:.4f}")
    
    return current_params, metrics, score

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='蒙特卡洛仿真评估噪声参数')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'optimize'],
                        help='运行模式: evaluate (评估当前参数) 或 optimize (优化参数)')
    parser.add_argument('--samples', type=int, default=20, help='每个参数范围内的采样数量')
    parser.add_argument('--scenario', type=str, default='turn_around', help='场景名称')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        # 评估当前参数
        current_params, metrics, score = evaluate_current_params()
    else:
        # 运行蒙特卡洛优化
        print(f"开始蒙特卡洛仿真优化噪声参数 (采样数量: {args.samples})...")
        best_params, all_results = monte_carlo_noise_evaluation(n_samples=args.samples, scenario=args.scenario)
        
        # 可视化结果
        visualize_results(all_results)
        
        print("\n优化完成!")
        print(f"最佳噪声参数组合:")
        print(f"速度噪声 (sigma_v): {best_params['sigma_v']:.4f}")
        print(f"加速度噪声 (sigma_a): {best_params['sigma_a']:.4f}")
        print(f"角速度噪声 (sigma_omega): {best_params['sigma_omega']:.4f}")
        print(f"扩展参数噪声 (sigma_ext): {best_params['sigma_ext']:.4f}")
        
        # 保存最佳参数
        np.save('best_noise_params.npy', best_params)
        print("最佳参数已保存到 'best_noise_params.npy'")