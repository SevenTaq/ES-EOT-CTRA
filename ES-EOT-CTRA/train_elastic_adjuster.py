import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from elastic_coefficient_adjuster import ElasticCoefficientAdjuster, train_model, save_model
from monte_carlo_noise_eval import run_simulation
from tqdm import tqdm

# 创建自定义数据集
class ElasticAdjusterDataset(data.Dataset):
    def __init__(self, size=100):
        self.size = size
        self.motion_params = []
        self.target_factors = []
        self.ious = []
        
        # 生成训练数据
        print("正在生成训练数据...")
        for _ in tqdm(range(size), desc="生成训练数据"):
            # 随机生成运动参数
            acceleration = np.random.normal(0, 1, 3)  # 随机加速度
            angular_velocity = np.random.normal(0, 0.5, 3)  # 随机角速度
            
            # 随机生成目标弹性系数因子（0.5到2.0之间）
            target_factor = np.random.uniform(0.5, 2.0)
            
            # 运行仿真获取IOU
            metrics = run_simulation(
                sigma_v_val=0.1,
                sigma_a_val=0.05,
                sigma_omega_val=1.0,
                sigma_ext_val=0.05,
                epsilon_val=200 * target_factor,  # 使用目标因子调整弹性系数
                rho_val=20.0
            )
            
            # 保存数据
            self.motion_params.append(np.concatenate([acceleration, angular_velocity]))
            self.target_factors.append([target_factor])
            self.ious.append([np.mean(metrics['iou_avg'])])
        
        # 转换为张量
        self.motion_params = torch.FloatTensor(self.motion_params)
        self.target_factors = torch.FloatTensor(self.target_factors)
        self.ious = torch.FloatTensor(self.ious)
        
        print("数据生成完成！")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.motion_params[idx], self.target_factors[idx], self.ious[idx]

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建数据集和数据加载器
    dataset = ElasticAdjusterDataset(size=200)
    train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = ElasticCoefficientAdjuster(base_epsilon=200)
    
    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, epochs=100, lr=0.001, use_tqdm=True)
    
    # 保存模型
    save_model(model, 'models/elastic_adjuster.pth')
    print("模型训练完成并保存！")

if __name__ == '__main__':
    main()