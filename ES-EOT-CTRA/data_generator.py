import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as Rt
import os
import sys

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from metrics import iou_of_convex_hulls

class ElasticCoefficientDataset(Dataset):
    """用于生成弹性系数调节器训练数据的数据集"""
    def __init__(self, data_dir='../data', num_samples=1000):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """生成训练样本"""
        samples = []
        
        # 生成不同运动场景的样本
        for _ in range(self.num_samples):
            # 随机生成运动参数
            v = np.random.uniform(0, 30)  # 速度范围：0-30 m/s
            a = np.random.uniform(-5, 5)  # 加速度范围：-5 to 5 m/s²
            omega = np.random.uniform(-1, 1, size=3)  # 角速度范围：-1 to 1 rad/s
            
            # 生成姿态角
            theta = np.random.uniform(-np.pi/4, np.pi/4, size=3)
            R = Rt.from_rotvec(theta).as_matrix()
            velocity_direction = R @ np.array([1, 0, 0])
            
            # 计算加速度向量
            acceleration_vector = a * velocity_direction
            
            # 生成目标弹性系数因子（基于经验规则）
            base_factor = 1.0
            # 高速时增加弹性系数
            if v > 20:
                base_factor *= 1.2
            # 大加速度时增加弹性系数
            if abs(a) > 3:
                base_factor *= 1.1
            # 大角速度时增加弹性系数
            if np.linalg.norm(omega) > 0.5:
                base_factor *= 1.15
            
            # 确保因子在合理范围内
            target_factor = np.clip(base_factor, 0.5, 2.0)
            
            # 模拟IOU值（基于运动参数的复杂度）
            complexity = (v/30 + abs(a)/5 + np.linalg.norm(omega))/3
            iou = 1.0 - 0.3 * complexity  # IOU范围：0.7-1.0
            iou = np.clip(iou, 0.7, 1.0)
            
            # 组合特征向量
            motion_params = np.concatenate([acceleration_vector, omega])
            
            samples.append({
                'motion_params': motion_params,
                'target_factor': target_factor,
                'iou': iou
            })
        
        return samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (torch.FloatTensor(sample['motion_params']),
                torch.FloatTensor([sample['target_factor']]),
                torch.FloatTensor([sample['iou']]))

def get_train_dataloader(batch_size=32, num_samples=1000):
    """创建训练数据加载器"""
    dataset = ElasticCoefficientDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)