import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
import matplotlib
from collections import deque
from tqdm import tqdm

# 设置中文字体支持
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ElasticCoefficientAdjuster(nn.Module):
    """轻量级神经网络模型，用于动态调整弹性系数，具有性能监控和自适应调整功能"""
    def __init__(self, base_epsilon=200):
        super(ElasticCoefficientAdjuster, self).__init__()
        self.base_epsilon = base_epsilon
        self.min_epsilon = 100  # 最小弹性系数
        self.max_epsilon = 200  # 最大弹性系数
        
        # 轻量级网络结构: 输入->8->4->1
        self.fc1 = nn.Linear(6, 8)  # 输入：线性加速度(3) + 角速度(3)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        
        # 性能监控相关
        self.performance_history = {  # 历史性能记录
            'iou': [],
            'rmse': [],
            'epsilon_history': [],
            'adjustment_factors': []
        }
        self.performance_window = 10  # 性能评估窗口大小
        self.exploration_factor = 1.0  # 探索因子
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 6] (线性加速度和角速度)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 使输出在0.5到1.0之间，避免过度调整
        adjustment_factor = 0.5 + 0.5 * torch.sigmoid(self.fc3(x))
        return adjustment_factor
        
    def update_performance_metrics(self, iou, rmse, epsilon, adjustment_factor):
        """更新性能指标记录"""
        self.performance_history['iou'].append(iou)
        self.performance_history['rmse'].append(rmse)
        self.performance_history['epsilon_history'].append(epsilon)
        self.performance_history['adjustment_factors'].append(adjustment_factor)
        
        # 保持历史记录在窗口大小范围内
        if len(self.performance_history['iou']) > self.performance_window:
            for key in self.performance_history:
                self.performance_history[key] = self.performance_history[key][-self.performance_window:]
    
    def adjust_exploration(self):
        """根据性能历史调整探索因子"""
        if len(self.performance_history['iou']) >= 3:
            recent_iou = self.performance_history['iou'][-3:]
            if all(x < y for x, y in zip(recent_iou[:-1], recent_iou[1:])):
                # 性能持续提升，减少探索
                self.exploration_factor = max(0.5, self.exploration_factor * 0.9)
            elif all(x > y for x, y in zip(recent_iou[:-1], recent_iou[1:])):
                # 性能持续下降，增加探索
                self.exploration_factor = min(2.0, self.exploration_factor * 1.2)
    
    def get_adjusted_epsilon(self, x_ref, current_iou=None, current_rmse=None):
        """根据当前状态向量获取调整后的弹性系数
        
        参数:
            x_ref: 状态向量，包含位置、速度、加速度、姿态、角速度等信息
            current_iou: 当前IOU值（可选），可以是单个值或包含三个平面IOU的数组
            current_rmse: 当前RMSE值（可选），可以是单个值或包含速度和加速度RMSE的字典
        
        返回:
            adjusted_epsilon: 调整后的弹性系数
        """
        with torch.no_grad():
            # 从状态向量中提取加速度和角速度
            # 获取完整的加速度向量，而仅仅是x方向
            # CTRA模型中，x_ref[4]是标量加速度，需要转换为向量形式
            # 使用速度方向作为加速度方向
            v = x_ref[3]  # 速度大小
            theta = x_ref[5:8]  # 姿态角
            
            # 使用姿态角计算速度方向
            from scipy.spatial.transform import Rotation as Rt
            R = Rt.from_rotvec(theta).as_matrix()
            velocity_direction = R @ np.array([1, 0, 0])  # 假设车辆前进方向为x轴
            
            # 计算加速度向量 = 加速度大小 * 速度方向
            acceleration_vector = x_ref[4] * velocity_direction
            acceleration = torch.tensor(acceleration_vector, dtype=torch.float32)
            angular_velocity = torch.tensor(x_ref[8:11], dtype=torch.float32)  # 角速度向量
            
            # 组合特征
            features = torch.cat([acceleration, angular_velocity]).unsqueeze(0)
            
            # 获取基础调整因子
            base_adjustment = self.forward(features).item()
            
            # 根据IOU和RMSE调整探索因子
            # 如果没有提供当前性能指标，尝试从跟踪系统获取
            if current_iou is None or current_rmse is None:
                # 尝试从跟踪系统获取实时性能指标
                try:
                    # 导入必要的模块
                    import sys
                    import os
                    if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
                        sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
                    from metrics import iou_of_convex_hulls, difference_between_velocity
                    
                    # 获取当前跟踪结果
                    # 这里假设我们可以访问最近一帧的跟踪结果和真实标签
                    # 实际实现中，这些数据可能需要从全局变量或缓存中获取
                    from config import frame_counter
                    
                    # 如果有可用的跟踪结果和标签数据
                    if 'latest_tracking_result' in globals() and 'latest_label' in globals():
                        result = globals()['latest_tracking_result']
                        label = globals()['latest_label']
                        
                        # 提取跟踪形状和真实形状
                        x_ref_current = result['x_ref']
                        pos = x_ref_current[:3]
                        theta = x_ref_current[5:8]
                        mu = result['mu']
                        u = mu[:, 3:6]
                        base = mu[:, :3]
                        
                        # 计算跟踪形状
                        R = Rt.from_rotvec(theta).as_matrix()
                        u = (R @ u.T).T + pos
                        base = (R @ base.T).T + pos
                        v = x_ref_current[3]
                        a = x_ref_current[4]
                        
                        # 提取真实形状
                        verts = np.array(label['keypoints_world_all'][0])[:, 1:4]
                        vel = np.linalg.norm(label['velocity'][0])
                        gt_quats = np.array(label['vehicle_quats'][0])
                        R_gt = Rt.from_quat(gt_quats).as_matrix()
                        gt_pos = np.array(label['vehicle_pos'][0])
                        
                        # 计算IOU
                        current_iou = iou_of_convex_hulls((R_gt.T @ (u - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
                        
                        # 计算速度误差
                        v_diff = difference_between_velocity(v, vel)
                        
                        # 计算加速度误差
                        if frame_counter > 0 and 'previous_label' in globals():
                            prev_label = globals()['previous_label']
                            v_gt_curr = np.linalg.norm(np.array(label['velocity'][0]))
                            v_gt_prev = np.linalg.norm(np.array(prev_label['velocity'][0]))
                            gt_a = (v_gt_curr - v_gt_prev) / 0.05  # 假设dt=0.05
                            a_diff = np.abs(a - gt_a)
                        else:
                            a_diff = 0.0
                        
                        # 构建RMSE字典
                        current_rmse = {
                            'v_rmse': np.abs(v_diff),
                            'a_rmse': a_diff
                        }
                except Exception as e:
                    # 如果获取失败，使用默认值
                    print(f"获取实时性能指标失败: {e}")
                    if current_iou is None:
                        current_iou = np.array([0.7, 0.7, 0.7])  # 默认中等IOU值
                    if current_rmse is None:
                        current_rmse = {'v_rmse': 0.5, 'a_rmse': 0.5}  # 默认中等RMSE值
            
            # 处理性能指标并调整探索因子
            if current_iou is not None and current_rmse is not None:
                # 处理不同格式的IOU输入
                if isinstance(current_iou, np.ndarray) and len(current_iou) >= 3:
                    # 如果是数组，取IOU平均值
                    iou_value = np.mean(current_iou)
                else:
                    # 否则直接使用输入值
                    iou_value = current_iou
                
                # 处理不同格式的RMSE输入
                if isinstance(current_rmse, dict):
                    # 如果是字典，获取速度和加速度RMSE
                    v_rmse = current_rmse.get('v_rmse', 0)
                    # 计算综合RMSE，加速度误差权重更高
                    rmse_value = v_rmse
                else:
                    # 否则直接使用输入值
                    rmse_value = current_rmse
                
                # 根据IOU和RMSE动态调整探索因子
                # IOU越高，探索因子越小（减少探索）
                # RMSE越高，探索因子越大（增加探索）
                iou_factor = max(0.5, 1.0 - iou_value * 0.5)  # IOU影响因子
                rmse_factor = min(1.5, 1.0 + rmse_value * 0.1)  # RMSE影响因子
                
                # 更新探索因子
                self.exploration_factor = self.exploration_factor * 0.8 + (iou_factor * rmse_factor) * 0.2
                self.exploration_factor = np.clip(self.exploration_factor, 0.5, 2.0)
                
                # 更新性能指标记录
                self.update_performance_metrics(iou_value, rmse_value, 
                                              self.base_epsilon, base_adjustment)
                self.adjust_exploration()
            
            # 应用探索因子
            adjustment_factor = base_adjustment * self.exploration_factor
            
            # 计算基础弹性系数
            adjusted_epsilon = self.base_epsilon * adjustment_factor
            
            # 确保弹性系数在合理范围内
            adjusted_epsilon = np.clip(adjusted_epsilon, self.min_epsilon, self.max_epsilon)
            
        return adjusted_epsilon

# 模型训练函数（离线训练）
def train_model(model, train_dataloader, epochs=5, lr=0.001, use_tqdm=False):
    """
    训练弹性系数调节器模型
    
    参数:
        model: 弹性系数调节器模型
        train_dataloader: 包含(运动参数, 最佳弹性系数因子, IOU值)的数据加载器
        epochs: 训练轮数
        lr: 学习率
        use_tqdm: 是否使用进度条
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epoch_pbar = tqdm(range(epochs), desc="训练进度") if use_tqdm else range(epochs)
    for epoch in epoch_pbar:
        running_loss = 0.0
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False) if use_tqdm else train_dataloader
        for motion_params, target_factors, ious in batch_pbar:
            optimizer.zero_grad()
            
            # 前向传播
            predicted_factors = model(motion_params)
            
            # 计算损失（加权MSE，基于IOU）
            loss = criterion(predicted_factors, target_factors)
            # IOU权重 - 使模型更注重提高IOU值高的情况
            iou_weights = F.softmax(ious, dim=0)
            weighted_loss = (loss * iou_weights).mean()
            
            # 反向传播
            weighted_loss.backward()
            optimizer.step()
            
            running_loss += weighted_loss.item()
            if use_tqdm:
                batch_pbar.set_postfix({"loss": f"{weighted_loss.item():.4f}"})
        
        avg_loss = running_loss/len(train_dataloader)
        if use_tqdm:
            epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        else:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return model

# 保存和加载模型
def save_model(model, path='models/elastic_adjuster.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
def load_model(path='models/elastic_adjuster.pth', base_epsilon=100):
    model = ElasticCoefficientAdjuster(base_epsilon)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
    return model