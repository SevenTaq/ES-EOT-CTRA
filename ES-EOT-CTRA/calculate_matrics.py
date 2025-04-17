import numpy as np
import os
from config import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './ES-EOT-CTRA_result.npy'), allow_pickle=True)

metrics = {'iou': [], 'e_v': [], 'pos': [], 'pos_gt': [], 'vel': [], 'vel_gt': [], 'acc': [], 'acc_gt': [], 'omega': [], 'omega_gt': []}

for frame in range(len(res)):
    x_ref = res[frame]['x_ref']
    pos = x_ref[:3]
    theta = x_ref[5:8]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]
    m = res[frame]['m']

    # detected shape ('u' for real shape, 'base' for radar detections)
    R = Rt.from_rotvec(theta).as_matrix()
    u = (R @ u.T).T + pos
    base = (R @ base.T).T + pos
    v = x_ref[3]
    v1 = np.abs(v)
    a = x_ref[4]
    a1 = np.abs(a)

    # ground truth shape
    verts = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    vel = np.linalg.norm(np.array(labels[frame]['velocity'][0]))
    gt_quats = np.array(labels[frame]['vehicle_quats'][0])
    R_gt = Rt.from_quat(gt_quats).as_matrix()
    R_gt1 = Rt.from_quat(gt_quats).as_rotvec()
    gt_pos = np.array(labels[frame]['vehicle_pos'][0])
    acc_gt = np.linalg.norm(np.array(labels[frame]['acceleration'][0]))

    # iou = iou_of_convex_hulls(u, verts)
    iou = iou_of_convex_hulls((R_gt.T @ (u - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
    diff_v = difference_between_velocity(v1, vel)

    metrics['iou'].append(iou)
    metrics['e_v'].append(diff_v**2)
    metrics['pos'].append(pos)
    metrics['pos_gt'].append(gt_pos)
    metrics['vel'].append(v)
    metrics['vel_gt'].append(vel)
    metrics['acc'].append(a)
    metrics['acc_gt'].append(acc_gt)
    metrics['omega'].append(x_ref[5:8])
    metrics['omega_gt'].append(R_gt1)

# 转换为numpy数组以便绘图
for key in ['pos', 'pos_gt', 'vel', 'vel_gt', 'acc', 'acc_gt', 'omega', 'omega_gt']:
    metrics[key] = np.array(metrics[key])

# 创建时间序列
time_steps = np.arange(len(res))

# 创建子图
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

# 位置对比图
ax1.plot(time_steps, metrics['pos'][:, 0], 'r-', label='估计位置 X')
ax1.plot(time_steps, metrics['pos'][:, 1], 'g-', label='估计位置 Y')
ax1.plot(time_steps, metrics['pos'][:, 2], 'b-', label='估计位置 Z')
ax1.plot(time_steps, metrics['pos_gt'][:, 0], 'r--', label='真实位置 X')
ax1.plot(time_steps, metrics['pos_gt'][:, 1], 'g--', label='真实位置 Y')
ax1.plot(time_steps, metrics['pos_gt'][:, 2], 'b--', label='真实位置 Z')
ax1.set_xlabel('时间步')
ax1.set_ylabel('位置 (m)')
ax1.grid(True)
ax1.legend()

# 速度对比图
ax2.plot(time_steps, metrics['vel'], 'r-', label='估计速度')
ax2.plot(time_steps, metrics['vel_gt'], 'b--', label='真实速度')
ax2.set_xlabel('时间步')
ax2.set_ylabel('速度 (m/s)')
ax2.grid(True)
ax2.legend()

# 加速度对比图
ax3.plot(time_steps, metrics['acc'], 'r-', label='估计加速度')
ax3.plot(time_steps, metrics['acc_gt'], 'b--', label='真实加速度')
ax3.set_xlabel('时间步')
ax3.set_ylabel('加速度 (m/s²)')
ax3.grid(True)
ax3.legend()

# 角速度对比图
ax4.plot(time_steps, metrics['omega'], 'r-', label='估计航向角')
ax4.plot(time_steps, metrics['omega_gt'], 'b--', label='真实航向角')
ax4.set_xlabel('时间步')
ax4.set_ylabel('航向角 (rad)')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()

print('平均IOU:', np.mean(metrics['iou'], axis=0))
print('速度RMSE:', np.sqrt(np.mean(metrics['e_v'])))
print(res[0]['x_ref'][:3])
print(np.array(labels[0]['vehicle_pos'][0]))