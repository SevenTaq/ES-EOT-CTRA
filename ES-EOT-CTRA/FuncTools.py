import numpy as np
from scipy.spatial.transform import Rotation as Rt
from config import *
from math import sin, cos
from scipy.special import digamma
from scipy.spatial import ConvexHull
from scipy.stats import multivariate_normal
import random
import time


def points_visibility(points, hull_points, hull_simplices, origin=None):
    if origin is None:
        origin = [0, 0, 0]
    points = points[np.newaxis, :] if points.ndim == 1 else points
    simplices_3d = np.asarray(hull_points)[hull_simplices]

    # Calculate the normal vectors of rays and triangles
    direction = (points - origin) / np.linalg.norm(points - origin, axis=1, keepdims=True)
    triangle_normal = np.cross(simplices_3d[:, 1] - simplices_3d[:, 0], simplices_3d[:, 2] - simplices_3d[:, 0], axis=1)

    # Calculate the intersection point of a ray and the plane where the triangle is located
    t = (triangle_normal * (simplices_3d[:, 0] - origin)).sum(axis=1) / (
            triangle_normal * direction[:, np.newaxis, :]).sum(axis=2)
    intersection_point = origin + t[:, :, np.newaxis] * direction[:, np.newaxis, :]

    # Judge if the intersection point is inside the triangle
    res = np.cross((np.roll(simplices_3d, -1, axis=1) - simplices_3d)[np.newaxis, :, :, :],
                   intersection_point[:, :, np.newaxis, :] - simplices_3d)
    now = res @ res.transpose((0, 1, 3, 2))

    inter_bool = np.logical_and(np.all(now >= 0, axis=(2, 3)),
                                t <= np.linalg.norm(points - origin, axis=1)[:, np.newaxis] - 1e-5, t >= 1e-5)

    return np.logical_not(np.any(inter_bool, axis=1))


def skew_symmetry(v):
    assert v.ndim == 1 or v.ndim == 2
    if v.ndim == 1:
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    else:
        return np.array([skew_symmetry(now) for now in v])


def exp_skew_v(v):
    assert len(v) == 3
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        skew = skew_symmetry(v / norm_v)
        return np.identity(3) + np.sin(norm_v) * skew + (1 - np.cos(norm_v)) * skew @ skew
    elif 0 < norm_v <= eps:
        skew = skew_symmetry(v / norm_v)
        return np.identity(3) + norm_v * skew + 1/2 * (norm_v**2) * skew @ skew
    else:
        return np.identity(3)

def f(x_ref):
    p_ref = x_ref[:3].copy()
    v_ref = np.abs(x_ref[3])
    a_ref = x_ref[4]
    theta_ref = x_ref[5:8].copy()
    omega_ref = x_ref[8:11].copy()
    ext_ref = x_ref[11:13].copy()

    R_ref = Rt.from_rotvec(theta_ref).as_matrix()
    norm_omega_ref = np.linalg.norm(omega_ref)
    norm_a_ref = abs(a_ref)
    if norm_a_ref > eps:
        v_ref1 = v_ref + a_ref * dt / 2
    else:
        v_ref1 = v_ref

    if norm_omega_ref > eps:
        skew = skew_symmetry(omega_ref / norm_omega_ref)
        p_ref += v_ref1 * (np.identity(3) * dt +
                           1 / norm_omega_ref * skew * (1 - cos(norm_omega_ref * dt)) +
                           skew @ skew * (dt - sin(norm_omega_ref * dt) / norm_omega_ref)
                           ) @ R_ref @ u_d
    elif 0 < norm_omega_ref <= eps :
        skew = skew_symmetry(omega_ref / norm_omega_ref)
        p_ref += v_ref1 * ((np.identity(3) + 1 / 2 * norm_omega_ref * dt * skew + 1 / 6 * (
                    (norm_omega_ref * dt) ** 2) * skew @ skew) * dt) @ R_ref @ u_d
    else:
        p_ref += v_ref1 * (np.identity(3) * dt) @ R_ref @ u_d

    R_ref = exp_skew_v(omega_ref * dt) @ R_ref
    theta_ref = Rt.from_matrix(R_ref).as_rotvec()
    v_ref = v_ref + a_ref*dt
    return np.array([*p_ref, v_ref, a_ref, *theta_ref, *omega_ref, *ext_ref])


def get_phi(x_ref):
    p_ref = x_ref[:3].copy()
    v_ref = np.abs(x_ref[3])
    a_ref = x_ref[4]
    theta_ref = x_ref[5:8].copy()
    omega_ref = x_ref[8:11].copy()


    R_ref = Rt.from_rotvec(theta_ref).as_matrix()
    M_0 = (R_ref @ u_d).reshape((3, 1))
    M_0v = M_0 * dt
    M_0a = 0.5 * M_0v * dt
    norm_a_ref = abs(a_ref)
    if norm_a_ref > eps:
        v_ref1 = v_ref + a_ref * dt / 2
        a = 1
    else:
        v_ref1 = v_ref
        a = 0
    M_1 = -v_ref1 * skew_symmetry(R_ref @ u_d)
    M_2 = skew_symmetry(omega_ref)

    norm_omega_ref = np.linalg.norm(omega_ref)
    if norm_omega_ref > eps:
        S_0 = Rt.from_rotvec(omega_ref * dt).as_matrix()
        S_1 = np.identity(3) * dt - M_2 / norm_omega_ref ** 2 @ (S_0 - np.identity(3) - M_2 * dt)
        S_2 = 1 / 2 * np.identity(3) * (dt ** 2) - (S_0 - np.identity(3) - M_2 * dt - 1 / 2 * M_2 @ M_2 * (dt ** 2)) / (norm_omega_ref ** 2)

    else:
        S_0 = np.identity(3)
        S_1 = np.identity(3) * dt
        S_2 = np.identity(3) * (dt**2) / 2
    Phi = np.block([
        [np.identity(3), M_0v, M_0a*a, M_1 @ S_1, M_1 @ S_2, np.zeros((3, 2))],
        [np.zeros((1, 3)), np.identity(1), np.identity(1)*dt, np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 2))],
        [np.zeros((1, 3)), np.zeros((1, 1)), np.identity(1), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 2))],
        [np.zeros((3, 3)), np.zeros((3, 1)), np.zeros((3, 1)), S_0, S_1+S_2, np.zeros((3, 2))],
        [np.zeros((3, 3)), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 3)), np.identity(3), np.zeros((3, 2))],
        [np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 3)), np.zeros((2, 3)), np.identity(2)]
    ])
    return Phi


def get_J(v):
    norm_v = np.linalg.norm(v)
    skew = skew_symmetry(v)
    if norm_v > eps:
        return np.identity(3) + (1 - cos(norm_v)) / norm_v**2 * skew + \
                    (norm_v - sin(norm_v)) / norm_v**3 * skew @ skew
    else:
        return np.identity(3) + 1 / 2 * skew + 1 / 6 * skew @ skew


def get_phi_vartheta(epsilon_list):

    # 遍历每个弹簧的 ε 和 ρ
        # 计算阻尼判别式
    now = rho ** 2 - 4 * epsilon  # Δ = c² - 4mk
    # 判断阻尼类型
    if now > eps:  # 过阻尼
      sqrt_now = np.sqrt(now)
      sinh_term = np.sinh(0.5 * sqrt_now * dt)
      cosh_term = np.cosh(0.5 * sqrt_now * dt)
      exp_term = np.exp(-rho * dt / 2)
      S_3 = -1 + exp_term * (cosh_term + (rho / sqrt_now) * sinh_term)
      S_4 = 2 * exp_term * sinh_term / sqrt_now
    elif now < -eps:  # 欠阻尼
            sqrt_now = np.sqrt(-now)
            sin_term = np.sin(0.5 * sqrt_now * dt)
            cos_term = np.cos(0.5 * sqrt_now * dt)
            exp_term = np.exp(-rho * dt / 2)
            S_3 = -1 + exp_term * (cos_term + (rho / sqrt_now) * sin_term)
            S_4 = 2 * exp_term * sin_term / sqrt_now
    else:  # 临界阻尼
            exp_term = np.exp(-rho * dt / 2)
            S_3 = -1 + exp_term * (1 + 0.5 * rho * dt)
            S_4 = exp_term * dt

    # 构建状态转移矩阵块
    concat = np.zeros((len(epsilon_list), 3, 3))
    concat[:, 0, 0] = 1.0
    concat[:, 1, 0] = -S_3
    concat[:, 1, 1] = 1 + S_3
    concat[:, 1, 2] = S_4
    concat[:, 2, 0] = epsilon_list * S_4
    concat[:, 2, 1] = -epsilon_list * S_4
    concat[:, 2, 2] = 1 + S_3 - rho * S_4

    return np.kron(concat, np.identity(3))


def predict(Theta):
    ret = Theta.copy()
    x_ref = ret.x_ref
    mu = ret.mu
    Sigma = ret.Sigma

    # Predict x
    Phi = get_phi(x_ref)
    ret.x_ref = f(x_ref)
    
    # 根据帧计数器选择不同的系数
    ret.P = Phi @ Theta.P @ Phi.T + W * dt

    # Predict vartheta
    # k_list = Theta.m / np.max(Theta.m) * epsilon
    k_list = np.ones_like(Theta.m) * epsilon
    # 获取Phi_vartheta并检查维度
    Phi_vartheta = get_phi_vartheta(k_list)

    ret.mu = (Phi_vartheta @ mu[:, :, np.newaxis]).squeeze()

    ret.Sigma = Phi_vartheta @ Sigma @ Phi_vartheta.transpose([0, 2, 1]) + W_vartheta * dt

    # 预测并调整m值（跟踪点权重）
    # 当最大权重超过阈值时应用衰减因子，防止权重无限增长
    max_m = np.max(ret.m)
    if max_m > 100:
        # 使用向量化操作应用衰减因子
        ret.m = np.maximum(ret.m * decay, 1.0)
    else:
        # 确保所有权重不小于最小阈值
        ret.m[ret.m < 1.0] = 1.0
        
    return ret

def Star(M):
    assert M.ndim == 2 or M.ndim == 3
    if M.ndim == 2:
        assert M.shape[0] == 3
        return np.concatenate(skew_symmetry(M.T), axis=1)
    else:
        ret = skew_symmetry(M.transpose(0, 2, 1).reshape([-1, 3]))
        ret = ret.reshape([-1, 3, 3, 3]).transpose(0, 2, 1, 3).reshape([-1, 3, 9])
        return ret


def Pentacle(M):
    assert M.ndim == 2 or M.ndim == 3
    if M.ndim == 2:
        return M.flatten('F')
    else:
        # return np.array([Pentacle(M[i]) for i in range(M.shape[0])])
        permute_index = list(range(M.ndim))
        permute_index[-1], permute_index[-2] = permute_index[-2], permute_index[-1]
        M_trans = M.transpose(permute_index)

        return M_trans.reshape([*M_trans.shape[:-2], -1])


def Sqrt(M):
    assert M.ndim == 2 or M.ndim == 3
    if M.ndim == 2:
        try:
            return np.linalg.cholesky(M)
        except:
            return np.zeros(M.shape)
    else:
        # return np.array([Sqrt(m) for m in M])

        ### M = d @ d.T ###
        d = np.zeros_like(M)
        d[:, 0, 0] = np.sqrt(M[:, 0, 0])
        d[:, 1, 0] = M[:, 1, 0] / d[:, 0, 0]
        d[:, 1, 1] = np.sqrt(M[:, 1, 1] - d[:, 1, 0] ** 2)
        d[:, 2, 0] = M[:, 2, 0] / d[:, 0, 0]
        d[:, 2, 1] = (M[:, 2, 1] - d[:, 2, 0] * d[:, 1, 0]) / d[:, 1, 1]
        d[:, 2, 2] = np.sqrt(M[:, 2, 2] - d[:, 2, 0] ** 2 - d[:, 2, 1] ** 2)
        d = np.transpose(d, axes=[0, 2, 1])
        return d


def diag_n(M, n):
    assert M.ndim == 2 or M.ndim == 3
    if M.ndim == 2:
        height, width = M.shape[-2:]
        ret = np.zeros((height * n, width * n))
        for i in range(n):
            ret[i * height:(i + 1) * height, i * width:(i + 1) * width] = M
        return ret
    elif M.ndim == 3:
        height, width = M.shape[-2:]
        ret = np.zeros((M.shape[0], height * n, width * n))
        for i in range(n):
            ret[:, i * height:(i + 1) * height, i * width:(i + 1) * width] = M
        return ret


def get_H_x_c(x_ref, varphi):
    varphi = varphi if varphi.ndim == 2 else varphi.reshape((1, -1))
    p_ref = x_ref[:3].copy()
    theta_ref = x_ref[5:8].copy()
    ext = x_ref[11:13].copy()

    R_ref = Rt.from_rotvec(theta_ref).as_matrix()

    varpi_S = (R_ref @ varphi.T).T + p_ref

    f_dx, f_dy = K[0, 0], K[1, 1]
    p_varpi_I_p_varpi_S = np.zeros([len(varphi), 2, 3])
    p_varpi_I_p_varpi_S[:, 0, 0] = f_dx / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 0, 2] = -varpi_S[:, 0] * f_dx / varpi_S[:, 2] ** 2
    p_varpi_I_p_varpi_S[:, 1, 1] = f_dy / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 1, 2] = -varpi_S[:, 1] * f_dy / varpi_S[:, 2] ** 2

    p_R_theta_p_delta_theta = -skew_symmetry((R_ref @ varphi.T).T) @ get_J(theta_ref)

    varphi_scale = varphi[:, :2] / ext
    varphi_scale_matrix = np.zeros([len(varphi), 3, 2])
    varphi_scale_matrix[:, 0, 0] = varphi_scale[:, 0]
    varphi_scale_matrix[:, 1, 1] = varphi_scale[:, 1]
    p_varpi_S_p_ext = R_ref @ varphi_scale_matrix

    p_varpi_S_p_delta_x = np.zeros([len(varphi), 3, 13])
    p_varpi_S_p_delta_x[:, :, :3] = np.tile(np.identity(3), reps=(len(varphi), 1, 1))
    p_varpi_S_p_delta_x[:, :, 5:8] = p_R_theta_p_delta_theta
    p_varpi_S_p_delta_x[:, :, 11:13] = p_varpi_S_p_ext

    return p_varpi_I_p_varpi_S @ p_varpi_S_p_delta_x


def get_H_varpi_c(x_ref, varphi):
    varphi = varphi if varphi.ndim == 2 else varphi.reshape((1, -1))
    p_ref = x_ref[:3].copy()
    theta_ref = x_ref[5:8].copy()

    R_ref = Rt.from_rotvec(theta_ref).as_matrix()

    varpi_S = (R_ref @ varphi.T).T + p_ref

    f_dx, f_dy, u_0, v_0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_varpi_I_p_varpi_S = np.zeros([len(varphi), 2, 3])
    p_varpi_I_p_varpi_S[:, 0, 0] = f_dx / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 0, 2] = -varpi_S[:, 0] * f_dx / varpi_S[:, 2] ** 2
    p_varpi_I_p_varpi_S[:, 1, 1] = f_dy / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 1, 2] = -varpi_S[:, 1] * f_dy / varpi_S[:, 2] ** 2

    return p_varpi_I_p_varpi_S @ R_ref


def get_H_ext_c(x_ref, varphi, ext):
    varphi = varphi if varphi.ndim == 2 else varphi.reshape((1, -1))
    p_ref = x_ref[:3].copy()
    theta_ref = x_ref[5:8].copy()

    R_ref = Rt.from_rotvec(theta_ref).as_matrix()

    # 计算变换点坐标
    varpi_S = (R_ref @ varphi.T).T + p_ref

    f_dx, f_dy, u_0, v_0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_varpi_I_p_varpi_S = np.zeros([len(varphi), 2, 3])
    p_varpi_I_p_varpi_S[:, 0, 0] = f_dx / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 0, 2] = -varpi_S[:, 0] * f_dx / varpi_S[:, 2] ** 2
    p_varpi_I_p_varpi_S[:, 1, 1] = f_dy / varpi_S[:, 2]
    p_varpi_I_p_varpi_S[:, 1, 2] = -varpi_S[:, 1] * f_dy / varpi_S[:, 2] ** 2

    varphi_scale = varphi[:, :2] / ext
    varphi_scale_matrix = np.zeros([len(varphi), 3, 2])
    varphi_scale_matrix[:, 0, 0] = varphi_scale[:, 0]
    varphi_scale_matrix[:, 1, 1] = varphi_scale[:, 1]
    p_varpi_S_p_ext = R_ref @ varphi_scale_matrix

    return p_varpi_I_p_varpi_S @ p_varpi_S_p_ext


def h_c(x_ref, varpi):
    varpi = varpi if varpi.ndim == 2 else varpi.reshape((1, -1))
    p_ref = x_ref[:3].copy()
    theta_ref = x_ref[5:8].copy()

    R_ref = Rt.from_rotvec(theta_ref).as_matrix()

    varpi_S = (R_ref @ varpi.T).T + p_ref
    varpi_I = K @ varpi_S.T
    varpi_I /= varpi_I[2, :]
    varpi_I = varpi_I[:2, :].T

    return varpi_I


def update(Theta, z_r, z_c):
    if len(z_c) == 0:
        return Theta
    z_r = z_r if z_r.ndim == 2 else z_r.reshape((1, -1))
    assert z_c.ndim == 2

    s_i = z_c[:, 0].astype(np.int64)

    ground_det_id = np.intersect1d(ground_id, s_i)
    ground_det_id_z = np.where(np.isin(s_i, ground_det_id))[0]

    corner_det_id = np.intersect1d(corner_id, s_i)
    corner_det_id_z = np.where(np.isin(s_i, corner_det_id))[0]

    s_i_skeleton = np.intersect1d(skeleton_knots_id, s_i)
    s_i_skeleton_z = np.where(np.isin(s_i, s_i_skeleton))[0]

    ret = Theta.copy()

    for it in range(N_iter):
        m = ret.m
        x_ref = ret.x_ref
        p_ref = x_ref[:3]
        theta = x_ref[5:8]
        omega = x_ref[8:11]
        ext = x_ref[11:13]

        R = Rt.from_rotvec(theta).as_matrix()
        R_line = Star(Sqrt(Q_inv)).T @ R
        J = get_J(theta)
        P = ret.P
        P_inv = np.linalg.inv(P)
        Cov_p = P[:3, :3]
        Cov_theta = P[4:7, 4:7]
        Cov_theta_p = P[4:7, 0:3]

        mu = ret.mu
        E_u = mu[:, :3]
        Sigma = ret.Sigma
        Sigma_inv = np.linalg.inv(Sigma)
        Cov_u = Sigma[:, :3, :3]
        Cov_u_inv = np.linalg.inv(Cov_u)

        # Pseudo measurement
        varpi_sym = mu[s_i_skeleton, 3:6]

        p_ref_norm = np.linalg.norm(p_ref)
        E_u_norm = np.linalg.norm(E_u, axis=1)

        R_E_u = R @ E_u.T
        dot_product = p_ref @ R_E_u

        # 计算分母并防止除以零
        denom = p_ref_norm * E_u_norm
        E_log_pi = -dot_product / denom - 1
        E_zeta = p_ref + (R @ E_u.T).T
        # 分解复杂计算，优化计算顺序
        term1 = (Q_inv @ Cov_p).trace()

        R_Q_R = R.T @ Q_inv @ R
        R_Q_R_Cov = R_Q_R @ Cov_u

        # 计算J @ Cov_theta @ J.T
        J_Cov_J = J @ Cov_theta @ J.T
        diag_term = diag_n(J_Cov_J, 3)

        # 计算E_u的外积
        E_u_outer = E_u[:, :, np.newaxis] @ E_u[:, np.newaxis, :]
        Cov_plus_outer = Cov_u + E_u_outer

        # 计算skew_symmetry项
        R_E_u = (R @ E_u.T).T
        skew_term = skew_symmetry(R_E_u)
        Q_skew_J = 2 * Q_inv @ skew_term @ J @ Cov_theta_p

        # 组合所有项并计算trace
        combined_term = (R_Q_R_Cov + R_line.T @ diag_term @ R_line @ Cov_plus_outer - Q_skew_J).transpose((1, 2, 0))
        term2 = combined_term.trace()
        E_zeta_Q = term1 + term2
        diff = z_r[:, np.newaxis, :] - E_zeta

        # 重塑为矩阵乘法所需形状
        diff_reshaped1 = diff.reshape((-1, 1, 3))
        diff_reshaped2 = diff.reshape((-1, 3, 1))

        # 计算二次型，优化计算顺序
        quad_form = (diff_reshaped1 @ Q_inv @ diff_reshaped2).reshape((-1, N_T))
        # 计算最终结果
        E_z_zeta_Q = quad_form + E_zeta_Q

        sub = E_log_pi - 1 / 2 * E_z_zeta_Q
        # sub = np.tile(E_log_pi, [len(z_r), 1])

        sub_t = sub[:, np.newaxis, :] - sub[:, :, np.newaxis]
        exp_sub_t = np.exp(sub_t)
        upsilon = 1 / exp_sub_t.sum(axis=2)

        # Calculate E_A
        n_t = upsilon.sum(axis=0)

        # set all n_t value bigger than eps
        n_t[n_t < eps] = eps

        z_line = np.sum(upsilon.T[:, :, np.newaxis] * z_r, axis=1) / n_t[:, np.newaxis]
        # Calculate parameters for pi
        m_new = n_t + m

        # Calculate parameters for x
        Q_n_inv = n_t[:, np.newaxis, np.newaxis] * Q_inv
        J_tilde = Star(Sqrt(Q_n_inv)).transpose((0, 2, 1)) @ J
        H_1_x = H_r - skew_symmetry((R @ E_u.T).T) @ J @ H_theta
        h_1_x = z_line - (R @ E_u.T).T - p_ref
        Q_x_inv = J_tilde.transpose((0, 2, 1)) @ diag_n(R @ Cov_u @ R.T, 3) @ J_tilde

        Q_x = np.linalg.inv(Q_x_inv)
        h_2_x = (Q_x.transpose((0, 2, 1)) @ J.T @ Star(Q_n_inv) @ Pentacle(R @ Cov_u @ R.T)[:, :, np.newaxis]).reshape([-1, 3])

        corner_det_rel_pos = np.array([np.append(keypoint_id_to_extend[idx] * ext, 0) for idx in corner_det_id])
        H_x_c = get_H_x_c(x_ref, corner_det_rel_pos)
        h_3_x = z_c[corner_det_id_z, 1:3] - h_c(x_ref, corner_det_rel_pos)
        H_4_x = ((R @ u_d) @ H_omega - omega @ skew_symmetry(R @ u_d) @ J @ H_theta).reshape([1, -1])
        h_4_x = -(R @ u_d) @ omega

        varphi_scale = corner_det_rel_pos[:, :2] / ext
        varphi_scale_matrix = np.zeros([len(corner_det_rel_pos), 3, 2])
        varphi_scale_matrix[:, 0, 0] = varphi_scale[:, 0]
        varphi_scale_matrix[:, 1, 1] = varphi_scale[:, 1]
        p_varpi_S_p_ext = R @ varphi_scale_matrix

        H_5_x = np.zeros([N_T + 8, 1, 13])
        h_5_x = np.zeros([N_T + 8, 1])
        H_5_x[corner_det_id, :, :3] = n_ground
        H_5_x[corner_det_id, :, 5:8] = -np.tile(n_ground, [len(corner_det_id), 1, 1]) @ skew_symmetry((R @ corner_det_rel_pos.T).T) @ J
        H_5_x[corner_det_id, :, 11:13] = n_ground[np.newaxis, np.newaxis, :] @ p_varpi_S_p_ext
        h_5_x[corner_det_id] = -((n_ground @ ((R @ corner_det_rel_pos.T).T + p_ref).T).T + d_ground).reshape([-1, 1])

        P_inv_sum = P_inv + np.sum(H_1_x.transpose((0, 2, 1)) @ Q_n_inv @ H_1_x + H_theta.T @ Q_x_inv @ H_theta,axis=0) + np.sum(H_x_c.transpose((0, 2, 1)) @ V_c_inv @ H_x_c,axis=0) + H_4_x.T @ Q_rot_inv @ H_4_x + np.sum(H_5_x.transpose((0, 2, 1)) @ Q_ground_inv @ H_5_x, axis=0)

        P_new = np.linalg.inv(P_inv_sum)
        dx_new = (P_new @ (np.sum(
        H_1_x.transpose((0, 2, 1)) @ Q_n_inv @ h_1_x[:, :, np.newaxis] + H_theta.T @ Q_x_inv @ h_2_x[:, :, np.newaxis],axis=0) + np.sum(H_x_c.transpose((0, 2, 1)) @ V_c_inv @ h_3_x[:, :, np.newaxis],axis=0) + H_4_x.T @ Q_rot_inv * h_4_x + np.sum(
        H_5_x.transpose((0, 2, 1)) @ Q_ground_inv * h_5_x[:, :, np.newaxis], axis=0))).flatten()

        x_ref_new = x_ref + dx_new
        dx_new = np.zeros_like(dx_new)

        # Calculate parameters for vartheta
        R_tilde = Star(Sqrt(Q_n_inv)).transpose((0, 2, 1)) @ R
        H_1_vartheta = R @ H_u
        h_1_vartheta = z_line - p_ref
        
        # 添加正则化确保矩阵可逆
        Q_vartheta_inv = R_tilde.transpose((0, 2, 1)) @ diag_n(J @ Cov_theta @ J.T, 3) @ R_tilde
        # 计算逆矩阵
        Q_vartheta = np.linalg.inv(Q_vartheta_inv)
        
        h_2_vartheta = (-Q_vartheta.transpose((0, 2, 1)) @ R.T @ Star(Q_n_inv) @ Pentacle(J @ Cov_theta_p)[np.newaxis, :, np.newaxis]).reshape([-1, 3])

        H_varpi_c = get_H_varpi_c(x_ref, mu[s_i_skeleton, 3:6])
        H_3_vartheta = np.zeros([N_T, 2, 9])
        H_3_vartheta[s_i_skeleton] = H_varpi_c @ H_varpi
        h_3_vartheta = np.zeros([N_T, 2])
        h_3_vartheta[s_i_skeleton] = z_c[s_i_skeleton_z, 1:3] - h_c(x_ref, mu[s_i_skeleton, 3:6]) + (H_varpi_c @ mu[s_i_skeleton, 3:6][:, :, np.newaxis]).squeeze()

        H_4_vartheta = H_varpi
        h_4_vartheta = np.zeros([N_T, 3])
        h_4_vartheta[flip_id[s_i_skeleton]] = (D @ varpi_sym.T).T
        Q_sym_inv = np.zeros([N_T, 3, 3])
        Q_sym_inv[flip_id[s_i_skeleton]] = Sigma_inv[s_i_skeleton, :3, :3]

        Sigma_new = np.linalg.inv(Sigma_inv + H_1_vartheta.T @ Q_n_inv @ H_1_vartheta + H_u.T @ Q_vartheta_inv @ H_u + H_3_vartheta.transpose((0, 2, 1)) @ V_c_inv @ H_3_vartheta + H_4_vartheta.T @ Q_sym_inv @ H_4_vartheta)
        mu_new = (Sigma_new @ (Sigma_inv @ mu[:, :, np.newaxis] + H_1_vartheta.T @ Q_n_inv @ h_1_vartheta[:, :,np.newaxis] + H_u.T @ Q_vartheta_inv @ h_2_vartheta[:,:,np.newaxis] + H_3_vartheta.transpose((0, 2, 1)) @ V_c_inv @ h_3_vartheta[:, :, np.newaxis] + H_4_vartheta.T @ Q_sym_inv @ h_4_vartheta[:, :,np.newaxis])).reshape([-1, 9])

        # Update all parameters
        ret.x_ref = x_ref_new
        ret.dx = dx_new
        ret.P = P_new
        ret.m = m_new
        ret.mu = mu_new
        ret.Sigma = Sigma_new

    return ret
