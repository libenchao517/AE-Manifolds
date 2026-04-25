############################################################
# 本文件用于实现线性流形学习方法的一些工具函数
############################################################
# 导入模块
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
from tqdm import tqdm
############################################################
# 定义函数
def init_projection(data, n_components):
    """
    初始化投影矩阵A
    :param data: 数据矩阵 [N, D]
    :param n_components: 低维维度 int
    :return: 投影矩阵 [D, d]
    """
    A = np.random.randn(data.shape[1], n_components)
    A = normalize(A, norm='l2', axis=1)
    return A

def adapt_lr(cost_change):
    """
    自适应选择学习率
    :param cost_change: 损失值变化 float
    :return: 学习率 float
    """
    if cost_change < 1:
        lr = 0.0000001
    elif cost_change < 10:
        lr = 0.000001
    elif cost_change < 100:
        lr = 0.00001
    elif cost_change < 1000:
        lr = 0.0001
    elif cost_change < 10000:
        lr = 0.001
    else:
        lr = 0.01
    return lr

def compute_centers(data, target):
    """
    计算各类数据的中心点
    :param data: 数据矩阵 [N, D]
    :param target: 数据标签 [N,]
    :return: 数据中心 [C, D]
    """
    F = []
    for t in np.unique(target):
        data_t = data[t == target]
        f_t = np.mean(data_t, axis=0)
        F.append(f_t)
    return np.array(F)

def compute_weight_for_lpp(data, neighbors, sigma=1):
    """
    计算LPP算法的权重矩阵W
    :param data: 数据矩阵 [N, D]
    :param neighbors: 图的邻居数 int
    :param sigma: 超参数 float
    :return: 权重矩阵 [N, N]
    """
    if neighbors >= data.shape[0]:
        neighbors = data.shape[0] - 1
    dist = pairwise_distances(data)
    nbrs_ = NearestNeighbors(n_neighbors=neighbors, metric="precomputed")
    nbrs_.fit(dist)
    W = kneighbors_graph(nbrs_, neighbors, metric="precomputed", mode='distance')
    W.data = W.data ** 2
    W.data /= np.max(W.data)
    W.data = np.exp((-W.data) / (2 * sigma ** 2))
    W = W.toarray()
    return W

def compute_knn(data, neighbors):
    """
    计算数据data的K近邻图
    :param data: 数据矩阵 [N, D]
    :param neighbors: 图的邻居数 int
    :return: 近邻距离和索引
    """
    nbrs_ = NearestNeighbors(n_neighbors=neighbors+1)
    nbrs_.fit(data)
    Q_distances = []
    Q_indices = []
    for i in range(len(data)):
        distance, index = nbrs_.kneighbors([data[i]])
        Q_distances.append(distance[0][1:])
        Q_indices.append(index[0][1:])
    return np.array(Q_distances), np.array(Q_indices)

def compute_weight_for_npe(data, neighbors, alpha=1e-5):
    """
    计算NPE算法的权重矩阵
    :param data: 数据矩阵 [N, D]
    :param neighbors: 图的邻居数 int
    :param alpha: 超参数 int
    :return: 权重矩阵 [N, N]
    [1] Pan S J, Wan L C, Liu H L, et al. Quantum algorithm for neighborhood
    preserving embedding[J]. Chinese Physics B, 2022, 31(6): 060304.
    [2] https://github.com/thomasverardo/NPE
    """
    if neighbors >= data.shape[0]:
        neighbors = data.shape[0] - 1
    _, indices = compute_knn(data, neighbors)
    W = []
    I = np.ones(neighbors)
    for i in range(len(data)):
        xi = data[i]
        C = []
        for j in range(neighbors):
            xj = data[indices[i][j]]
            C_aux = []
            for m in range(neighbors):
                xm = data[indices[i][m]]
                C_jk = (xi - xj).T @ (xi - xm)
                C_aux.append(C_jk)
            C.append(C_aux)
        C = np.array(C)
        C = C + alpha * np.eye(*C.shape)
        w = np.linalg.inv(C) @ I
        w = w / (I.T @ np.linalg.inv(C) @ I)
        w_zeros = np.zeros(len(data))
        np.put(w_zeros, indices[i], w)
        W.append(w_zeros)
    W = np.array(W)
    return W

def compute_within_weight(weight_func, data, target, neighbors, *args):
    """
    计算判别方法的类内权重矩阵W
    :param weight_func: 权重计算方法
    :param data: 数据矩阵 [N, D]
    :param target: 数据标签 [N,]
    :param neighbors: 图的邻居数 int
    :param args: 超参数
    :return: 类内权重矩阵 [N, N]
    """
    W = []
    for t in np.unique(target):
        data_t = data[t == target]
        W_t = weight_func(data_t, neighbors, *args)
        W.append(W_t)
    return block_diag(*W)

def compute_between_weight(weight_func, F, neighbors, *args):
    """
    计算判别方法的类间权重矩阵B
    :param weight_func: 权重计算方法
    :param F: 数据中心矩阵 [C, D]
    :param neighbors: 图的邻居数 int
    :param args: 超参数
    :return: 类间权重矩阵 [C, C]
    """
    return weight_func(F, neighbors, *args)

def compute_P(A, B):
    """
    计算A^TBA
    :param A: [N, D]
    :param B: [N, N]
    :return: A^TBA [D, D]
    """
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    B = B / np.linalg.norm(B, 'fro')
    BP = np.linalg.multi_dot([A.T, B, A])
    BP = (BP + BP.T) / 2
    BP = np.nan_to_num(BP, nan=0.0, posinf=0.0, neginf=0.0)
    return BP

def optimization(
        cost_func, gradient,
        X, A, epochs, converged_tol, lr, rho,
        adapt_lr_func,
        *args, **kwargs):
    """
    线性流形学习的优化器
    :param cost_func: 损失函数
    :param gradient: 梯度计算方法
    :param X: 数据矩阵 [N, D]
    :param A: 投影矩阵 [D, d]
    :param epochs: 最大迭代次数 int
    :param converged_tol: 终止条件 float
    :param lr: 学习率 float
    :param rho: 动量系数 float
    :param adapt_lr_func: 自适应的学习率函数
    :param args: 损失项的系数
    :param kwargs: DP、LP、HP、MP
    :return: 优化的投影矩阵和损失序列
    """
    costs = [0]
    V = np.zeros_like(A)
    args = list(args)
    for _ in tqdm(range(epochs)):
        cost_current, cost_sub = cost_func(X, A, *args, **kwargs)
        costs.append(cost_current)
        A = A + rho * V
        deltaL = gradient(X, A, *args, **kwargs)
        V = rho * V - lr * deltaL
        A = A + V
        for i in range(len(args)):
            args[i] = args[i] - lr * cost_sub[i] / sum(cost_sub)

        cost_change = abs(costs[-1] - costs[-2])
        lr = adapt_lr_func(cost_change)
        if cost_change < converged_tol:
            break
    A = normalize(A, norm='l2', axis=0)
    costs = np.array(costs[1:])
    return A, costs
