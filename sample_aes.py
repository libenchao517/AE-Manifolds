############################################################
# 本文件用于实现基于线性自编码器的流形学习方法
############################################################
# 导入模块
import numpy as np
from sklearn.preprocessing import normalize
from .utils import compute_weight_for_lpp
from .utils import compute_weight_for_npe
from .utils import compute_within_weight
from .utils import compute_between_weight
from .utils import init_projection
from .utils import adapt_lr
from .utils import compute_centers
from .utils import compute_P
from .utils import optimization
############################################################
class Autoencoder_LPP:
    """
    [1] Ruisheng Ran, Ji Feng, Zheng Li, Jinping Wang, Bin Fang.
    Locality Preserving Projections with Autoencoder[J].
    Expert Systems with Applications, 2024, 242: 122750.
    [2] Ruisheng Ran, Ting Wang, Wenfeng Zhang, Bin Fang.
    Autoencoder-based Discriminant Locality Preserving Projections for Fault Diagnosis[J].
    IEEE Transactions on Instrumentation and Measurement, 2025, 74 3526113.
    """
    def __init__(
            self,
            n_components=30,
            neighbors=10,
            epochs = 200,
            converged_tol = 0.1,
            lr=0.001,
            rho=0.6,
            lambda_=3,
            gamma=1,
            eta = 0.0,
            alpha = 1,
            sigma = 1
    ):
        """
        初始化函数
        :param n_components: 低维维度 int
        :param neighbors: 图的邻居数 int
        :param epochs: 最大迭代次数 int
        :param converged_tol: 终止条件 float
        :param lr: 学习率 float
        :param rho: 动量系数 float
        :param lambda_: LPP的约束项的系数 float
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param alpha: MMC的系数
        :param sigma: 权重系数 float
        """
        self.n_components = n_components
        self.neighbors = neighbors
        self.epochs = epochs
        self.converged_tol = converged_tol
        self.lr = lr
        self.rho = rho
        self.lambda_ = lambda_
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha
        self.sigma = sigma
        self.costs = None

    @staticmethod
    def compute_laplacian(X, W, F=None, B=None):
        """
        计算中间变量X^TDX、X^TLX、F^THF
        :param X: 数据矩阵 [N, D]
        :param W: (类内)权重矩阵 [N, N]
        :param F: 数据中心 [C, D]
        :param B: 类间权重矩阵 [C, C]
        :return: LP, XP [D, D], [D, D]
        """
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        LP = compute_P(X, L)
        if F is None:
            XP = compute_P(X, D)
        else:
            E = np.diag(np.sum(B, axis=1))
            H = E - B
            XP = compute_P(F, H)
        return LP, XP

    def cost_lpp(self, X, A, lambda_, gamma, eta, DP, LP):
        """
        LPP-AE的损失函数
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param lambda_: LPP的约束项的系数 float
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param DP: X^TDX [D, D]
        :param LP: X^TLX [D, D]
        :return: 损失值
        """
        cost1 = np.trace(np.linalg.multi_dot([A.T, LP, A]))
        cost2 = lambda_ * (np.trace(np.linalg.multi_dot([A.T, DP, A])) - self.n_components)
        cost3_ = np.dot(X, np.eye(X.shape[1])-np.dot(A, A.T).T)
        cost3 = np.trace(gamma * np.dot(cost3_.T, cost3_))
        cost4 = eta * np.trace(np.dot(A, A.T))
        cost = cost1 + cost2 + cost3 + cost4
        return cost, (cost2, cost3, cost4)

    @staticmethod
    def cost_dlpp(X, A, alpha, gamma, eta, HP, LP):
        """
        DLPP-AE的损失函数
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param alpha: MMC的系数
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param HP: F^THF [D, D]
        :param LP: X^TLX [D, D]
        :return: 损失值
        """
        cost1_ = (alpha * LP - HP)
        cost1 = np.trace(np.linalg.multi_dot([A.T, cost1_, A]))
        cost2_ = np.dot(X, (np.eye(X.shape[1]) - np.dot(A, A.T)).T)
        cost2 = np.trace(gamma * np.dot(cost2_.T, cost2_))
        cost3 = eta * np.trace(np.dot(A, A.T))
        upp = np.trace(np.linalg.multi_dot([A.T, LP, A]))
        cost = cost1 + cost2 + cost3
        return cost, (upp, cost2, cost3)

    @staticmethod
    def gradient_lpp(X, A, lambda_, gamma, eta, DP, LP):
        """
        计算LPP-AE的梯度
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param lambda_: LPP的约束项的系数 float
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param DP: X^TDX [D, D]
        :param LP: X^TLX [D, D]
        :return: 梯度 [D, d]
        """
        d1 = 2 * np.dot(LP, A)
        d2 = lambda_ * 2 * np.dot(DP, A)
        d3 = -4 * gamma * np.linalg.multi_dot([np.eye(X.shape[1]) - np.dot(A, A.T), X.T, X, A])
        d4 = 2 * eta * A
        deltaL = d1 + d2 + d3 + d4
        deltaL = normalize(deltaL, norm="l2", axis=1)
        return deltaL

    @staticmethod
    def gradient_dlpp(X, A, alpha, gamma, eta, HP, LP):
        """
        计算DLPP-AE的梯度
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param alpha: MMC的系数
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param HP: F^THF [D, D]
        :param LP: X^TLX [D, D]
        :return: 梯度 [D, d]
        """
        d1 = 2 * np.dot((alpha * LP - HP), A)
        d2 = -4 * gamma * np.linalg.multi_dot([np.eye(X.shape[1]) - np.dot(A, A.T), X.T, X, A])
        d3 = 2 * eta * A
        deltaL = d1 + d2 + d3
        deltaL = normalize(deltaL, norm='l2', axis=1)
        return deltaL

    def fit(self, data, target=None):
        """
        训练LPP-AE和DLPP-AE
        :param data: 数据矩阵 [N, D]
        :param target: 数据标签 [N,]
        :return: 投影矩阵 A [D, d]
        """
        A = init_projection(data, self.n_components)
        if target is None:
            W = compute_weight_for_lpp(data, self.neighbors, self.sigma)
            LP, DP = self.compute_laplacian(data, W)
            A, self.costs = optimization(
                self.cost_lpp, self.gradient_lpp,
                data, A, self.epochs, self.converged_tol, self.lr, self.rho,
                adapt_lr,
                self.lambda_, self.gamma, self.eta,
                DP=DP, LP=LP
            )
        else:
            F = compute_centers(data, target)
            W = compute_within_weight(compute_weight_for_lpp, data, target, self.neighbors, self.sigma)
            B = compute_between_weight(compute_weight_for_lpp, F, self.neighbors, self.sigma)
            LP, HP = self.compute_laplacian(data, W, F, B)
            A, self.costs = optimization(
                self.cost_dlpp, self.gradient_dlpp,
                data, A, self.epochs, self.converged_tol, self.lr, self.rho,
                adapt_lr,
                self.alpha, self.gamma, self.eta,
                HP=HP, LP=LP
            )
        return A

    def fit_transform(self, train, test, target=None):
        """
        对训练数据和测试数据进行嵌入
        :param train: 训练数据
        :param test: 测试数据
        :param target: 训练标签
        :return: 训练嵌入, 测试嵌入
        """
        A = self.fit(train, target)
        train_embed = np.dot(train, A)
        test_embed = np.dot(test, A)
        return train_embed, test_embed
############################################################
class Autoencoder_NPE:
    """
    [1] Ruisheng Ran, Jinping Wang, Bin Fang, Weiming Yang.
    Neighborhood Preserving Embedding with Autoencoder[J].
    Digital Signal Processing, 2024, 145: 104331.
    [2] Ting Wang, Yisha Xie.
    Discriminant Neighborhood Preserving Embedding with Autoencoder for fault diagnosis[C].
    In: International Conference on Robotics, Intelligent Control and Artificial Intelligence (RICAI). IEEE, 2024: 532-535.
    """
    def __init__(
            self,
            n_components=30,
            neighbors=10,
            epochs = 200,
            converged_tol = 0.1,
            lr=0.001,
            rho=0.6,
            lambda_=3,
            gamma=1,
            eta = 0.0,
            alpha=1,
            beta=1e-5,
    ):
        """
        初始化函数
        :param n_components: 低维维度 int
        :param neighbors: 图的邻居数 int
        :param epochs: 最大迭代次数 int
        :param converged_tol: 终止条件 float
        :param lr: 学习率 float
        :param rho: 动量系数 float
        :param lambda_: NPE的约束项的系数 float
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化系数 float
        :param alpha: MMC的系数
        :param beta: 权重系数 float
        """
        self.n_components = n_components
        self.neighbors = neighbors
        self.epochs = epochs
        self.converged_tol = converged_tol
        self.lr = lr
        self.rho = rho
        self.lambda_ = lambda_
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.costs = None

    @staticmethod
    def compute_media(X, W, F=None, B=None):
        """
        计算中间变量X^TMX、F^THF
        :param X: 数据矩阵 [N, D]
        :param W: (类内)权重矩阵 [N, N]
        :param F: 数据中心 [C, D]
        :param B: 类间权重矩阵 [C, C]
        :return: MP, HP [D, D], [D, D]
        """
        M_ = np.zeros(W.shape[0]) - W
        M = np.dot(M_.T, M_)
        MP = compute_P(X, M)
        HP = None
        if F is not None:
            H_ = np.zeros(B.shape[0]) - B
            H = np.dot(H_.T, H_)
            HP = compute_P(F, H)
        return MP, HP

    def cost_npe(self, X, A, lambda_, gamma, eta, MP):
        """
        NPE-AE的损失函数
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param lambda_: NPE的约束项的系数 float
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化参数 float
        :param MP: X^TMX [D, D]
        :return: 损失值
        """
        cost1 = np.trace(np.linalg.multi_dot([A.T, MP, A]))
        cost2 = lambda_ * np.trace(np.linalg.multi_dot([A.T, X.T, X, A])) - self.n_components
        cost3_ = np.dot(X, np.eye(X.shape[1])-np.dot(A, A.T).T)
        cost3 = np.trace(gamma * np.dot(cost3_.T, cost3_))
        cost4 = eta * np.trace(np.dot(A, A.T))
        cost = cost1 + cost2 + cost3 + cost4
        return cost, (cost2, cost3, cost4)

    @staticmethod
    def cost_dnpe(X, A, alpha, gamma, eta, HP, MP):
        """
        DNPE-AE的损失函数
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param alpha: MMC的系数
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化参数 float
        :param HP: HP: X^THX [D, D]
        :param MP: MP: X^TMX [D, D]
        :return: 损失值
        """
        cost1_ = (alpha * MP - HP)
        cost1 = np.trace(np.linalg.multi_dot([A.T, cost1_, A]))
        cost2_ = np.dot(X, (np.eye(X.shape[1]) - np.dot(A, A.T)).T)
        cost2 = np.trace(gamma * np.dot(cost2_.T, cost2_))
        cost3 = eta * np.trace(np.dot(A, A.T))
        upp = np.trace(np.linalg.multi_dot([A.T, MP, A]))
        cost = cost1 + cost2 + cost3
        return cost, (upp, cost2, cost3)

    @staticmethod
    def gradient_npe(X, A, lambda_, gamma, eta, MP):
        """
        计算NPE-AE的梯度
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param lambda_: 损失项2的系数 float
        :param gamma: 损失项3的系数 float
        :param eta: 正则化参数 float
        :param MP: X^TDX [D, D]
        :return: 梯度 [D, d]
        """
        d1 = 2 * np.dot(MP, A)
        d2 = 2 * lambda_ * np.linalg.multi_dot([X.T, X, A])
        d3 = -4 * gamma * np.linalg.multi_dot([np.eye(X.shape[1]) - np.dot(A, A.T), X.T, X, A])
        d4 = 2 * eta * A
        deltaL = d1 + d2 + d3 + d4
        deltaL = normalize(deltaL, norm="l2", axis=1)
        return deltaL

    @staticmethod
    def gradient_dnpe(X, A, alpha, gamma, eta, MP, HP):
        """
        计算DNPE-AE的梯度
        :param X: 数据矩阵 [N, D]
        :param A: 投影矩阵 [D, d]
        :param alpha: MMC的系数
        :param gamma: 自动编码器的损失系数 float
        :param eta: 正则化参数 float
        :param HP: HP: X^THX [D, D]
        :param MP: MP: X^TMX [D, D]
        :return: 梯度 [D, d]
        """
        d1 = 2 * np.dot((alpha * MP - HP), A)
        d2 = -4 * gamma * np.linalg.multi_dot([np.eye(X.shape[1]) - np.dot(A, A.T), X.T, X, A])
        d3 = 2 * eta * A
        deltaL = d1 + d2 + d3
        deltaL = normalize(deltaL, norm='l2', axis=1)
        return deltaL

    def fit(self, data, target=None):
        """
        训练NPE-AE和DNPE-AE
        :param data: 数据矩阵 [N, D]
        :param target: 数据标签 [N,]
        :return: 投影矩阵 A [D, d]
        """
        A = init_projection(data, self.n_components)
        if target is None:
            W = compute_weight_for_npe(data, self.neighbors, self.beta)
            MP, _ = self.compute_media(data, W)
            A, self.costs = optimization(
                self.cost_npe, self.gradient_npe,
                data, A, self.epochs, self.converged_tol, self.lr, self.rho,
                adapt_lr,
                self.lambda_, self.gamma, self.eta,
                MP = MP
            )
        else:
            F = compute_centers(data, target)
            W = compute_within_weight(compute_weight_for_npe, data, target, self.neighbors, self.beta)
            B = compute_between_weight(compute_weight_for_npe, F, self.neighbors, self.beta)
            MP, HP = self.compute_media(data, W, F, B)
            A, self.costs = optimization(
                self.cost_dnpe, self.gradient_dnpe,
                data, A, self.epochs, self.converged_tol, self.lr, self.rho,
                adapt_lr,
                self.alpha, self.gamma, self.eta,
                MP=MP, HP=HP
            )
        return A

    def fit_transform(self, train, test, target=None):
        """
        对训练数据和测试数据进行嵌入
        :param train: 训练数据
        :param test: 测试数据
        :param target: 训练标签
        :return: 训练嵌入, 测试嵌入
        """
        A = self.fit(train, target)
        train_embed = np.dot(train, A)
        test_embed = np.dot(test, A)
        return train_embed, test_embed
############################################################
class Autoencoder_LTSA:
    """
    [1] Ruisheng Ran, Jinping Wang, Bin Fang.
    Linear Local Tangent Space Alignment with Autoencoder[J].
    Complex & Intelligent Systems, 2023, 9(6): 6255-6268.
    """
    def __init__(self):
        pass
############################################################
class Autoencoder_IsoP:
    """
    [1] Ruisheng Ran, Qianghui Zeng, Xiaopeng Jiang, Bin Fang.
    Isometric Projection with Reconstruction[J].
    The Journal of Supercomputing, 2023, 79(16): 18648-18666.
    """
    def __init__(self):
        pass
