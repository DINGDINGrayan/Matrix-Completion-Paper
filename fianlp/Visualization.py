import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic import generate_synthetic_data


def plot_softImpute_als_rmse(iterations, rmse_values, label='softImpute-ALS'):
    """
    绘制 softImpute-ALS 算法的 RMSE 收敛图
    :param iterations: 迭代次数列表
    :param rmse_values: RMSE 值列表
    :param label: 曲线标签
    """
    plt.plot(iterations, rmse_values, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Convergence of softImpute-ALS')
    plt.legend()
    plt.show()


def plot_als_rmse(iterations, rmse_values, label='ALS'):
    """
    绘制 ALS 算法的 RMSE 收敛图
    :param iterations: 迭代次数列表
    :param rmse_values: RMSE 值列表
    :param label: 曲线标签
    """
    plt.plot(iterations, rmse_values, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Convergence of ALS')
    plt.legend()
    plt.show()


def als_algorithm(X, max_iter=100, tol=1e-4):
    """
    实现 ALS 算法
    :param X: 输入矩阵
    :param max_iter: 最大迭代次数
    :param tol: 收敛容忍度
    :return: 迭代次数列表和对应的 RMSE 值列表
    """
    A = np.random.rand(X.shape[0], 10)
    B = np.random.rand(X.shape[1], 10)
    iterations = []
    rmse_values = []
    for iteration in range(max_iter):
        # 更新 A
        for i in range(X.shape[0]):
            observed_i = ~np.isnan(X[i, :])
            B_i = B[observed_i, :]
            X_i = X[i, observed_i]
            A[i, :] = np.linalg.lstsq(B_i, X_i, rcond=None)[0]
        # 更新 B
        for j in range(X.shape[1]):
            observed_j = ~np.isnan(X[:, j])
            A_j = A[observed_j, :]
            X_j = X[observed_j, j]
            B[j, :] = np.linalg.lstsq(A_j, X_j, rcond=None)[0]
        # 计算 RMSE
        X_hat = A @ B.T
        observed = ~np.isnan(X)
        rmse = np.sqrt(np.mean((X[observed] - X_hat[observed]) ** 2))
        iterations.append(iteration + 1)
        rmse_values.append(rmse)
        # 检查收敛
        if iteration > 0 and abs(rmse_values[-1] - rmse_values[-2]) < tol:
            print(f"ALS Converged after {iteration + 1} iterations.")
            break
    return iterations, rmse_values


def softImpute_als_algorithm(X, max_iter=100, tol=1e-4):
    """
    实现 softImpute-ALS 算法
    :param X: 输入矩阵
    :param max_iter: 最大迭代次数
    :param tol: 收敛容忍度
    :return: 迭代次数列表和对应的 RMSE 值列表
    """
    A = np.random.rand(X.shape[0], 10)
    B = np.random.rand(X.shape[1], 10)
    iterations = []
    rmse_values = []
    for iteration in range(max_iter):
        observed = ~np.isnan(X)
        # 更新 A
        for i in range(X.shape[0]):
            B_i = B[observed[i, :], :]
            X_i = X[i, observed[i, :]]
            A[i, :] = np.linalg.solve(B_i.T @ B_i + 0.05 * np.eye(10), B_i.T @ X_i.T)
        # 更新 B
        for j in range(X.shape[1]):
            A_j = A[observed[:, j], :]
            X_j = X[observed[:, j], j]
            B[j, :] = np.linalg.solve(A_j.T @ A_j + 0.05 * np.eye(10), A_j.T @ X_j)
        # 计算 RMSE
        X_hat = A @ B.T
        rmse = np.sqrt(np.mean((X[observed] - X_hat[observed]) ** 2) )
        iterations.append(iteration + 1)
        rmse_values.append(rmse)
        # 检查收敛
        if iteration > 0 and abs(rmse_values[-1] - rmse_values[-2]) < tol:
            print(f"softImpute-ALS Converged after {iteration + 1} iterations.")
            break
    return iterations, rmse_values


# Generate synthetic data
X = generate_synthetic_data(500, 300, rank=10, missing_percentage=0.9)
print("Synthetic Data Generated.")


# 调用 softImpute-ALS 算法
softImpute_iterations, softImpute_rmse = softImpute_als_algorithm(X)


# 调用 ALS 算法
als_iterations, als_rmse = als_algorithm(X)


# 分别绘制 softImpute-ALS 和 ALS 的 RMSE 收敛图
plot_softImpute_als_rmse(softImpute_iterations, softImpute_rmse)
plot_als_rmse(als_iterations, als_rmse)