import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error

# Main softImpute-ALS Algorithm
def softImpute_als(X, rank, lambda_reg, max_iter=100, tol=1e-4):
    """
    softImpute-ALS algorithm for matrix completion
    
    Parameters:
    - X: Matrix with missing values (use np.nan for missing values)
    - rank: Desired rank of the completed matrix
    - lambda_reg: Regularization parameter
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - A, B: Low-rank factors of the matrix
    """
    # Initialize matrices A and B with random values
    m, n = X.shape
    A = np.random.rand(m, rank)
    B = np.random.rand(n, rank)
    
    # Identify observed entries
    observed = ~np.isnan(X)
    
    for iteration in range(max_iter):
        # Update A by minimizing the objective w.r.t B
        for i in range(m):
            B_i = B[observed[i, :], :]
            X_i = X[i, observed[i, :]]
            A[i, :] = np.linalg.solve(B_i.T @ B_i + lambda_reg * np.eye(rank), B_i.T @ X_i.T)
        
        # Update B by minimizing the objective w.r.t A
        for j in range(n):
            A_j = A[observed[:, j], :]
            X_j = X[observed[:, j], j]
            B[j, :] = np.linalg.solve(A_j.T @ A_j + lambda_reg * np.eye(rank), A_j.T @ X_j)
        
        # Compute RMSE on observed entries
        X_hat = A @ B.T
        rmse = np.sqrt(mean_squared_error(X[observed], X_hat[observed]))
        
        # Check for convergence
        if iteration > 0 and abs(prev_rmse - rmse) < tol:
            print(f"Converged at iteration {iteration} with RMSE: {rmse}")
            break
        
        prev_rmse = rmse
    
    return A, B

