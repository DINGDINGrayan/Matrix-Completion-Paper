import numpy as np
from sklearn.metrics import mean_squared_error


def softimpute_als(X, rank, lambda_reg, max_iter=100, tol=1e-4):
    """
    Implements the softImpute-ALS algorithm for matrix completion.

    Parameters:
        X: 2D numpy array with missing values (use np.nan for missing entries)
        rank: Desired rank of the matrix approximation
        lambda_reg: Regularization parameter
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        A, B: Low-rank factors such that X ≈ AB.T
    """
    # Initialize matrices A and B with random values
    m, n = X.shape
    A = np.random.rand(m, rank)
    B = np.random.rand(n, rank)

    # Mask for observed entries
    observed = ~np.isnan(X)

    for iteration in range(max_iter):
        # Update A
        for i in range(m):
            B_i = B[observed[i, :], :]
            X_i = X[i, observed[i, :]]
            A[i, :] = np.linalg.solve(B_i.T @ B_i + lambda_reg * np.eye(rank), B_i.T @ X_i.T)

        # Update B
        for j in range(n):
            A_j = A[observed[:, j], :]
            X_j = X[observed[:, j], j]
            B[j, :] = np.linalg.solve(A_j.T @ A_j + lambda_reg * np.eye(rank), A_j.T @ X_j)

        # Compute RMSE on observed entries
        X_hat = A @ B.T
        rmse = np.sqrt(mean_squared_error(X[observed], X_hat[observed]))
        print(f"Iteration {iteration + 1}, RMSE: {rmse}")

        # Check for convergence
        if iteration > 0 and abs(prev_rmse - rmse) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        prev_rmse = rmse

    return A, B
