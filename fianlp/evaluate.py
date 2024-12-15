import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_rmse(X_true, X_pred, observed):
    """
    Computes RMSE for observed entries in the matrix.
    """
    return np.sqrt(mean_squared_error(X_true[observed], X_pred[observed]))


if __name__ == "__main__":
    # Load synthetic data and completed matrix
    X = np.load("synthetic_data.npy")
    observed = ~np.isnan(X)

    # Load the low-rank factors A and B
    A = np.load("results/A.npy")  # Low-rank factor A
    B = np.load("results/B.npy")  # Low-rank factor B

    # Reconstruct the completed matrix
    X_hat = A @ B.T
    rmse = evaluate_rmse(X, X_hat, observed)
    print(f"RMSE on observed entries: {rmse}")
