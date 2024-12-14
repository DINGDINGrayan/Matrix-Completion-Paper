import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_rmse(X_true, X_pred, observed):
    """
    Evaluate RMSE between the true and predicted matrices.
    Only considers the observed entries (non-NaN).
    """
    return np.sqrt(mean_squared_error(X_true[observed], X_pred[observed]))

