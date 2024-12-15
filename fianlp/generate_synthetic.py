import numpy as np


def generate_synthetic_data(m, n, rank, missing_percentage=0.9, seed=42):
    """
    Generates a synthetic low-rank matrix with missing entries.

    Parameters:
        m: Number of rows
        n: Number of columns
        rank: Rank of the matrix
        missing_percentage: Fraction of missing entries
        seed: Random seed for reproducibility

    Returns:
        X: Synthetic matrix with missing entries as np.nan
    """
    np.random.seed(seed)
    # Generate low-rank matrix
    A = np.random.rand(m, rank)
    B = np.random.rand(n, rank)
    X = A @ B.T

    # Introduce missing entries
    num_missing = int(m * n * missing_percentage)
    missing_indices = np.random.choice(m * n, num_missing, replace=False)
    X.flat[missing_indices] = np.nan

    return X


if __name__ == "__main__":
    # Generate synthetic data and save to a file
    synthetic_data = generate_synthetic_data(500, 300, rank=10, missing_percentage=0.9)
    np.save("synthetic_data.npy", synthetic_data)
    print("Synthetic data saved to data/synthetic_data.npy")
