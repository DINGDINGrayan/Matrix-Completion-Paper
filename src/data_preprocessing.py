import numpy as np
import pandas as pd

# Function to load and preprocess the MovieLens dataset
def load_movielens_data(file_path):
    data = pd.read_csv(file_path)
    matrix = np.zeros((943, 1682))
    
    for _, row in data.iterrows():
        matrix[int(row['userId'])-1, int(row['movieId'])-1] = row['rating']
    
    return matrix

# Function to create synthetic data
def create_synthetic_data(m, n, rank, missing_percentage=0.9):
    np.random.seed(0)
    A = np.random.rand(m, rank)
    B = np.random.rand(n, rank)
    X = A @ B.T
    
    # Introduce missing values
    num_missing = int(m * n * missing_percentage)
    missing_indices = np.random.choice(m * n, num_missing, replace=False)
    
    X.flat[missing_indices] = np.nan  # Set missing values as NaN
    
    return X

