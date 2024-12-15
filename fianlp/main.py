import numpy as np
import os
from softimpute_als import softimpute_als
from generate_synthetic import generate_synthetic_data


X = generate_synthetic_data(500, 300, rank=10, missing_percentage=0.9)
print("Synthetic Data Generated.")

# Apply softImpute-ALS algorithm
A, B = softimpute_als(X, rank=10, lambda_reg=0.05)

if not os.path.exists('results'):
        os.makedirs('results')
# Save the low-rank factors A and B for evaluation
np.save("results/A.npy", A)
np.save("results/B.npy", B)

# Evaluate the results (you can also use the evaluate.py script)
observed = ~np.isnan(X)
X_hat = A @ B.T
rmse = np.sqrt(np.mean((X[observed] - X_hat[observed])**2))
print(f"RMSE: {rmse}")

