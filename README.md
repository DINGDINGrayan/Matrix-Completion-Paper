# README for ALS and SoftImpute-ALS Implementation

## Overview
This repository contains an implementation of the Alternating Least Squares (ALS) and SoftImpute-ALS algorithms for matrix completion tasks. The project demonstrates their convergence behavior using synthetic data and plots their Root Mean Square Error (RMSE) as a function of the number of iterations.

## Features
- **ALS Algorithm**: Implements matrix completion using Alternating Least Squares.
- **SoftImpute-ALS Algorithm**: Combines the soft-thresholding technique with ALS for matrix completion.
- **Synthetic Data**: Uses randomly generated data with missing entries for testing.
- **Visualization**: Plots the convergence of both algorithms with RMSE over iterations.

## Files
- `main.py`: Responsible for running experiments.
- `visualization.py`: Responsible for running experiments, visualizing results, and generating plots.
- `README.md`: This documentation file.

## Requirements
- Python 3.7+
- Required Python packages (install via `pip`):
  ```bash
  pip install numpy matplotlib
  ```

## Running the Code
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Run the experiment script to execute both algorithms and generate results:
   ```bash
   python main.py
   ```
3. Results will be saved in the `results/` directory, including plots comparing the convergence of the algorithms.

## Output
- **Convergence Plot**: Displays the RMSE vs. Iterations for both ALS and SoftImpute-ALS.
- **RMSE Value**: Reports the RMSE achieved by both algorithms and the number of iterations required for convergence.

## Synthetic Data
The dataset is generated with the following characteristics:
- Matrix size: `100 x 100`
- Missing value ratio: `90%`
- Random seed for reproducibility.

## Acknowledgments
This project was created to illustrate matrix completion techniques using ALS and SoftImpute-ALS algorithms. The results highlight their strengths and weaknesses when applied to synthetic datasets.

For further inquiries or suggestions, feel free to reach out!

---

**Disclaimer**: This implementation is for educational purposes only and is not optimized for large-scale datasets.
