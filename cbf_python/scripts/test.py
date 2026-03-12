import numpy as np
import matplotlib.pyplot as plt
from Controller.gaussian_controller import GaussianControllerConfig
from math import sqrt
from scripts.util.gaussian_process_util import read_config_data_from_csv

def plot_multiple_gaussians(cfg:GaussianControllerConfig):
    """
    Plots multiple Gaussian PDFs on the same figure.

    params_list: A list of tuples, where each tuple is (mean, std_dev)
    """
    plt.figure(figsize=(10, 6))

    # 1. Determine a good range for the X-axis based on all distributions
    # We want to go from the lowest mean - 4*std to the highest mean + 4*std
    params_list = []
    for gs in cfg.gaussian_sets:
        temp = (gs.means["h"], sqrt(gs.covariance[0][0]))
        params_list.append(temp)


    min_x = min([m - 4 * s for m, s in params_list])
    max_x = max([m + 4 * s for m, s in params_list])
    x = np.linspace(min_x, max_x, 1000)

    # Colors for the plots to make them distinct
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 2. Calculate and plot each Gaussian PDF
    for i, (mean, std_dev) in enumerate(params_list):
        # The mathematical formula for the Gaussian PDF
        pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

        # Plot the curve
        plt.plot(x, pdf, label=f'Mean: {mean}, Std Dev: {std_dev}',
                 color=colors[i % len(colors)], linewidth=2)

        # Optional: Fill under the curve slightly for better visualization
        plt.fill_between(x, pdf, alpha=0.1, color=colors[i % len(colors)])

    # 3. Add labels, title, and formatting
    plt.title('Gaussian Probability Density Functions for $h$', fontsize=14)
    plt.xlabel('Value of $h$', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)

    # Display the plot
    plt.tight_layout()
    plt.show()

cfg = GaussianControllerConfig()

# read_config_data_from_csv(cfg, filename="../log_best_trials.csv", h_mean=-0.1, v_mean=1)
# print("1")
# read_config_data_from_csv(cfg, filename="../log_best_trials.csv", h_mean=1, v_mean=1)
# print("2")
# read_config_data_from_csv(cfg, filename="../log_best_trials.csv", h_mean="0.5", v_mean=1)
# print("3")
# print(cfg)
# plot_multiple_gaussians(cfg)

import itertools

# First value: -0.1 to 1.0 (step 0.05)
val1_list = [round(-0.1 + i * 0.05, 2) for i in range(23)]  # 23 steps reach 1.0

# Second value: 0.2 to 1.4 (step 0.3)
val2_list = [round(0.2 + i * 0.3, 2) for i in range(5)]     # 5 steps reach 1.4

# Generate all combinations
combinations = list(itertools.product(val1_list, val2_list))

# Create the final dictionary
par_values = {i: list(comb) for i, comb in enumerate(combinations)}
print ( par_values)


