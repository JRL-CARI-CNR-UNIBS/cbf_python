import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_gaussians(params_list):
    """
    Plots multiple Gaussian PDFs on the same figure.

    params_list: A list of tuples, where each tuple is (mean, std_dev)
    """
    plt.figure(figsize=(10, 6))

    # 1. Determine a good range for the X-axis based on all distributions
    # We want to go from the lowest mean - 4*std to the highest mean + 4*std
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

st_dev = 0.15

# --- Example Usage ---
# Let's say these are three different scenarios for your target h
# Format: (mean, standard_deviation)
h_parameters = [
    (0.0, st_dev),  # Scenario A: Tight variance around 5m
    (0.5, st_dev),  # Scenario B: Wider variance around 8m
    (1.0, st_dev)  # Scenario C: Very strict variance around 3m
]




ref_std_dev = 0.1
ref_h_mean   = [x / 10.0 for x in range(-1, 11)]

print(ref_h_mean)