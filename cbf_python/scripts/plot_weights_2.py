import pandas as pd
import matplotlib.pyplot as plt

# --- Parameterized Font Dimensions and Text ---
figure_title_text = "Evaluation of the parameter sets in different scenarios"
figure_title_fontsize = 20
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 15
legend_title_fontsize = 18
# ----------------------------------------------

# 1. Load the data (using local path as files are stored in the current working directory)
df = pd.read_csv("../resullts/simulation_data_dynamic_params_comparison.csv")

# 2. Define the exact metrics requested
metrics = [
    'on_target_rate',
    'lap_count',
    'mean_trajectory_error'
]
titles = [
    'On-Target Rate',
    'Lap Count',
    'Mean Trajectory Error'
]
x_values = [-0.16235027545036934, 0.16709220387410248, 0.3656552506963782]
legend_vertical_values = ["optimization value for h = -0.1", "optimization value for h = 0.2",
                          "optimization value for h = 0.4"]

# 3. Extract unique test types dynamically
test_types = df['test_type'].unique()

# 4. Initialize a 3x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Add the overall figure title
fig.suptitle(figure_title_text, fontsize=figure_title_fontsize, fontweight='bold')

axes = axes.flatten()
colors = ["orange", "green", "red"]

# 5. Loop over the metrics and populate the first 5 subplots
for i, metric in enumerate(metrics):
    ax = axes[i]

    # Configure reference horizontal lines
    if metric in ['on_target_rate', 'mean_scale']:
        ref_value = 1.0
    elif metric in ['viol_rate', 'mean_trajectory_error']:
        ref_value = 0.0
    else:
        ref_value = 35.0
    if metric != "lap_count":
        ax.axhline(y=ref_value, color='purple', linestyle='--', linewidth=1.5, zorder=0, label='Reference value')

    # Plot each test type separately, locally sorting by h_mean_test in ascending order
    for tt in test_types:
        subset = df[df['test_type'] == tt].sort_values(by='h_mean_test')
        ax.plot(subset['h_mean_test'], subset[metric], marker='o', label=tt)

    # Plot the vertical reference lines using the correct index variable 'j'
    for j in range(len(legend_vertical_values)):
        ax.axvline(x=x_values[j], linestyle=':', linewidth=1.5, zorder=0, label=legend_vertical_values[j],
                   color=colors[j])

    # Apply parameterized fonts to titles and labels
    ax.set_title(titles[i], fontweight='bold', fontsize=title_fontsize)
    ax.set_xlabel('h [m]', fontsize=label_fontsize)
    ax.set_ylabel('Metric Value', fontsize=label_fontsize)

    # Apply parameterized fonts to axis ticks
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax.grid(True, linestyle='--', alpha=0.7)

# 6. Configure the 6th subplot (index 5 - wait, 3 in a 2x2 grid) exclusively for the legend
ax_legend = axes[3]
ax_legend.axis('off')

# Extract handles and labels to build the master legend without duplicates
handles, labels = axes[0].get_legend_handles_labels()
labels = ["Reference", "Dynamic parameters", "Set Optimized for case 1", "Set Optimized for case 2",
          "Set Optimized for case 3", "Case 1", "Case 2", "Case 3"]
by_label = dict(zip(labels, handles))

# Apply parameterized fonts to the legend
ax_legend.legend(by_label.values(), by_label.keys(), loc='center',
                 fontsize=legend_fontsize, title='Test Types & References',
                 title_fontsize=legend_title_fontsize)


# 7. Finalize layout and save as PDF
# rect=[left, bottom, right, top] -> top=0.98 reduces the space below the suptitle
# h_pad increases the vertical distance between subplot rows
plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3.0)
plt.savefig("metrics_subplots_h_mean.pdf")
plt.show()