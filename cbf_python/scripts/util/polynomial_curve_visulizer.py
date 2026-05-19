import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# 1. Define the transition function for the interval [0, 1]
def calculate_lambda(eta, r, s, t, l_inf, l_sup):
    return l_inf + (l_sup - l_inf) * (r * eta ** s + (1 - r) * eta ** t)


# 2. Initial parameter configuration
init_r = 1.0
init_s = 2.0
init_t = 4.0
init_l_inf = 1.0
init_l_sup = 5.0

# Generate the eta array (from 0 to 1)
eta = np.linspace(0, 1, 500)

# 3. Setup the figure and the main plot axis
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.4)  # Make room for the sliders below

# Plot the initial curve
[line] = ax.plot(eta, calculate_lambda(eta, init_r, init_s, init_t, init_l_inf, init_l_sup),
                 linewidth=2, color='blue', label=r'$\lambda(\eta)$')

# Add the strict non-negativity boundary (y = 0)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Strict Non-negativity Constraint (y=0)')

# Configure axes
ax.set_xlim(0, 1)
ax.set_ylim(-2, 10)  # Set a fixed Y limit to easily see undershoots
ax.set_xlabel(r'$\eta$ (Normalized Transition)', fontsize=12)
ax.set_ylabel(r'$\lambda_i$', fontsize=12)
ax.set_title('Interactive Weight Adaptation Rule', fontsize=14)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper left')

# 4. Define the axes for the sliders
# [left, bottom, width, height]
axcolor = 'lightgoldenrodyellow'
ax_r = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_s = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_t = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_linf = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_lsup = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

# 5. Create the slider objects
slider_r = Slider(ax_r, 'r (Weight)', -5.0, 5.0, valinit=init_r, valstep=0.1)
slider_s = Slider(ax_s, 's (Exp 1)', 0.1, 5.0, valinit=init_s, valstep=0.1)
slider_t = Slider(ax_t, 't (Exp 2)', 0.1, 10.0, valinit=init_t, valstep=0.1)
slider_linf = Slider(ax_linf, r'$\lambda_{inf}$', 0.0, 10.0, valinit=init_l_inf, valstep=0.5)
slider_lsup = Slider(ax_lsup, r'$\lambda_{sup}$', 0.0, 10.0, valinit=init_l_sup, valstep=0.5)


# 6. Define the update function
def update(val):
    # Fetch current values from sliders
    r = slider_r.val
    s = slider_s.val
    t = slider_t.val
    l_inf = slider_linf.val
    l_sup = slider_lsup.val

    # Optional: Enforce the structural constraint t > s dynamically
    if t <= s:
        slider_t.set_val(s + 0.1)
        t = slider_t.val

    # Recalculate and update the plot data
    new_y = calculate_lambda(eta, r, s, t, l_inf, l_sup)
    line.set_ydata(new_y)

    # Dynamically adjust the Y-axis if the function goes out of bounds
    current_ymin, current_ymax = ax.get_ylim()
    min_y = min(-2.0, np.min(new_y) - 1)
    max_y = max(10.0, np.max(new_y) + 1)

    if min_y < current_ymin or max_y > current_ymax:
        ax.set_ylim(min_y, max_y)

    fig.canvas.draw_idle()


# 7. Connect sliders to the update function
slider_r.on_changed(update)
slider_s.on_changed(update)
slider_t.on_changed(update)
slider_linf.on_changed(update)
slider_lsup.on_changed(update)

# Show the plot
plt.show()