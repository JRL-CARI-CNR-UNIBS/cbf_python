import numpy as np
import matplotlib.pyplot as plt
from example_bcf import compute_h
from example_bcf import dh_dx
# Constants
C = 0.25  # [m]
Tr = 0.015  # [s]
a_s = 2.5  # [m/s²]




for v_h in np.linspace(-1.6,1.6,10):  # [m/s]

    # Create a grid of d and v values
    d_vals = np.linspace(0, 1, 20)
    v_vals = np.linspace(-3, 3, 20)
    D, V = np.meshgrid(d_vals, v_vals)

    # Compute derivatives
    U = np.zeros_like(D)
    W = np.zeros_like(V)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            dh_dd, dh_dv = dh_dx(D[i, j], V[i, j])
            U[i, j] = dh_dd
            W[i, j] = dh_dv


    # Compute h values for the grid
    H = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            H[i, j] = compute_h(D[i, j], V[i, j])

    # Create custom colormap: bluish for h > 0, reddish for h < 0
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.colors import CenteredNorm

    plt.figure(figsize=(10, 8))

    # Plot heatmap
    minH=np.min(H)
    maxH=np.max(H)
    print(f"min(h)={minH}, max(h)={maxH}")
    if minH<0:
        norm = TwoSlopeNorm(vmin=minH, vcenter=0, vmax=maxH)
    else:
        norm = CenteredNorm()
    heatmap = plt.pcolormesh(D, V, H, shading='auto', cmap='bwr', norm=norm)

    # Add colorbar
    plt.colorbar(heatmap, label='h value')

    # Overlay quiver plot
    plt.quiver(D, V, U, W, color='black', angles='xy', scale=10)

    plt.xlabel('d (distance) [m]')
    plt.ylabel('v (velocity) [m/s]')
    plt.title(f'Heatmap of h with Quiver Plot of ∂h/∂d and ∂h/∂v with v_h={v_h}')
    plt.grid(True)
    plt.show()
