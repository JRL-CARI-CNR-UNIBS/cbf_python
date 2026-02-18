from Controller.Numba_scripts.ssm_cbf_acc import *
import numpy as np
Tc: float = 2e-3
C: float = 0.25
Tr: float = 0.5
a_s: float = 4.5

def generate_d_value(h_ref, v_ref):
    d = 0.0
    h = -10
    while h < h_ref:
        h,_ =  h_and_jacobian_numba(d, -0.1, v_ref, 0.05, Tr, a_s, C, 1e-9)
        d+=0.01
    return d


def generate_pos(d, ee_x, ee_y, ee_z):
    """
    Genera un punto casuale (x, y, z) distribuito uniformemente sulla superficie
    di una sfera di raggio 'd' centrata in (ee_x, ee_y, ee_z),
    con il vincolo che z < ee_z (semisfera inferiore).

    Ritorna:
        np.array di shape (1, 3)
    """
    # 1. Genera l'angolo azimutale (da 0 a 360 gradi in radianti)
    theta = np.random.uniform(0, 2 * np.pi)

    # 2. Genera il coseno dell'angolo polare (v = cos(phi))
    # Intervallo [-1, 0) assicura di stare nella semisfera INFERIORE (z < ee_z)
    v = np.random.uniform(-1.0, 0.0)

    # 3. Calcola il seno dell'angolo polare derivandolo da v
    # sin^2(phi) + cos^2(phi) = 1  =>  sin(phi) = sqrt(1 - v^2)
    sin_phi = np.sqrt(1.0 - v ** 2)

    # 4. Calcola le coordinate cartesiane relative al centro
    x = ee_x + d * sin_phi * np.cos(theta)
    y = ee_y + d * sin_phi * np.sin(theta)
    z = ee_z + d * v  # Poiché v è negativo, z sarà sempre < ee_z

    # 5. Ritorna il punto come np.array di shape (1, 3)
    return np.array([[x, y, z]])