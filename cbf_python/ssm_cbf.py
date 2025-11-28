import numpy as np

def compute_h(d, v, C, Tr, a_s, v_h):
    """Optimized version: fewer branches, no extra allocations."""
    if v < 0.0:
        if v_h > 0:
            dmin = C + v_h * (Tr - v / a_s) - v * Tr + 0.5 * v ** 2 / a_s
            return d - dmin
        if v_h < v:
            return (d - C) if d >= C else (d - C) + (C - d) * (Tr / C) * v
        return d - C + (v - v_h) * Tr - 0.5 * (v_h - v) ** 2 / a_s

    # v >= 0 branch
    if v_h < 0:
        dmin = C
        coef = Tr
    else:
        dmin = C + v_h * Tr
        coef = Tr + v_h / a_s

    return coef * v if d < dmin else (d - dmin) + coef * v


def jacobian_h(d, v, v_h, C, Tr, a_s):
    """Optimized version: uses precomputed scalars, avoids extra arrays."""
    if v < 0.0:
        if v_h > 0:
            return np.array([1.0, Tr, -Tr + v / a_s])
        if v_h < v:
            if d >= C:
                return np.array([1.0, 0.0, 0.0])
            return np.array([1.0 - (Tr / C) * v, (C - d) * (Tr / C), 0.0])
        return np.array([1.0, Tr + (v_h - v) / a_s, -Tr - (v_h - v) / a_s])

    # v >= 0
    if v_h < 0:
        dmin = C
        coef = Tr
        d_coef_vh = 0.0
    else:
        dmin = C + v_h * Tr
        coef = Tr + v_h / a_s
        d_coef_vh = 1.0 / a_s

    dh_dd = 0.0 if d < dmin else 1.0
    return np.array([dh_dd, coef, d_coef_vh * v])


def jacobian_h2(d, v, v_h, a_h, C, Tr, a_s):
    """Optimized but still follows envelope theorem logic."""
    eps = 1e-12

    if v >= 0.0:
        dh_dd = 0.0 if d < C else 1.0
        return dh_dd, Tr, 0.0

    if a_s <= 0:
        raise ValueError("a_s must be > 0 when v < 0.")

    t_stop = Tr - v / a_s
    # Candidate evaluation function (fast inline version)
    def d_total(t):
        dr = v * t if t <= Tr else (v * t + 0.5 * a_s * (t - Tr) ** 2)
        return d + dr - (v_h * t + 0.5 * a_h * t * t)

    candidates = np.array([
        (0.0, d_total(0.0)),
        (Tr, d_total(Tr)),
        (t_stop, d_total(t_stop))
    ])

    # Interior candidates
    if abs(a_h) > eps:
        t_star = (v - v_h) / a_h
        if 0.0 < t_star < Tr:
            candidates = np.vstack((candidates, (t_star, d_total(t_star))))

    if abs(a_h - a_s) > eps:
        t_prime = ((v - v_h) - a_s * Tr) / (a_h - a_s)
        if Tr < t_prime < t_stop:
            candidates = np.vstack((candidates, (t_prime, d_total(t_prime))))

    t_min, _ = candidates[np.argmin(candidates[:, 1])]
    dh_dd = 1.0

    if t_min <= 1e-9:
        return dh_dd, 0.0, 0.0
    if abs(t_min - Tr) <= 1e-9:
        return dh_dd, Tr, -Tr
    if abs(t_min - t_stop) <= 1e-9:
        vh_at_stop = v_h + a_h * t_stop
        return dh_dd, t_stop + vh_at_stop / a_s, -t_stop

    return dh_dd, t_min, -t_min


def range_state_derivative(v_lin, v_human):
    """No change needed, already efficient."""
    zero3 = np.zeros(3)
    f = np.concatenate([v_lin, v_human, zero3, zero3])
    g = np.zeros((12, 3))
    g[6:9] = np.eye(3)
    return f, g


def jacobian_psi(p_r, p_h, v_lin, v_human):
    """Optimized to avoid repeated allocations and @ operator overhead."""
    diff = p_r - p_h
    norm = np.linalg.norm(diff)
    u_rh = (diff / norm).reshape(3, 1)
    P = np.eye(3) - u_rh @ u_rh.T

    vlinP = v_lin @ P
    vhumP = v_human @ P

    return np.vstack((
        np.hstack((u_rh.T, -u_rh.T, np.zeros((1, 3)), np.zeros((1, 3)))),
        np.hstack((vlinP.reshape(1, -1), -vlinP.reshape(1, -1), u_rh.T, np.zeros((1, 3)))),
        np.hstack((vhumP.reshape(1, -1), -vhumP.reshape(1, -1), np.zeros((1, 3)), u_rh.T))
    ))

def lie_fg_h_fast(p_r, p_h, v_lin, v_human, C, Tr, a_s,
                  Lie_f_h_out=None, Lie_g_h_out=None, eps=1e-12):
    """
    Compute Lie_f_h (scalar) and Lie_g_h (1x3) directly,
    writing into provided arrays if given.
    """
    diff = p_r - p_h
    d = np.linalg.norm(diff)
    if d < eps:
        u = np.zeros(3)
        vlin_u = vhum_u = 0.0
        vlinP = v_lin
        vhumP = v_human
    else:
        u = diff / d
        vlin_u = np.dot(v_lin, u)
        vhum_u = np.dot(v_human, u)
        vlinP = v_lin - vlin_u * u
        vhumP = v_human - vhum_u * u

    v = vlin_u - vhum_u
    v_h = vhum_u

    hd, hv, hvh = jacobian_h(d=d, v=v, v_h=v_h, C=C, Tr=Tr, a_s=a_s)

    dh_dpr = hd * u + hv * vlinP + hvh * vhumP
    dh_dph = -dh_dpr
    dh_dvlin = hv * u

    if Lie_f_h_out is None:
        Lie_f_h = np.dot(dh_dpr, v_lin) + np.dot(dh_dph, v_human)
    else:
        Lie_f_h_out = np.dot(dh_dpr, v_lin) + np.dot(dh_dph, v_human)
        Lie_f_h = Lie_f_h_out

    if Lie_g_h_out is None:
        Lie_g_h = dh_dvlin
    else:
        Lie_g_h_out[:] = dh_dvlin
        Lie_g_h = Lie_g_h_out

    return Lie_f_h, Lie_g_h
