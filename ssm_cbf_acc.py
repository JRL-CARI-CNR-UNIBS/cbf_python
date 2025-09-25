import numpy as np

def dmin_and_jacobian(
    d: float,
    v_r: float,
    a_r: float,
    v_h: float,
    a_h: float,
    tr: float,
    a_max: float,
    atol: float = 1e-12,
):
    """
    Compute minimum distance and Jacobian at that instant.

    Parameters
    ----------
    d : float
        Initial distance d(0).
    v_r, a_r : float
        Pursuer's initial velocity and acceleration (phase 1).
    v_h, a_h : float
        Human's initial velocity and acceleration.
    tr : float
        Reaction time (this is t2).
    a_max : float
        Pursuer's deceleration magnitude (>0).
    atol : float
        Tolerance for interval checks.

    Returns
    -------
    d_min : float
        Minimum distance across candidate times.
    jac : ndarray shape (5,)
        Gradient of d at the minimizing instant, with columns ordered as:
        (d, v_r, a_r, v_h, a_h).
    """

    if a_max <= 0:
        raise ValueError("a_max must be positive.")

    # Fix t0=0 and rename
    t0 = 0.0
    t2 = float(tr)

    # End of phase-2 braking
    t_dec_raw = (v_r + a_r * t2) / a_max
    t_dec = max(0.0, t_dec_raw)
    t4 = t2 + t_dec

    s1 = a_r - a_h
    m  = a_h + a_max

    # Relative speed
    def dotd(t):
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        mask1 = (t <= t2 + atol)
        out[mask1] = (v_r - v_h) + (a_r - a_h) * t[mask1]
        mask2 = ~mask1
        out[mask2] = (v_r - v_h + (a_r + a_max) * t2) - (a_h + a_max) * t[mask2]
        return out

    # Distance
    d_t2 = d + (v_r - v_h) * t2 + 0.5 * (a_r - a_h) * (t2**2)

    def dist(t):
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        mask1 = (t <= t2 + atol)
        out[mask1] = d + (v_r - v_h) * t[mask1] + 0.5 * (a_r - a_h) * (t[mask1]**2)
        mask2 = ~mask1
        out[mask2] = d_t2 + (v_r - v_h + (a_r + a_max) * t2) * (t[mask2] - t2) \
                     - 0.5 * (a_h + a_max) * (t[mask2]**2 - t2**2)
        return out

    # Intersections
    t1 = None
    if abs(a_h - a_r) > atol:
        t1_raw = (v_r - v_h) / (a_h - a_r)
        if (t0 + atol) < t1_raw < (t2 - atol):
            t1 = t1_raw

    t3 = None
    if abs(m) > atol and (t4 - t2) > atol:
        t3_raw = (v_r - v_h + (a_r + a_max) * t2) / m
        if (t2 + atol) < t3_raw < (t4 - atol):
            t3 = t3_raw

    # Conditions
    C0 = bool(dotd(t0) > 0)
    C2 = bool(dotd(t2) > 0)
    C4 = bool(dotd(t4) > 0)
    C1 = (abs(s1) > atol) and (C0 != C2) and (t1 is not None)
    C3 = (abs(m)  > atol) and (C2 != C4) and (t3 is not None)

    # Candidate set
    v_r_t2 = v_r + a_r * t2
    candidates = []
    if v_r_t2 < 0:
        candidates.extend([t0, t4])
        if C1: candidates.append(t1)
        if C3: candidates.append(t3)
    else:
        candidates.extend([t0, t2])
        if C1: candidates.append(t1)

    uniq = []
    for tt in candidates:
        if not any(abs(tt - uu) <= atol for uu in uniq):
            uniq.append(tt)

    vals = dist(np.array(uniq))
    i_min = int(np.argmin(vals))
    t_star = float(uniq[i_min])
    d_min = float(vals[i_min])

    # Jacobian at t_star
    def jac_at_time(tk):
        if abs(tk - t0) <= atol:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        if (t1 is not None) and (abs(tk - t1) <= atol):
            return np.array([1.0,
                             t1,
                             0.5 * (t1**2),
                             -t1,
                             -0.5 * (t1**2)])

        if abs(tk - t2) <= atol:
            return np.array([1.0,
                             t2,
                             0.5 * (t2**2),
                             -t2,
                             -0.5 * (t2**2)])

        if (t3 is not None) and (abs(tk - t3) <= atol):
            return np.array([1.0,
                             t3,
                             0.5 * (t2**2) + t2 * (t3 - t2),
                             -t3,
                             -0.5 * (t3**2)])

        if abs(tk - t4) <= atol:
            base = np.array([1.0,
                             t4,
                             0.5 * (t2**2) + t2 * (t4 - t2),
                             -t4,
                             -0.5 * (t4**2)])
            if t_dec > atol:
                dotd_t4 = float(dotd(t4))  # = -(v_h + a_h*t4)
                corr = np.array([0.0, dotd_t4 / a_max, (t2 * dotd_t4) / a_max, 0.0, 0.0])
                return base + corr
            else:
                return base

        raise RuntimeError("Unexpected candidate time for Jacobian.")

    jac = jac_at_time(t_star)
    return d_min, jac


def h_and_jacobian(
    d: float,
    v_r: float,
    a_r: float,
    v_h: float,
    a_h: float,
    tr: float,
    a_max: float,
    C: float,
    atol: float = 1e-12,
):
    d_min, dist_jac = dmin_and_jacobian(d=d, v_r=v_r, a_r=a_r, v_h=v_h, a_h=a_h, tr=tr, a_max=a_max, atol=atol)
    h=d_min-C+max(0.0,(C-d_min)*tr/C*v_r)
    h_jac = dist_jac
    if (d_min<C):
        h_jac += -dist_jac*tr/C*v_r + (C-d_min)/C*tr*np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    return h,h_jac

