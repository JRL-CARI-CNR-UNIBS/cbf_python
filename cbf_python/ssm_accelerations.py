import math

def compute_h(d, v, v_h=0.0, a_h=0.0, C=0.25, Tr=0.15, a_s=2.5):
    """
    Compute h = min_t d(t) - C over t in [0, t_stop],
    where:
      d(t) = d_act + d_r(t) - d_h(t)
      d_r(t) = ∫_0^t v_r(τ) dτ with v_r(t) = v for t < Tr, and v_r(t) = v + a_s (t - Tr) for t ≥ Tr
      d_h(t) = ∫_0^t (v_h + a_h τ) dτ = v_h t + 0.5 a_h t^2
      t_stop = Tr - v/a_s  (requires a_s > 0)

    Parameters
    ----------
    d   : float  (d_act)
    v   : float  (v_r,act)
    C   : float
    Tr  : float  (t_r)
    a_s : float  (slope/accel for v_r on t ≥ Tr; must be > 0)
    v_h : float  (v_h,act), default 0.0
    a_h : float  (a_h), default 0.0

    Returns
    -------
    result : dict with
        - 'h': min d(t) - C
        - 't_min': argmin time in [0, t_stop]
        - 'which': one of {'t=0','t=Tr','t=t*','t=t\'','t=t_stop'}
        - 'A','B','Ccheck': booleans for vh<vr at t=0, t=Tr, t=t_stop
        - 'candidates': list of (t, d(t)) checked
    """

    if v < 0.0:
        if a_s <= 0:
            raise ValueError("a_s must be > 0.")
        # stop time (when v_r hits 0 on the linear branch)
        t_stop = Tr - v / a_s

        # piecewise v_r and the cumulative integrals
        def d_r(t):
            if t <= Tr:
                return v * t
            elif t <= t_stop:
                return v * t + 0.5 * a_s * (t - Tr) ** 2
            else:
                # beyond t_stop, v_r=0 so integral is constant
                return v * Tr - 0.5 * (v ** 2) / a_s

        def d_h_of(t):
            return v_h * t + 0.5 * a_h * (t ** 2)

        def d_total(t):
            return d + d_r(t) - d_h_of(t)

        # endpoint logic checks
        vr0 = v
        vr_Tr = v
        vr_stop = 0.0
        vh0 = v_h
        vh_Tr = v_h + a_h * Tr
        vh_stop = v_h + a_h * t_stop

        A = (vh0 < vr0)
        B = (vh_Tr < vr_Tr)
        Ccheck = (vh_stop < vr_stop)

        # candidate times: endpoints + interior stationary points (if valid)
        eps = 1e-12
        candidates = []

        # endpoints
        candidates.append((0.0, d_total(0.0)))
        candidates.append((Tr, d_total(Tr)))
        candidates.append((t_stop, d_total(t_stop)))

        # pre-Tr stationary point t* if a_h != 0 and inside (0,Tr)
        if abs(a_h) > eps:
            t_star = (v - v_h) / a_h
            if 0.0 < t_star < Tr:
                candidates.append((t_star, d_total(t_star)))

        # post-Tr stationary point t' if a_h != a_s and inside (Tr, t_stop)
        if abs(a_h - a_s) > eps:
            t_prime = ((v - v_h) - a_s * Tr) / (a_h - a_s)
            if Tr < t_prime < t_stop:
                candidates.append((t_prime, d_total(t_prime)))

        # choose minimum over [0, t_stop]
        # (filter any numerical strays outside the interval)
        valid = [(t, val) for (t, val) in candidates if 0.0 - eps <= t <= t_stop + eps]
        if not valid:
            # Shouldn't happen, but fallback to endpoints
            valid = [(0.0, d_total(0.0)), (t_stop, d_total(t_stop))]

        t_min, d_min = min(valid, key=lambda x: x[1])

        # label which candidate
        def which_label(t):
            if abs(t - 0.0) <= 1e-9: return 't=0'
            if abs(t - Tr)  <= 1e-9: return 't=Tr'
            if abs(t - t_stop) <= 1e-9: return 't=t_stop'
            # try to distinguish t* vs t'
            if t < Tr: return 't=t*'
            return "t=t'"

        h = d_min -C
    else:
        if d<C:
            h = Tr*v
        else:
            h = (d-C)+Tr*v

    return h




h=compute_h(d=0.3,v=0.1,v_h=0.2,a_h=0.4)

print(f"h={h}")