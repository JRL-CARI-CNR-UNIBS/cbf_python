import numpy as np
import quadprog

class QPSolver:
    def __init__(self, nq, Tc, DDq_MAX, Dq_MAX, eps_track):
        self.nq = nq
        self.nv = nq + 2  # Variabili: [ddq(0:5), delta(6), s_ddot(7)]
        self.Tc = Tc
        self.DDq_MAX = DDq_MAX
        self.Dq_MAX = Dq_MAX
        self.eps_track = eps_track
        
        # --- TUNING PESI E LIMITI DINAMICI ---
        self.w_delta = 1e5
        self.w_dds = 5.0
        
        self.s_ddot_decel_max = -15.0
        self.s_ddot_accel_max = 0.5
        self.delta_growth_max = 5.0
        self.delta_decay_max = 0.1
        
        # === TOGGLES PER I VINCOLI ===
        self.enable_velocity_limits = True
        self.enable_delta_dynamics = True
        self.enable_scaling_dynamics = True
        
        # Inizializzazione
        self.C_list = []
        self.v_list = []
        self.P = None
        self.b = None

    def reset_constraints(self):
        """Pulisce le matrici dei vincoli del ciclo precedente."""
        self.C_list = []
        self.v_list = []
        self.P = None
        self.b = None

    def set_cost_function(self, J, dJ, dq, dtwist_base, nominal_twist_goal, s_ddot_des):
        """Costruisce la matrice P e il vettore b della funzione di costo gerarchica."""
        err_acc = dtwist_base - dJ @ dq
        v_nom_col = nominal_twist_goal.reshape(6, 1) 
        
        # Blocchi Diagonali
        P_qq = J.T @ J + 1e-6 * np.eye(self.nq)
        P_delta = np.array([[self.w_delta]])
        P_ss = v_nom_col.T @ v_nom_col + np.array([[self.w_dds]])
        
        # Blocchi intralelazionali
        P_qs = -J.T @ v_nom_col  
        P_sq = P_qs.T            
        
        # Assemblaggio Vettori Lineari (b)
        b_q = (J.T @ err_acc).flatten()
        b_delta = np.array([0.0])
        b_s = (-v_nom_col.T @ err_acc).flatten() + np.array([self.w_dds * s_ddot_des])

        # Costruzione P finale a blocchi
        zeros_qd = np.zeros((self.nq, 1)) 
        zeros_dq = np.zeros((1, self.nq)) 
        zeros_sd = np.zeros((1, 1))        
        zeros_ds = np.zeros((1, 1))        

        self.P = np.block([
            [P_qq,     zeros_qd, P_qs    ],
            [zeros_dq, P_delta,  zeros_ds],
            [P_sq,     zeros_sd, P_ss    ]
        ])
        self.b = np.concatenate([b_q, b_delta, b_s])

    def add_tracking_and_state_constraints(self, Jlin, dJlin, dq, translation_bt, next_x_des_base, v_nom_lin, delta_prev, s_dot):
        """Aggiunge i vincoli di accelerazione, velocità (ai giunti), vincoli di tracciamento della traiettoria, sullo scaling e sulla variabile slack"""
        
        # Acceleration constraint
        A_acc = np.hstack([np.eye(self.nq), np.zeros((self.nq, 2))])
        b_acc = np.ones((self.nq, 1)) * self.DDq_MAX
        self.C_list.extend([A_acc, -A_acc])
        self.v_list.extend([-b_acc, -b_acc])

        # Tracking constraint (Tubo)
        A_track_up = np.hstack([-Jlin * 0.5 * self.Tc**2, np.ones((3, 1)), v_nom_lin * 0.5 * self.Tc**2])
        A_track_lower = np.hstack([Jlin * 0.5 * self.Tc**2, np.ones((3, 1)), -v_nom_lin * 0.5 * self.Tc**2])
        
        v_curr = Jlin @ dq
        x_free = translation_bt + v_curr * self.Tc + 0.5 * (dJlin @ dq) * self.Tc**2
        b_track_up = (-self.eps_track + (x_free - next_x_des_base)).reshape(-1, 1)
        b_track_lower = (-self.eps_track - (x_free - next_x_des_base)).reshape(-1, 1)
        
        self.C_list.extend([A_track_up, A_track_lower])
        self.v_list.extend([b_track_up, b_track_lower])

        # Velocity constraint 
        if self.enable_velocity_limits:
            A_vel = np.hstack([np.eye(self.nq), np.zeros((self.nq, 2))])
            b_vel_up = ((self.Dq_MAX - dq)/self.Tc).reshape(-1, 1)
            b_vel_low = ((-self.Dq_MAX - dq)/self.Tc).reshape(-1, 1)
            self.C_list.extend([A_vel, -A_vel])
            self.v_list.extend([b_vel_low, -b_vel_up])

        # Slack variable constraint
        A_delta_pos = np.zeros((1, self.nv))
        A_delta_pos[0, self.nq] = 1.0
        self.C_list.append(A_delta_pos)
        self.v_list.append(np.zeros((1, 1)))


        if self.enable_delta_dynamics:
            A_delta_dyn = np.zeros((2, self.nv))
            A_delta_dyn[0, self.nq] = 1.0   
            A_delta_dyn[1, self.nq] = -1.0  
            b_delta_dyn = np.array([
                [delta_prev - self.delta_decay_max * self.Tc],
                [-(delta_prev + self.delta_growth_max * self.Tc)]
            ])
            self.C_list.append(A_delta_dyn)
            self.v_list.append(b_delta_dyn)

        # Time scaling constraint
        if self.enable_scaling_dynamics:
            A_s_dyn = np.zeros((4, self.nv))
            A_s_dyn[0, self.nq + 1] = 1.0   
            A_s_dyn[1, self.nq + 1] = -1.0  
            A_s_dyn[2, self.nq + 1] = 1.0   
            A_s_dyn[3, self.nq + 1] = -1.0  
            b_s_dyn = np.array([
                [self.s_ddot_decel_max],  
                [-self.s_ddot_accel_max],  
                [-s_dot / self.Tc],
                [-(1.0 - s_dot) / self.Tc]
            ])
            self.C_list.append(A_s_dyn)
            self.v_list.append(b_s_dyn)

    def add_custom_constraint(self, A_mat, b_vec):
        """Passare la matrice e il vettore del vincolo derivato dalla CBF"""
        self.C_list.append(A_mat)
        self.v_list.append(b_vec)

    def solve(self, fallback_dq):
        """Assemblo matrici e vettori dei vincoli"""
        C_total = np.concatenate(self.C_list, axis=0)
        v_total = np.concatenate(self.v_list, axis=0)

        try:
            sol = quadprog.solve_qp(self.P, self.b, C_total.T, v_total.flatten(), 0)
            qp_sol = sol[0]
            ddq = qp_sol[:self.nq]
            delta = qp_sol[self.nq]
            dds = qp_sol[self.nq + 1]
            return ddq, delta, dds, True
        except ValueError:
            # QP Infeasible Fallback
            return -10.0 * fallback_dq, 0.0, 0.0, False