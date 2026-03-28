import numpy as np

class PFLSafetyUtils:
    """
    Libreria matematica e cinematica per la gestione della sicurezza HRC 
    tramite Control Barrier Functions (CBF) e Time Scaling.
    Tr : Tempo di campionamento
    a_s : accelerazione/decelerazione massima lineare
    v_pfl : velocità di sicurezza di impatto (ISO 15066)
    v_max : velocità massima lineare
    rho : smothness dell'approssimazione softmax (maggiore rho più accuratezza)
    limit_err: errore massimo sul tracking della traiettoria
    """
    def __init__(self, Tr=0.002, a_s=2.5, v_pfl=0.25, v_max=2.0, rho=20.0, traj_max_err=0.1):
        # Parametri Fisici e Normativi
        self.Tr = Tr #tempo di recupero (tempo di reazione del sistema di controllo)
        self.a_s = a_s
        self.v_pfl = v_pfl
        self.v_max = v_max
        
        # Parametro di asprezza per aggregazione SoftMax
        self.rho = rho
        
        # Parametri Time Scaling
        self.traj_max_err = traj_max_err  # Errore massimo di tracking "fisiologico" (es. 3 cm)
        self.n_power = 6.0          # Ordine della Super-Gaussiana (plateau piatto)
        self.slope_d = 100.0        # Pendenza della sigmoide per la distanza

    def compute_h_softmax_and_grad(self, d, v_rel):
        """
        Calcola il valore e il gradiente della CBF globale di prossimità (Frenata + PFL)
        utilizzando l'operatore di aggregazione liscia SoftMax.
        """
        # 1. Barriera di Frenata
        h_br = d - (-v_rel * self.Tr + (v_rel**2) / (2.0 * abs(self.a_s)))
        grad_br = np.array([1.0, self.Tr - v_rel / self.a_s, 0.0])
        
        # 2. Barriera PFL (impatto ammissibile)
        h_pfl = (self.v_pfl + v_rel) * self.Tr
        grad_pfl = np.array([0.0, self.Tr, 0.0])

        # Aggregazione SoftMax con log-sum-exp trick per evitare overflow
        max_inner = max(h_br, h_pfl)
        exp_br = np.exp(self.rho * (h_br - max_inner))
        exp_pfl = np.exp(self.rho * (h_pfl - max_inner))
        sum_inner = exp_br + exp_pfl
        
        h_softmax = max_inner + (1.0 / self.rho) * np.log(sum_inner)
        
        # Calcolo dei pesi per il gradiente composto
        omega_br = exp_br / sum_inner
        omega_pfl = exp_pfl / sum_inner
        
        grad_hsoftmax = omega_br * grad_br + omega_pfl * grad_pfl
        
        return h_softmax, grad_hsoftmax

    def compute_h_vmax_and_grad(self, vr_act):
        """
        Calcola il valore e il gradiente della CBF per la limitazione 
        della velocità proiettata massima (v_max).
        """
        h_vmax_pos = (self.v_max - vr_act) * (self.v_max + vr_act)
        grad_vmax = np.array([0.0, 0.0, -2.0 * vr_act])
        
        return h_vmax_pos, grad_vmax

    def range_state_derivative(self, v_lin, v_human):
        """
        Calcola le derivate dello stato rispetto al tempo (f) 
        e rispetto all'input di controllo (g).
        """
        zero3 = np.zeros(3)
        f = np.concatenate([v_lin, v_human, zero3, zero3])
        g = np.zeros((12, 3))
        g[6:9] = np.eye(3)
        return f, g

    def jacobian_psi(self, p_r, p_h, v_lin, v_human):
        """
        Costruisce la matrice Jacobiana di trasformazione dallo spazio cartesiano
        globale allo spazio degli stati ridotto espanso [d, v_rel, v_R||].
        """
        diff = p_r - p_h
        norm = np.linalg.norm(diff)
        if norm < 1e-9: 
            norm = 1e-9
            
        u_rh = (diff / norm).reshape(3, 1)
        P = np.eye(3) - u_rh @ u_rh.T
        
        w = v_lin - v_human
        wP_over_d = (w @ P) / norm
        vrP_over_d = (v_lin @ P) / norm
        
        # Riga 1: Gradiente della distanza
        row_d = np.hstack((u_rh.T, -u_rh.T, np.zeros((1, 3)), np.zeros((1, 3))))
        
        # Riga 2: Gradiente della velocità relativa
        row_vrel = np.hstack((wP_over_d.reshape(1, -1), -wP_over_d.reshape(1, -1), u_rh.T, -u_rh.T))
        
        # Riga 3: Gradiente della velocità proiettata
        row_vract = np.hstack((vrP_over_d.reshape(1, -1), -vrP_over_d.reshape(1, -1), u_rh.T, np.zeros((1, 3))))
        
        return np.vstack((row_d, row_vrel, row_vract))

    def compute_ds_scaling(self, distance, tracking_error, d_thresh):
        """
        Calcola il fattore di Time Scaling combinando la distanza (Sigmoide)
        e l'errore di inseguimento ingiustificato (Super-Gaussiana).
        """
        # Fattore Sicurezza basato su Distanza fisica (Sigmoide)
        term_safety = 1.0 / (1.0 + np.exp(-self.slope_d * (distance - d_thresh)))
        
        # Fattore Errore basato su Super-Gaussiana
        term_error = np.exp(- (abs(tracking_error) / self.traj_max_err)**self.n_power)

        # Ritorna il caso più restrittivo tra i due
        return min(term_safety, term_error)
    
    

    @staticmethod
    def damped_pinv_svd(J, lam=1e-4):
        """
        Utility statica per il calcolo della pseudoinversa smorzata (Damped Pseudo-Inverse)
        usata per il calcolo dell'accelerazione nominale senza singolarità.
        """
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        S_damped = S / (S ** 2 + lam ** 2)
        return (Vt.T * S_damped) @ U.T