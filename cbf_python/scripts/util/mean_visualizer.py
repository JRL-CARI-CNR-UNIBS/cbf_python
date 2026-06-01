import numpy as np
import matplotlib.pyplot as plt

class StochasticCBFVisualizer:
    def __init__(self, n=50):
        self.v_mean = None
        self.d_mean = None
        self.h_mean = None
        self.cov_matrix = None
        self.n = n
        self.cycles = 0
        self.h_window = np.zeros(n)
        self.h_vec= []
        self.d_vec= []
        self.v_vec=[]
        self.time_vec=[]

        self.lambda_vec = []


    def update_vectors(self, h, d, v_rel, t):
        self.h_vec.append(h)
        self.d_vec.append(d)
        self.v_vec.append(v_rel)
        self.time_vec.append(t)



    def compute_mean_cov(self, print_val:bool=False):

        # 2. Impacchettare i dati in una singola matrice
        # np.vstack impila le liste una sull'altra.
        # Otteniamo una matrice dove ogni RIGA è una variabile (h, d, v) e ogni COLONNA è un campione.
        data_matrix = np.vstack((self.h_vec, self.d_vec, self.v_vec))

        # 3. Calcolare l'intera Matrice di Covarianza (3x3)
        # NumPy calcola automaticamente sia le varianze (sulla diagonale) che le covarianze
        self.cov_matrix = np.cov(data_matrix)
        self.h_mean =  np.mean(self.h_vec, axis=0)
        self.d_mean = np.mean(self.d_vec, axis=0)
        self.v_mean = np.mean(self.v_vec, axis=0)
        # 4. (Opzionale) Calcolare le varianze singole per verifica
        # ddof=1 indica che stiamo lavorando su un *campione* statistico, non sull'intera popolazione
        var_h = np.std(self.h_vec, ddof=1)
        var_d = np.std(self.d_vec, ddof=1)
        var_v = np.std(self.v_vec, ddof=1)
        if print_val:
            print(f"--- Medie dati ---")
            print(f"h: {np.mean(self.h_vec):.4f}")
            print(f"d: {np.mean(self.d_vec):.4f}")
            print(f"v: {np.mean(self.v_vec):.4f}")
    
            print("\n--- Matrice di Covarianza (Σ) ---")
            print(np.round(self.cov_matrix, 4))
            print("\n--- Deviazioni Standard Singole ---")
            print(f"Deviazione standard di h: {var_h:.4f}")
            print(f"Deviazione standard di d: {var_d:.4f}")
            print(f"Deviazione standard di v: {var_v:.4f}")


    def plot_mean_std(self, lambda_0, lambda_f):
        # generates subplots of h_mean and h_std
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        print("DEVIATION OF DEVIATION: "+str(np.std(self.h_dev_vec)))
        print("MEAN OF DEVIATION: "+str(np.mean(self.h_dev_vec)))
        # X vs time
        axes[0].plot(self.t_vec, self.h_mean_vec, color="#CC5500", label="h mean over time")
        axes[0].set_ylabel("h_mean")
        axes[0].set_xlabel("time")
        axes[0].grid(True)
        axes[0].legend()

        # Y vs time
        axes[1].plot(self.t_vec, self.h_dev_vec, color="#AD00CC", label="std dev over time")
        axes[1].set_ylabel("standard deviation")
        axes[1].set_xlabel("time")
        axes[1].grid(True)
        axes[1].legend()

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(self.t_vec, self.lambda_det_vec, label="LAMBDA DETERMINISTIC")
        # ax1.plot(self.t_vec, self.lambda_stoc_vec, label="LAMBDA STOCHASTIC")
        # ax1.axhline(lambda_f, color="black", linestyle="--", label = "lamda_f")
        # ax1.axhline(lambda_0, color="black", linestyle="--", label = "lambda_0")
        #
        # ax2.plot(self.t_vec, self.h_vec, label="h", color = "red")
        # ax2.axhline(0, color="black", linewidth = 2)
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
        # plt.title("LAMBDA COMPARISON")
        # ax1.grid(True)
        #
        # ax1.set_xlabel("time")
        # plt.show()