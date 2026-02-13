import numpy as np
import matplotlib.pyplot as plt

class StocasticalCBFVisualizer:
    def __init__(self):
        self.n = 50
        self.cycles = 0
        self.h_window = np.zeros(50)
        self.h_mean_vec= []

        self.h_dev_vec= []
        self.t_vec = []
        self.lambda_det_vec = []
        self.lambda_stoc_vec = []
        self.h_vec = []


    def update_vectors(self, h, t, cycles):
        self.h_window = np.roll(self.h_window, -1)
        self.h_window[-1] = h
        self.h_vec.append(h)
        if cycles < self.n:
            self.h_mean_vec.append(np.mean(self.h_window[-cycles:]))
            self.h_dev_vec.append(np.std(self.h_window[-cycles:]))
        else:
            self.h_mean_vec.append(np.mean(self.h_window))
            self.h_dev_vec.append(np.std(self.h_window))
        self.t_vec.append(t)


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

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.t_vec, self.lambda_det_vec, label="LAMBDA DETERMINISTIC")
        ax1.plot(self.t_vec, self.lambda_stoc_vec, label="LAMBDA STOCHASTIC")
        ax1.axhline(lambda_f, color="black", linestyle="--", label = "lamda_f")
        ax1.axhline(lambda_0, color="black", linestyle="--", label = "lambda_0")

        ax2.plot(self.t_vec, self.h_vec, label="h", color = "red")
        ax2.axhline(0, color="black", linewidth = 2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
        plt.title("LAMBDA COMPARISON")
        ax1.grid(True)

        ax1.set_xlabel("time")
        plt.show()