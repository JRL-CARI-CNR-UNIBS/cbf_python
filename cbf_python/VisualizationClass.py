import matplotlib.pyplot as plt
import numpy as np

class LogPlotter:
    def __init__(self, logs, config):
        """
        Inizializza il plotter con i log della simulazione e i parametri di sicurezza.
        """
        # Estrazione e conversione dei log
        self.time = np.array(logs['time'])
        self.pos_act = np.array(logs['pos_act'])
        self.pos_nom = np.array(logs['pos_nom'])
        self.dist = np.array(logs['dist'])
        self.vrel = np.array(logs['vrel'])
        self.h = np.array(logs['h'])
        self.ddq = np.array(logs['ddq'])
        self.ddq_nom = np.array(logs['ddq_nom'])
        self.ds_time = np.array(logs['ds_time'])
        self.delta = np.array(logs['delta'])

        # Parametri di sicurezza
        self.v_pfl = config['v_pfl']
        self.a_s = config['a_s']
        self.Tr = config['Tr']
        self.v_max = config['v_max']

        # Impostazioni di stile globali per i grafici
        self._setup_style()

    def _setup_style(self):
        """Imposta lo stile standard per la tesi."""
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'legend.fontsize': 10,
            'figure.facecolor': 'white'
        })

    def plot_position_tracking(self):
        """Plot dell'inseguimento della traiettoria cartesiana (X, Y, Z)."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['x [m]', 'y [m]', 'z [m]']
        for i in range(3):
            axs[i].plot(self.time, self.pos_act[:, i], 'r-', label='Reale (Attuale)')
            axs[i].plot(self.time, self.pos_nom[:, i], 'k--', label='Nominale (Riferimento)')
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True, linestyle=':', alpha=0.7)
        
        axs[0].legend(loc='upper right')
        axs[0].set_title('Inseguimento della Traiettoria in Spazio Cartesiano')
        axs[2].set_xlabel('Tempo [s]')
        plt.tight_layout()

    def plot_phase_diagram(self):
        """Plot del diagramma di fase con la Safe Zone a tratti."""
        plt.figure(figsize=(12, 8))

        # Calcolo di d_crit
        d_crit = (self.v_pfl**2) / (2 * self.a_s) - (-self.v_pfl) * self.Tr
        
        # Confine di frenata
        v_br_plot = np.linspace(-self.v_pfl, -self.v_max - 0.5, 100)
        d_br_plot = -v_br_plot * self.Tr + (v_br_plot**2) / (2 * self.a_s)
        
        plt.plot(d_br_plot, v_br_plot, 'k-', linewidth=2.5, label='Confine Frenata (h_br = 0)')
        plt.plot([0, d_crit], [-self.v_pfl, -self.v_pfl], 'k--', linewidth=2.5, label='Confine PFL (h_PFL = 0)')
        
        max_d_plot = max(np.max(self.dist) + 0.5, np.max(d_br_plot) + 0.5)
        plt.axhline(y=self.v_max, color='k', linestyle=':', linewidth=2.5, label='Confine Allontanamento (h_vmax = 0)')

        # Colorazione Zone
        plt.fill_between([0, d_crit], -self.v_pfl, -self.v_max - 1, color='red', alpha=0.15)
        plt.fill_between(d_br_plot, v_br_plot, -self.v_max - 1, color='red', alpha=0.15, label='Zona di Violazione Normativa')
        plt.fill_between([0, max_d_plot], self.v_max, self.v_max + 1, color='red', alpha=0.15)
        
        plt.fill_between(d_br_plot, v_br_plot, self.v_max, color='green', alpha=0.05, label='Safe Zone')
        plt.fill_between([0, d_crit], -self.v_pfl, self.v_max, color='green', alpha=0.05)

        # Plot della traiettoria con gradiente temporale
        scatter = plt.scatter(self.dist, self.vrel, c=self.time, cmap='viridis', 
                              s=15, alpha=0.8, edgecolor='none', zorder=5)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Tempo [s]')
        
        # Punti chiave
        plt.plot(self.dist[0], self.vrel[0], 'bo', markersize=8, label='Inizio Traiettoria', zorder=6)
        min_dist_idx = np.argmin(self.dist)
        plt.plot(self.dist[min_dist_idx], self.vrel[min_dist_idx], 'ro', markersize=8, 
                 label=f'Minima Distanza\n({self.dist[min_dist_idx]:.2f}m, {self.vrel[min_dist_idx]:.2f}m/s)', zorder=6)

        plt.xlim(0, max_d_plot)
        plt.ylim(min(np.min(self.vrel) - 0.2, -self.v_max - 0.5), self.v_max + 0.5)
        plt.xlabel('Distanza Robot-Umano $d$ [m]')
        plt.ylabel('Velocità Relativa $v_{rel}$ [m/s]')
        plt.title('Spazio degli Stati: Verifica dell\'Invarianza della Safe Zone', weight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

    def plot_cbf_trend(self):
        """Plot dell'evoluzione della CBF nel tempo."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.h, 'b', label='$h(X)$ Effettivo')
        plt.axhline(0, color='k', linestyle='--', label='Limite Barriera (h = 0)')
        plt.ylabel('Valore Control Barrier Function $h(X)$')
        plt.xlabel('Tempo [s]')
        plt.title('Evoluzione Temporale della Control Barrier Function')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

    def plot_joint_accelerations(self):
        """Plot delle accelerazioni (Reali vs Nominali) dell'ultimo giunto."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.ddq[:, -1], 'r', label='Reale ($\ddot{q}_{act}$)')
        plt.plot(self.time, self.ddq_nom[:, -1], 'k--', label='Nominale ($\ddot{q}_{des}$)')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Accelerazione Giunto [rad/s^2]')
        plt.title('Accelerazione dell\'Ultimo Giunto del Manipolatore')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
    
    def plot_jerk_analysis(self):
        """
        Calcola e visualizza il Jerk (derivata dell'accelerazione) per i 6 giunti.
        Fondamentale per diagnosticare i picchi causati dal disallineamento 
        tra la frequenza dei sensori e la frequenza di controllo.
        """
        import numpy as np
        import matplotlib.pyplot as plt


        
        # Recupera il tempo di campionamento Tc (di default 2ms)
        Tc = self.Tr

        # Calcolo del Jerk usando il gradiente numerico
        # axis=0 calcola la derivata lungo le righe (il tempo) per ogni colonna (giunto)
        jerk_array = np.gradient(self.ddq, Tc, axis=0)

        # Creazione della griglia di subplot 3x2
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('Analisi del Jerk ai Giunti (Derivata terza $\\dddot{q}$)', fontsize=16, fontweight='bold')

        joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']

        for i in range(6):
            row = i // 2
            col = i % 2
            ax = axs[row, col]

            # Plottiamo il jerk in rosso per indicare la natura "critica" del dato
            ax.plot(self.time, jerk_array[:, i], color='firebrick', linewidth=1.0)

            # Linea guida dello zero per facilitare la lettura
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

            ax.set_title(f'Giunto {i+1}: {joint_names[i]}', fontsize=12)
            ax.set_ylabel('Jerk [rad/s³]')
            ax.set_xlabel('Tempo [s]')
            ax.grid(True, linestyle=':', alpha=0.7)

        # Aggiusta i margini per evitare che i grafici si sovrappongano
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()
        
    

    def plot_time_scaling(self):
        """Plot del fattore di Dynamic Time Scaling."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.ds_time, 'g', label='Fattore di Scaling ($\dot{s}$)')
        plt.axhline(1, color='k', linestyle=':', alpha=0.5)
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Velocità Tempo Fittizio')
        plt.title('Dynamic Time Scaling')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

    def plot_slack_variable(self):
        """Plot dell'andamento della variabile Slack (Delta)."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.delta, 'm', label='Variabile di Slack ($\delta$)')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Tempo [s]')
        plt.ylabel('Valore Rilassamento Tracking')
        plt.title('Attivazione della Variabile Slack')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

    def show_all_plots(self):
        """Genera e mostra tutti i grafici della tesi."""
        self.plot_position_tracking()
        self.plot_phase_diagram()
        self.plot_cbf_trend()
        self.plot_joint_accelerations()
        self.plot_time_scaling()
        self.plot_slack_variable()
        self.plot_jerk_analysis()
        plt.show()