import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carica i dati dal file CSV
df = pd.read_csv('../resullts/simulation_data_dynamic_params_comparison.csv')

# 2. Ordina i dati per l'asse X per evitare linee a zigzag
df = df.sort_values(by=['h_mean_test'])
x_values = [0.2681242732904403, -0.1600670502627323, 0.8291488806255026]
legend_vertical_values=["optimization value for paper params", "optimization value for h = -0.1", "optimization value for h = 1.0"]
# 3. Riorganizza il DataFrame per portare 'dynamic_params' in fondo.
# Questo garantisce che venga plottata per ultima e rimanga in primo piano.
df_others = df[df['test_type'] != 'dynamic_params']
df_dynamic = df[df['test_type'] == 'dynamic_params']
df = pd.concat([df_others, df_dynamic])

# 4. Definisci le metriche e la griglia
metrics = ['on_target_rate', 'lap_count', 'viol_rate', 'mean_scale', 'mean_trajectory_error']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
axes = axes.flatten()
colors_palette = {
    'dynamic_params': 'red',
    'Paper_params': 'green',
    'params_-0.1': 'blue',
    'params_1.0': 'orange'
}
# 5. Genera i grafici
for i, metric in enumerate(metrics):
    ax = axes[i]

    # Crea il lineplot (dynamic_params verrà disegnata per ultima e sarà in primo piano)
    sns.lineplot(
        data=df,
        x='h_mean_test',
        y=metric,
        hue='test_type',
        palette=colors_palette,
        marker='o',
        ax=ax
    )
    # --- Aggiunta di linee di riferimento ---
    # Linea orizzontale (es. valore di riferimento y=0.5).
    # zorder=0 la mette in secondo piano dietro i dati.
    if metric in ['on_target_rate',  'mean_scale']:
        ref_value = 1.0
    elif metric in ['viol_rate', 'mean_trajectory_error']:
        ref_value = 0.0
    else:
        ref_value = 35.0

    ax.axhline(y=ref_value, color='purple', linestyle='--', linewidth=1.5, zorder=0, label='Reference value')

    # Linea verticale (es. valore di riferimento x=0.2)
    for i in range(len(legend_vertical_values)):
        ax.axvline(x=x_values[i], color=list(colors_palette.values())[i+1], linestyle=':', linewidth=1.5, zorder=0, label=legend_vertical_values[i])
    # Imposta titolo e etichette
    ax.set_title(metric)
    ax.set_xlabel('h_mean_test')
    ax.set_ylabel(metric)
    ax.grid()

    # Mostra la legenda (includerà anche le linee di riferimento aggiunte)
    ax.legend()

# 6. Rimuovi l'ultimo subplot vuoto
fig.delaxes(axes[5])

# 7. Ottimizza il layout e mostra il grafico
plt.tight_layout()
plt.show()