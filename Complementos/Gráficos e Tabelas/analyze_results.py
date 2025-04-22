import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Carregar os dados obtidos
df_omp = pd.read_csv('../Inverse_Matriz/02_Parallel_openmp/results_omp.csv')
df_serial = pd.read_csv('../Inverse_Matriz/01_Serial/results_row.csv')

# Agrupar os resultados do OpenMP e calcular a média para cada combinação de tamanho e threads
df_omp_avg = df_omp.groupby(['tamanho_matriz', 'num_threads'])['tempo_execucao'].mean().reset_index()
print("Tempo médio por tamanho de matriz e número de threads:")
print(df_omp_avg.pivot(index='tamanho_matriz', columns='num_threads', values='tempo_execucao'))

# Preparar os dados seriais
serial_times = df_serial.set_index('tamanho_matriz')['tempo_execucao']

# Calcular o speedup usando os tempos médios
df_omp_avg['speedup'] = df_omp_avg.apply(
    lambda row: serial_times.loc[row['tamanho_matriz']] / row['tempo_execucao'] 
    if row['tamanho_matriz'] in serial_times.index else np.nan, axis=1
)

# Configuração estética - Paleta de cores bem distintas
colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#FFFF00', '#000000', '#FF00FF']
sns.set_style("whitegrid")
plt.figure(figsize=(16, 12))

# Plot 1: Tempo de execução médio X Tamanho da Matriz para diferentes números de threads
plt.subplot(2, 2, 1)
threads = df_omp_avg['num_threads'].unique()
for i, thread in enumerate(threads):
    subset = df_omp_avg[df_omp_avg['num_threads'] == thread]
    plt.plot(subset['tamanho_matriz'], subset['tempo_execucao'], 
             marker='o', linewidth=2.5, 
             label=f"{thread}", 
             color=colors[i % len(colors)])

plt.xscale('log')
plt.yscale('log')
plt.title('Tempo de Execução Médio X Tamanho da Matriz', fontsize=14)
plt.xlabel('Tamanho da Matriz (N)', fontsize=12)
plt.ylabel('Tempo de Execução Médio (s)', fontsize=12)
plt.grid(True)
plt.legend(title='Threads', title_fontsize=12, fontsize=10)

# Plot 2: Speedup X Número de Threads para diferentes tamanhos de matriz
plt.subplot(2, 2, 2)
sizes = df_omp_avg['tamanho_matriz'].unique()
for i, size in enumerate(sizes):
    subset = df_omp_avg[df_omp_avg['tamanho_matriz'] == size]
    plt.plot(subset['num_threads'], subset['speedup'], 
             marker='o', linewidth=2.5, 
             label=f"{size}", 
             color=colors[i % len(colors)])

plt.title('Speedup X Número de Threads', fontsize=14)
plt.xlabel('Número de Threads', fontsize=12)
plt.ylabel('Speedup (T_serial / T_paralelo)', fontsize=12)
plt.grid(True)
plt.legend(title='Tamanho', title_fontsize=12, fontsize=10)
plt.xticks([1, 2, 4, 8, 16])

# Plot 3: Eficiência (Speedup/Threads) X Número de Threads
df_omp_avg['efficiency'] = df_omp_avg['speedup'] / df_omp_avg['num_threads']
plt.subplot(2, 2, 3)
for i, size in enumerate(sizes):
    subset = df_omp_avg[df_omp_avg['tamanho_matriz'] == size]
    plt.plot(subset['num_threads'], subset['efficiency'], 
             marker='o', linewidth=2.5, 
             label=f"{size}", 
             color=colors[i % len(colors)])

plt.title('Eficiência X Número de Threads', fontsize=14)
plt.xlabel('Número de Threads', fontsize=12)
plt.ylabel('Eficiência (Speedup/Threads)', fontsize=12)
plt.grid(True)
plt.legend(title='Tamanho', title_fontsize=12, fontsize=10)
plt.xticks([1, 2, 4, 8, 16])

# Plot 4: Comparação direta serial X paralelo
best_parallel_avg = df_omp_avg.loc[df_omp_avg.groupby('tamanho_matriz')['tempo_execucao'].idxmin()]
best_parallel_avg = best_parallel_avg.set_index('tamanho_matriz')

# Criar um DataFrame combinado para comparação
comparison = pd.DataFrame({
    'Serial': serial_times,
    'Paralelo': best_parallel_avg['tempo_execucao'],
    'Threads Usadas': best_parallel_avg['num_threads']
})

plt.subplot(2, 2, 4)
ax = comparison[['Serial', 'Paralelo']].plot(kind='bar', figsize=(10, 6), log=True, 
                                             color=['#FF0000', '#0000FF'], ax=plt.gca())
                                             
# Ajustar rótulos de barras para mostrar valores
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=8)
    
plt.title('Comparação: Serial X Paralelo', fontsize=14)
plt.xlabel('Tamanho da Matriz', fontsize=12)
plt.ylabel('Tempo de Execução (s)', fontsize=12)
plt.grid(True, axis='y')
plt.xticks(rotation=45)
plt.legend(fontsize=10)

# Salvar a imagem com alta resolução
plt.tight_layout()
plt.savefig('inverse_matrix_performance_avg.png', dpi=300)
plt.show()

# Imprimir estatísticas
print("\nEstatísticas de Speedup (baseadas na média de 3 execuções):")
print(df_omp_avg.groupby('tamanho_matriz')['speedup'].max().reset_index().rename(
    columns={'speedup': 'Max Speedup'}))

print("\nMelhor número de threads por tamanho de matriz (baseado na média):")
print(best_parallel_avg['num_threads'])

# Adicionar uma tabela com as melhorias percentuais
improvement = pd.DataFrame({
    'Tamanho': best_parallel_avg.index,
    'Tempo Serial (s)': serial_times[best_parallel_avg.index],
    'Tempo Paralelo (s)': best_parallel_avg['tempo_execucao'],
    'Melhoria (%)': ((serial_times[best_parallel_avg.index] - best_parallel_avg['tempo_execucao']) / 
                     serial_times[best_parallel_avg.index]) * 100,
    'Speedup': best_parallel_avg['speedup'],
    'Threads Ótimo': best_parallel_avg['num_threads']
})

print("\nTabela de melhorias por tamanho de matriz:")
print(improvement)