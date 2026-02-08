# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import stats

df = pd.read_csv("simulation_history.csv")
df.head()
# %%

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Count_Def_D'], label='Count Def D')
plt.plot(df['Time'], df['Count_Def_U'], label='Count Def U')
plt.plot
plt.xlabel('Time')
plt.ylabel('Count Def')
plt.title('Count Def over Time')
plt.legend()
plt.grid()
plt.show()
# %%

plt.hist(df['Count_Def_Z'], bins=20, alpha=0.7, label='Count Def Z')
plt.hist(df['Count_Def_U'], bins=20, alpha=0.7, label='Count Def U')
plt.hist(df['Count_Def_D'], bins=20, alpha=0.7, label='Count Def D')

plt.xlabel('Count Def')
plt.ylabel('Frequency')
plt.title('Distribution of Count Def')
plt.legend()
plt.grid()
plt.show()

# %%

def avalanches_falencia(df):

    bad_debt = df["Bad_Debt"].values
    bd_median = np.median(bad_debt)
    bd_std = np.std(bad_debt)
    bd_prime = np.abs(bad_debt - bd_median)
    threshold = 2 * bd_std
    is_avalanche = bd_prime > threshold
    
    avalanche_times = df[is_avalanche]["Time"].values
    prob_evento = len(avalanche_times) / len(df)
    
    random_mask = np.random.rand(len(df)) < prob_evento
    random_times = df[random_mask]["Time"].values

    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(
        x=np.zeros(len(avalanche_times)), 
        y=avalanche_times,
        mode='markers',
        name=f'Credit Network (BD > 2σ)',
        marker=dict(symbol='circle', size=6, color='blue')
    ))

    fig5.add_trace(go.Scatter(
        x=np.ones(len(random_times)), 
        y=random_times,
        mode='markers',
        name='Random Process',
        marker=dict(symbol='circle', size=6, color='red')
    ))

    fig5.update_layout(
        title="Figura 5: Agrupamento de Grandes Calotes (Large Bad Debt)",
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=[f"nz={len(avalanche_times)}", f"nz={len(random_times)}"],
            range=[-0.5, 1.5],
            showgrid=False
        ),
        yaxis=dict(
            title="Tempo (t)",
            autorange="reversed" # Tempo 0 no topo
        ),
        template="plotly_white",
        width=600, height=700
    )
    fig5.show()


avalanches_falencia(df)
# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import model as mc

# Função auxiliar para o método de Rank
def get_rank_tent_shape(data):
    """
    Calcula os ranks para o gráfico 'Tent Shape' (Laplace).
    - Para valores < mediana: Rank Crescente (Subida)
    - Para valores > mediana: Rank Decrescente (Descida)
    """
    # 1. Ordenar os dados
    data_sorted = np.sort(data)
    n = len(data_sorted)
    
    # 2. Encontrar o ponto de virada (mediana ou zero)
    # O artigo usa taxa de crescimento, que oscila em torno de 0.
    # Vamos usar 0 como divisor se os dados forem centralizados, ou a mediana.
    median = np.median(data_sorted)
    
    # 3. Separar as caudas
    # Lado Esquerdo (Subida)
    left_mask = data_sorted <= median
    left_data = data_sorted[left_mask]
    # Rank: 1, 2, 3... (Probabilidade Acumulada Crescente)
    left_rank = np.arange(1, len(left_data) + 1)
    
    # Lado Direito (Descida)
    right_mask = data_sorted > median
    right_data = data_sorted[right_mask]
    # Rank: N, N-1, ... 1 (Probabilidade Acumulada Decrescente / Sobrevivência)
    right_rank = np.arange(len(right_data), 0, -1)
    
    return left_data, left_rank, right_data, right_rank

def plot_figura_6_rank(df):
    print("Gerando Figura 6 (Método Rank/Tent Shape)...")
    
    # 1. Calcular Taxa de Crescimento Agregado
    Y = df["Production"].values
    log_Y = np.log(np.maximum(Y, 1e-6))
    g = np.diff(log_Y)
    
    # Centralizar na média (opcional, mas ajuda a alinhar o pico no zero)
    g = g - np.mean(g)
    
    # 2. Calcular Ranks (Left e Right Tails)
    l_data, l_rank, r_data, r_rank = get_rank_tent_shape(g)
    
    fig6 = go.Figure()

    # Lado Esquerdo (Negativo)
    fig6.add_trace(go.Scatter(
        x=l_data, y=l_rank,
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Crescimento Negativo'
    ))
    
    # Lado Direito (Positivo)
    fig6.add_trace(go.Scatter(
        x=r_data, y=r_rank,
        mode='markers',
        marker=dict(size=4, color='blue'), # Mesma cor para parecer uma curva só
        name='Crescimento Positivo',
        showlegend=False
    ))

    fig6.update_layout(
        title="Figura 6: Distribuição de Crescimento (Rank Plot / Tent Shape)",
        xaxis_title="Taxa de Crescimento Agregado (g)",
        yaxis_title="log(Rank)", # Agora o rótulo bate com o artigo
        yaxis=dict(
            type='log', # Escala Log no Rank cria a Tenda
            exponentformat='power'
        ),
        template="plotly_white",
        width=700, height=600
    )
    fig6.show()

get_rank_tent_shape(df["Production"].values)
plot_figura_6_rank(df)
# %%
