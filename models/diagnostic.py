# %%
import sys
import subprocess

# Isso força a instalação do seaborn no exato Python que está rodando esta janela
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
print("Seaborn instalado com sucesso no ambiente atual!")
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("simulation_history.csv")
df.head()
# %%

plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Count_Def_D'], label='Count Def D')
plt.plot(df['Time'], df['Count_Def_U'], label='Count Def U')
plt.plot(df['Time'], df['Count_Def_Z'], label='Count Def Z')
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
        marker=dict(size=4, color='red'),
        name='Crescimento Negativo'
    ))
    
    # Lado Direito (Positivo)
    fig6.add_trace(go.Scatter(
        x=r_data, y=r_rank,
        mode='markers',
        marker=dict(size=4, color='blue'), # Mesma cor para parecer uma curva só
        name='Crescimento Positivo'
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


df.columns
# %%

def plot_diagnosticos_avancados(df):
    
    # ==============================================================================
    # 1. CICLO DA FRAGILIDADE: Alavancagem (D/U) vs Bad Debt
    # ==============================================================================
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    # Área de Bad Debt (Fundo)
    fig1.add_trace(
        go.Bar(x=df["Time"], y=df["Bad_Debt"], name="Bad Debt (Crises)", 
               marker_color="lightgrey", opacity=0.5),
        secondary_y=False
    )

    # Linhas de Alavancagem
    fig1.add_trace(
        go.Scatter(x=df["Time"], y=df["Avg_Leverage_D"], name="Alavancagem Média (D)", 
                   line=dict(color="blue", width=2)),
        secondary_y=True
    )
    fig1.add_trace(
        go.Scatter(x=df["Time"], y=df["Avg_Leverage_U"], name="Alavancagem Média (U)", 
                   line=dict(color="green", width=2, dash="dot")),
        secondary_y=True
    )

    fig1.update_layout(
        title="Ciclo da Fragilidade: Alavancagem vs. Avalanches de Bad Debt",
        template="plotly_white",
        hovermode="x unified"
    )
    fig1.update_yaxes(title_text="Volume de Bad Debt", secondary_y=False)
    fig1.update_yaxes(title_text="Índice de Alavancagem (L)", secondary_y=True)
    fig1.show()

    # ==============================================================================
    # 2. TRANSMISSÃO DE JUROS: Bancos vs Crédito Comercial
    # ==============================================================================
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df["Time"], y=df["Avg_r_bank_u"], name="Custo Banco (U)",
                             line=dict(color="firebrick", width=1.5)))
    
    fig2.add_trace(go.Scatter(x=df["Time"], y=df["Avg_r_trade_u"], name="Custo Comercial (Trade Credit)",
                             line=dict(color="forestgreen", width=1.5)))
    
    # Diferença (Spread) preenchida
    fig2.add_trace(go.Scatter(
        x=df["Time"], y=df["Avg_r_trade_u"],
        fill='tonexty', # Preenche até a linha anterior (Avg_r_bank_u)
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        showlegend=False,
        name="Spread"
    ))

    fig2.update_layout(
        title="Transmissão Financeira:Spread entre Custo Bancário e Crédito Comercial",
        yaxis_title="Taxa de Juros (r)",
        xaxis_title="Tempo",
        template="plotly_white"
    )
    fig2.show()

    # ==============================================================================
    # 3. DIAGRAMA DE FASE: Produção vs Juros (Acelerador Financeiro)
    # ==============================================================================
    tail = 100
    df_tail = df.tail(tail)

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=df_tail["Production"], 
        y=df_tail["Avg_r_bank_d"],
        mode='markers+lines',
        marker=dict(
            size=6,
            color=df_tail["Time"], # Cor evolui com o tempo
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Tempo")
        ),
        line=dict(color='gray', width=0.5, dash='dot'),
        name="Trajetória"
    ))

    fig3.update_layout(
        title=f"Diagrama de Fase (Últimos {tail} períodos): Produção vs Custo do Crédito",
        xaxis_title="Produção Agregada (Y)",
        yaxis_title="Taxa de Juros Média (Downstream)",
        template="plotly_white",
        height=600
    )
    fig3.show()

# Executar
if 'df' in locals():
    plot_diagnosticos_avancados(df)
# %%
# %% Visão 1: A Marcha das Taxas de Juros (Feedback Loop Financeiro)
from plotly.subplots import make_subplots

# Criando figura com dois eixos Y
fig1 = make_subplots(specs=[[{"secondary_y": True}]])

# Eixo Secundário (Barras no fundo): O Choque (Má Dívida)
fig1.add_trace(go.Bar(
    x=df['Time'], y=df['Bad_Debt'], 
    name='Bad Debt (Calotes)', 
    marker_color='red' # Vermelho transparente
), secondary_y=True)

# Eixo Primário (Linhas na frente): A Reação (Taxas de Juros)
fig1.add_trace(go.Scatter(
    x=df['Time'], y=df['Avg_r_trade_u'], 
    mode='lines', name='Juros Comercial (U -> D)', line=dict(color='orange')
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=df['Time'], y=df['Avg_r_bank_u'], 
    mode='lines', name='Juros Banco -> U', line=dict(color='blue')
), secondary_y=False)

fig1.update_layout(
    title='Visão 1: O Ciclo Vicioso dos Juros (Contágio)',
    xaxis_title='Tempo (t)',
    yaxis_title='Taxa de Juros (r)',
    yaxis2_title='Volume de Má Dívida (Bad Debt)',
    template='plotly_white'
)
fig1.show()
# %%
# %% Visão 2: Dinâmica de Alavancagem (Deleveraging Cycle)
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df['Time'], y=df['Avg_Leverage_D'], 
    mode='lines', name='Alavancagem Média D (Dívida/Patrimônio)'
))

fig2.add_trace(go.Scatter(
    x=df['Time'], y=df['Avg_Leverage_U'], 
    mode='lines', name='Alavancagem Média U'
))

# Adicionando um sombreamento onde houve picos de falência (opcional, visual)
avalanche_threshold = df['Count_Def_D'].mean() + 2 * df['Count_Def_D'].std()
for i, row in df.iterrows():
    if row['Count_Def_D'] > avalanche_threshold:
        fig2.add_vrect(
            x0=row['Time']-0.5, x1=row['Time']+0.5, 
            fillcolor="red", opacity=0.1, line_width=0
        )

fig2.update_layout(
    title='Visão 2: Ciclos de Alavancagem (Áreas vermelhas = Avalanches em D)',
    xaxis_title='Tempo (t)',
    yaxis_title='Índice de Alavancagem (l)',
    template='plotly_white'
)
fig2.show()
# %%
# %% Visão 3: Correlação Sistêmica (Heatmap)
import seaborn as sns
import matplotlib.pyplot as plt

# Selecionar variáveis chave para a macroeconomia do modelo
cols_macro = [
    'Production', 'Bad_Debt', 'Avg_r_bank_d', 'Avg_r_trade_u', 
    'Avg_Leverage_D', 'Avg_Leverage_U', 'Count_Def_D', 'Count_Def_U'
]
df_macro = df[cols_macro]

# Renomear para ficar bonito no gráfico
df_macro.columns = [
    'Produção', 'Má Dívida', 'Juros B->D', 'Juros U->D', 
    'Alavancagem D', 'Alavancagem U', 'Falências D', 'Falências U'
]

plt.figure(figsize=(8, 6))
corr = df_macro.corr()

# Plotar o heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
plt.title('Visão 3: Matriz de Correlação Macroeconômica')
plt.tight_layout()
plt.show()
# %%
def plot_diagrama_fase_animado(df, tail=200):
    df_tail = df.tail(tail).copy()
    
    # 1. Definir limites fixos para os eixos não "pularem" durante a animação
    x_min = df_tail["Production"].min() * 0.95
    x_max = df_tail["Production"].max() * 1.05
    y_min = df_tail["Avg_r_bank_d"].min() * 0.95
    y_max = df_tail["Avg_r_bank_d"].max() * 1.05
    
    # Limites da escala de cores (Tempo)
    c_min = df_tail["Time"].min()
    c_max = df_tail["Time"].max()

    # 2. Construir os Quadros (Frames) da animação
    frames = []
    for i in range(1, len(df_tail) + 1):
        # Pega os dados do início da cauda até o período atual 'i'
        df_frame = df_tail.iloc[:i]
        
        frames.append(go.Frame(
            data=[go.Scatter(
                x=df_frame["Production"],
                y=df_frame["Avg_r_bank_d"],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=df_frame["Time"],
                    colorscale="Viridis",
                    cmin=c_min, # Trava a cor para não mudar no meio do caminho
                    cmax=c_max,
                ),
                line=dict(color='gray', width=1, dash='dot')
            )],
            name=str(df_frame["Time"].iloc[-1]) # O nome do frame é o tempo atual
        ))

    # 3. Criar a Figura Base (Estado inicial)
    fig3 = go.Figure(
        data=[go.Scatter(
            x=[df_tail["Production"].iloc[0]], 
            y=[df_tail["Avg_r_bank_d"].iloc[0]],
            mode='markers+lines',
            marker=dict(
                size=6,
                color=[df_tail["Time"].iloc[0]],
                colorscale="Viridis",
                cmin=c_min,
                cmax=c_max,
                showscale=True,
                colorbar=dict(title="Tempo (t)")
            ),
            name="Trajetória"
        )],
        layout=go.Layout(
            title=f"    Diagrama de Fase Animado (Últimos {tail} períodos)",
            xaxis=dict(title="Produção Agregada (Y)", range=[x_min, x_max]),
            yaxis=dict(title="Taxa de Juros Média (Downstream)", range=[y_min, y_max]),
            template="plotly_white",
            height=600,
            # Adiciona os botões de Play e Pause
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.16, # Posição Y dos botões
                x=1.10, # Posição X dos botões
                buttons=[
                    dict(label="► Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 200, "redraw": True}, 
                                      "fromcurrent": True, 
                                      "transition": {"duration": 0}}]),
                    dict(label="❚❚ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, 
                                        "mode": "immediate", 
                                        "transition": {"duration": 0}}])
                ]
            )]
        ),
        frames=frames
    )

    fig3.show()

# Para testar:
plot_diagrama_fase_animado(df)
# %%
