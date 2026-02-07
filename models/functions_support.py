import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx


def plot_network_organic(econ, num_sample_d=30, k_spacing=2.5, iterations=100):
    """
    Plota a rede com layout orgânico.
    
    Novos Parâmetros:
    - k_spacing: Controla a repulsão entre nós (aumente para espalhar mais). 
                 Valor padrão aumentado para 2.5 (era 0.5 hardcoded).
    - iterations: Quantidade de passos da simulação física (padrão 100).
    """
    
    # --- 1. SELEÇÃO DE DADOS (Igual ao anterior) ---
    limit = min(num_sample_d, econ.params.N_d)
    d_indices = np.arange(econ.params.N_d)[:limit]
    
    u_indices_set = set()
    z_indices_set = set()
    
    edges_u_d = [] 
    edges_z_d = []
    edges_z_u = [] 
    
    for i in d_indices:
        # Fornecedores
        suppliers = econ.supplier[i]
        for u in suppliers:
            u_indices_set.add(u)
            edges_u_d.append((f'U{u}', f'D{i}'))
            
        # Bancos (Verifica se bank_links é uma lista de listas ou lista plana)
        # O código original assume bank_links[i] retornando iterável
        if len(econ.bank_links_d) > i: # Proteção de índice
            banks = econ.bank_links_d[i] 
            for z in banks:
                z_indices_set.add(z)
                edges_z_d.append((f'Z{z}', f'D{i}'))
            
    u_indices = list(u_indices_set)
    
    # Conexões de U com Bancos
    if hasattr(econ, 'bank_links_u') and econ.bank_links_u is not None:
        for u in u_indices:
            if len(econ.bank_links_u) > u:
                banks_of_u = econ.bank_links_u[u]
                for z in banks_of_u:
                    z_indices_set.add(z)
                    edges_z_u.append((f'Z{z}', f'U{u}'))
    
    z_indices = list(z_indices_set)
    
    # --- 2. CÁLCULO DO LAYOUT (AJUSTADO PARA ESPAÇO) ---
    G = nx.Graph()
    
    # Adicionar nós
    G.add_nodes_from([f'U{u}' for u in u_indices])
    G.add_nodes_from([f'D{d}' for d in d_indices])
    G.add_nodes_from([f'Z{z}' for z in z_indices])
    
    # Adicionar arestas
    G.add_edges_from(edges_u_d)
    G.add_edges_from(edges_z_d)
    G.add_edges_from(edges_z_u)
    
    # ---------------------------------------------------------
    # AQUI ESTÁ A MUDANÇA PRINCIPAL
    # ---------------------------------------------------------
    # k: Distância ótima. A fórmula teórica é 1/sqrt(n), mas para visualização
    #    espaçada, forçamos um valor alto (ex: 2.0 ou 3.0).
    # scale: Estica o gráfico inteiro.
    pos = nx.spring_layout(G, k=k_spacing, iterations=iterations, seed=42, scale=2)
    
    # DICA: Se spring_layout não ficar bom, tente o algoritmo Kamada-Kawai (mais lento, mas separa melhor):
    # pos = nx.kamada_kawai_layout(G, scale=2) 
    
    # --- 3. PLOTAGEM ---
    fig = go.Figure()

    def add_edge_trace(edge_list, color, dash, name):
        x_coords = []
        y_coords = []
        for node1, node2 in edge_list:
            if node1 in pos and node2 in pos:
                x0, y0 = pos[node1]
                x1, y1 = pos[node2]
                x_coords.extend([x0, x1, None])
                y_coords.extend([y0, y1, None])
        
        # Opacidade (opacity) ajuda a ver quando muitas linhas se cruzam
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            line=dict(width=0.5, color=color, dash=dash), # Linha mais fina (0.5) reduz poluição visual
            hoverinfo='none',
            mode='lines',
            opacity=0.6, 
            name=name
        ))

    add_edge_trace(edges_u_d, '#888', 'solid', 'Trade (U-D)')
    add_edge_trace(edges_z_d, 'purple', 'dot', 'Bank (Z-D)')
    add_edge_trace(edges_z_u, '#FF4500', 'dash', 'Bank (Z-U)')

    def add_node_trace(indices, prefix, color, symbol, name):
        node_x = []
        node_y = []
        text_labels = []
        
        for idx in indices:
            node_id = f'{prefix}{idx}'
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                text_labels.append(node_id)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers', 
            # Diminuí levemente o tamanho (size=12) para reduzir sobreposição
            marker=dict(symbol=symbol, size=12, color=color, line=dict(color='black', width=1)),
            text=text_labels,
            hoverinfo='text',
            name=name
        ))

    add_node_trace(u_indices, 'U', 'lightgreen', 'square', 'Upstream')
    add_node_trace(d_indices, 'D', 'lightblue', 'circle', 'Downstream')
    add_node_trace(z_indices, 'Z', 'yellow', 'triangle-up', 'Bancos')

    fig.update_layout(
        title="Rede Financeira",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        # Aumentar a altura ajuda a dar espaço físico para os nós
        height=800, 
        width=1000
    )
    
    fig.show()

def probabilities_extreme_event(bad_debt_data):
    bd_data = np.array(bad_debt_data)

    median_bd = np.median(bd_data)
    bd_prime = np.abs(bd_data - median_bd)
    sigma_bd = np.std(bd_data)

    x_values = np.linspace(0, 10, 100) 
    prob_y = []


    for x in x_values:
        threshold = x * sigma_bd
        count_extremes = np.sum(bd_prime > threshold)
        probability = count_extremes / len(bd_prime)
        prob_y.append(probability)

    x_values = np.array(x_values)
    prob_y = np.array(prob_y)
    mask = np.array(prob_y) > 0

    x_plot = x_values[mask]
    y_plot = prob_y[mask]

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode='markers',
        name='Prob(BD\' > xσ)',
        marker=dict(size=8, color='blue')
    ))

    fig4.update_layout(
        title="Figura 4: Agregado de má dívida - Probabilidade de Eventos Extremos",
        xaxis_title="x (Múltiplos do Desvio Padrão)",
        yaxis_title="log(Probabilidade)",

        yaxis=dict(
            type='log',
            dtick=1,
            exponentformat='power',
            showexponent='all'),

        template='plotly_white'
    )

    fig4.show()
