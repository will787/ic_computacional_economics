# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# PARAMETROS
# ============================================================

class Params:
    N_d = 500 # downstream - i
    N_u = 250 # upstream - j
    N_z = 100 # bancos

    phi = 1.2
    beta = 0.8
    delta_d = 0.5
    delta_u = 1
    gamma = 0.5
    alpha = 0.1
    sigma = 0.05
    theta = 1
    w = 1
    M = 5 # parametro relacao D -> U (formacao de crédito comercial e fornecimento de bens) - firmas d buscam firmas u
    #N = 5 # parametro relacao D e U -> Bancos (Z) rede formacao de empréstimo bancarios
    Z = 2 #banco por firma
    e = 0.01
    p_jt = 0.4
    r_b = 0.02


# ============================================================
# FUNÇÕES DE PRODUÇÃO DOWNSTREAM - BENS FINAIS
# ============================================================

def Yit(A_it, beta, phi):
    """"
        Y_it = ϕ * A_it^β
    """
    return phi * (A_it ** beta)

def uit(size=1):
    """"
        uit ~ U(0, 2)
        preço estocastico, exogeno e aleatorio do bem final
    """
    return np.random.uniform(0, 2, size)


def Qit(A_it, gamma, phi, beta):
    """"
        Q_it = γ * (ϕ * A_it^β)
    """
    return gamma * (phi * (A_it ** beta))

def Nit(Y_it, delta_d):
    return delta_d * Y_it

def Leontief(gamma, delta_d, N, Q):
    """"
        Y_it_eff = min(N_it / δ_d, Q_it / γ)
    """
    if delta_d > 0 and gamma > 0:
        return np.minimum(N / delta_d, Q / gamma)
    return 0

# ============================================================
# PRODUÇÃO UPSTREAM - BENS INTERMEDIÁRIOS (fornecedores para downstream)
# ============================================================

def Qjt(params, A_d, upstream_to_downstream):
    """"
        Qjt = γ * ϕ * Σ_{i ∈ S_j} A_it^β
        Quantidade de bens intermediarios produzidos por cada firma upstream j.
    """
    Q_u = np.zeros(params.N_u)
    for j in range(params.N_u):
        S_j = upstream_to_downstream[j]
        soma = sum((A_d[i]**params.beta) for i in S_j)
        Q_u[j] = params.gamma * params.phi * soma
    return Q_u

def Njt(delta_u, gamma, phi, phi_sets, beta, A_it):
    """"
        N_jt = δ_u * γ * ϕ * Σ_{i ∈ S_j} A_it^β     
    """
    total = 0
    for S_j in phi_sets:
        total += delta_u * gamma * phi * sum((A_it[i] ** beta) for i in S_j)
    return total

def Leontief_u(delta_u, gamma, phi, phi_sets, beta, A_it):
    """"
        Leontief_jt = N_jt / δ_u = (δ_u * γ * ϕ * Σ_{i ∈ S_j} A_it^β) / δ_u
        Tecnologia de produção Leontief, utilizando apenas trabalho
    """
    if delta_u > 0:
        N = Njt(delta_u, gamma, phi, phi_sets, beta, A_it) / delta_u
        return (N / delta_u)
    return 0

# ============================================================
# TAXAS E CRÉDITO
# ============================================================

def rj(At, alpha):
    """"
        r_jt = α * A_t^(-α)
        rjt é a taxa de juros no crédito comercial
        o nível de rjt depende da condição financeira da firma upstream.
    """
    if alpha > 0:
        return alpha * At ** (-alpha)
    return 0

def Bxt(w_vector, A_vector):

    gap = w_vector - A_vector

    demanda_credito = np.where(gap > 0, gap, 0) #caso precise de dinheiro ele retorna 0
    print(demanda_credito)

    return demanda_credito

def riz(self, B_down):
    """"
        r_izt = σ * A_zt^(-σ) + θ * B_xt^(θ)
        taxa de juros do empréstimo bancário para cada firma downstream
    """
    p = self.params

    r_iz_list = np.zeros(p.N_d)
    
    for i in range(p.N_d):
        banks = self.bank_links[i]
        B_i = B_down[i]
        A_i = self.A_d[i]

        if len(banks) == 0:
            r_iz_list[i] = 0
            continue

        lit = B_i / A_i  # razão entre empréstimo bancário e patrimônio líquido da firma downstream i.

        r_banks = []
        for z in banks: 
            A_z = self.A_z[z]

            term_bank = p.sigma * (A_z ** (-p.sigma))
            term_risk = p.theta * (lit ** p.theta)
            r_banks.append(term_bank + term_risk)

        r_iz_list[i] = np.mean(r_banks)
    
    return r_iz_list


def lit(B_i, A_i):
    """
        l_it = B_it / A_it
        Define a razão entre o empréstimo bancário e o patrimônio líquido da firma downstream i.
        um indice de alavancagem: relação entre crédito demanda (Bit) e patrimônio líquido (Ait).
    """
    return B_i / A_i
    
# ============================================================
# PREÇOS
# ============================================================

def p_jt(At, Alpha):
    if Alpha > 0:
        return 1 + (rj(At, Alpha))
    return 0

# ============================================================
# LUCROS
# ============================================================

def pi_it(p_jt_list, Yit_list, riz_list, Bit_list, Qit_list, rj_list):
    profits = []
    for p_jt, Y_it, r_izt, Bi, Qi, r_jt in zip(
        p_jt_list, Yit_list, riz_list, Bit_list, Qit_list, rj_list
    ):
        profit = p_jt * Y_it - (1 + r_izt) * Bi - (1 + r_jt) * Qi
        profits.append(profit)
    return np.array(profits)


# ============================================================
# ECONOMIA
# ============================================================

class Economy:

    def __init__(self, params, supplier, bank_links):
        self.params = params
        self.supplier = supplier
        self.bank_links = bank_links


        self.A_d = np.random.uniform(0.5, 1.5, params.N_d)
        self.A_u = np.random.uniform(0.5, 1.5, params.N_u)
        self.A_z = np.random.uniform(0.5, 1.5, params.N_z)


        self.history = {"Y": [], "A_d": [], "profits_d": []}

    def production_downstream(self):
        p = self.params
        A_safe = np.maximum(self.A_d, 1e-6)  # evita capital negativo
        Y = Yit(A_safe, p.beta, p.phi)
        Q = Qit(A_safe, p.gamma, p.phi, p.beta)
        N = Nit(Y, p.delta_d)
        Y_eff = Leontief(p.gamma, p.delta_d, N, Q)
        return Y, Q, N, Y_eff

    def credit_demand(self, Y, N):
        p = self.params
        w_vector = p.w * np.array(N)
        return Bxt(w_vector, self.A_d)

    def profits_downstream(self, Y_eff, B_down):
        p = self.params
        prices = uit(len(Y_eff))
        #r_b = [p.r_b] * len(B_down)
        
        rj_list = np.zeros(p.N_d)
        for i in range(p.N_d):
            upstream_j = self.supplier[i][0]  # escolhe primeiro supplier
            rj_list[i] = rj(self.A_u[upstream_j], p.alpha)

        profits = pi_it(
            p_jt_list=prices,
            Yit_list=Y_eff,
            riz_list= riz(self, B_down),#ajustar a funcao
            Bit_list=B_down,
            Qit_list=Qit(self.A_d, p.gamma, p.phi, p.beta),
            rj_list=rj_list
        )

        return np.array(profits)
    
    def propagate_bad_debt(self, profits, B_down, Q):
        p = self.params
        new_A = self.A_d + profits

        #identificação do bad debt por defaulting
        default_idx = np.where(new_A < 0)[0]
        BD_per_down = np.zeros(p.N_d)
        BD_per_down[default_idx] = -new_A[default_idx]


        #alocacao bd para credores proporcional a exposicao
        loss_to_upstream = np.zeros(p.N_d)
        loss_to_banks = np.zeros(p.N_z)

        for i in default_idx:
            bd = BD_per_down[i]

            #exposicoes
            expo_banks = B_down[i]
            expo_up = Q[i]

            total_expo = expo_banks + expo_up
            if total_expo <= 0:
                continue
            
            #fracionamento para escoamentos sobre fornecedores e bancos
            frac_up = expo_up / total_expo
            frac_banks = expo_banks / total_expo

            loss_up = bd * frac_up
            loss_banks = bd * frac_banks

            suppliers = self.supplier[i]
            if len(suppliers) > 0:
                per_supplier_expo = expo_up / len(suppliers)
                for j in suppliers:
                    loss_to_upstream[j] += loss_up * (per_supplier_expo / expo_up)

            banks = self.bank_links[i]
            if len(banks) > 0:
                per_bank_expo = loss_banks / len(banks)
                for z in banks:
                    loss_to_banks[z] += per_bank_expo


        #perdas nos credores
        #reduz patriminoio das upstream e bancos

        self.A_u = self.A_u - loss_to_upstream
        self.A_z = self.A_z - loss_to_banks

        self.A_d = new_A - BD_per_down

        # substituicao de agentes por novos entrantes
        bankrupt_d = np.where(self.A_d <= 0)[0]
        if len(bankrupt_d) > 0:
            self.A_d[bankrupt_d] = np.random.uniform(0.5, 1.5, len(bankrupt_d))

        bankrupt_u = np.where(self.A_u <= 0)[0]
        if len(bankrupt_u) > 0:
            self.A_u[bankrupt_u] = np.random.uniform(0.5, 1.5, len(bankrupt_u))

        bankrupt_z = np.where(self.A_z <= 0)[0]
        if len(bankrupt_z) > 0:
            self.A_z[bankrupt_z] = np.random.uniform(0.5, 1.5, len(bankrupt_z))

        return

    def update_A(self, profits):
        new_A= self.A_d + (profits)
        BD = np.where(self.A_d < 0, -new_A, 0) #bad debt
        self.A_d = new_A - BD 
        bankrupt = np.where(self.A_d <= 0)[0]
        if len(bankrupt) > 0:
            self.A_d[bankrupt] = np.random.uniform(0.5, 1.5, len(bankrupt))


    def simulate(self, T):
        for t in range(T):
            Y, Q, N, Y_eff = self.production_downstream()

            B_down = self.credit_demand(Y, N)
            profits = self.profits_downstream(Y_eff, B_down)

            self.update_A(profits)

            self.history["Y"].append(Y_eff.copy())
            self.history["A_d"].append(self.A_d.copy())
            self.history["profits_d"].append(profits)


# ============================================================
# REDES
# ============================================================

def degree_distribution_new(supplier, N_u, M):
    down_deg = np.array([len(supplier[i]) for i in range(N_u)])
    up_deg = np.zeros(M)
    for i in range(N_u):
        for j in supplier[i]:
            up_deg[j] += 1

    return down_deg, up_deg



def bank_degree(A_vector):
    return np.ones(len(A_vector))  # cada firma liga ao banco uma vez

if __name__ == "__main__":

    p = Params()

    #1. Downstream -> upstream
    supplier = [
        list(np.random.choice(p.N_u, size=p.M, replace=False))
        for _ in range(p.N_d)
    ]

    upstream_to_downstream = [[] for _ in range(p.N_u)]
    for i in range(p.N_d):
        for j in supplier[i]:
            upstream_to_downstream[j].append(i)

    # Banks -> D-U
    bank_links = [
        np.random.choice(p.N_z, size=p.Z, replace=False)
        for _ in range(p.N_d)
    ]
    econ = Economy(p, supplier, bank_links)
    econ.simulate(T=100)
    Y_series = econ.history["Y"]
    A_series = econ.history["A_d"]

# %% 1 model

Y_by_period = np.array(econ.history["Y"])      # shape (T, N_d)
Y_total_t = Y_by_period.sum(axis=1)
Y_by_period = np.array(econ.history["Y"])
fig = go.Figure()
fig.add_trace(go.Scatter(y=Y_total_t, mode="lines"))
fig.update_yaxes(type="log", title="log(Y)")
fig.update_xaxes(title="t")
fig.update_layout(title="Aggregate Production (Downstream)")
fig.show()


# 2 model

A_final = np.array(econ.history["A_d"][-1])
A_final = A_final[A_final > 0]

A_sorted = np.sort(A_final)[::-1]
ranks = np.arange(1, len(A_sorted) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(A_sorted),
    y=np.log10(ranks),
    mode='markers'
))
fig.update_xaxes(title="log(firms' net worth)")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(b) Firm Size Distribution (Rank-Size)")
fig.show()


# 3 model

down_deg, up_deg = degree_distribution_new(supplier, p.N_d, p.N_u)

# ---- Downstream degree ----
d_sorted = np.sort(down_deg)[::-1]
rank_d = np.arange(1, len(d_sorted) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(d_sorted),
    y=np.log10(rank_d),
    mode="markers"
))
fig.update_xaxes(title="log(number of links)")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(c1) Degree Distribution — Downstream")
fig.show()


# 4 model
bank_deg = bank_degree(A_final)
bank_deg = bank_deg[bank_deg > 0]

deg_sorted = np.sort(bank_deg)[::-1]
rank = np.arange(1, len(deg_sorted) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(deg_sorted),
    y=np.log10(rank),
    mode="markers"
))
fig.update_xaxes(title="log(number of bank links)")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(d) Degree Distribution — Banks")
fig.show()
# %%
print("Downstream degrees:", np.unique(down_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))


#tentar ver o que esta acontecendo no grafico (C)


