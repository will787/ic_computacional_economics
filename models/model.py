# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# ============================================================
# PARAMETROS
# ============================================================

class Params:
    N_d = 50 # downstream - i
    N_u = 25 # upstream - j
    N_z = 10 # bancos

    phi = 1.2
    beta = 0.8
    delta_d = 0.5
    delta_u = 1.0
    gamma = 0.5
    alpha = 0.1
    sigma = 0.05
    theta = 3
    w = 1
    M = 5 # parametro relacao D -> U (formacao de crédito comercial e fornecimento de bens) - firmas d buscam firmas u
    N = 5 # parametro relacao D e U -> Bancos (Z) rede formacao de empréstimo bancarios
    Z = 2 #banco por firma
    e = 0.01
    #p_jt = 0.4
    #r_b = 0.02


# ============================================================
# FUNÇÕES DE PRODUÇÃO DOWNSTREAM - BENS FINAIS
# ============================================================

#def Yit(A_it, beta):
#    """"
#        Y_it = ϕ * A_it^β
#    """
#    return (A_it ** beta)

def uit(size=1):
    """"
        uit ~ U(0, 2)
        preço estocastico, exogeno e aleatorio do bem final
        distribuição uniforme entre 0 e 2
    """
    return np.random.uniform(0, 2, size)


def Qit(A_it, gamma):
    """"
        Q_it = γ * (ϕ * A_it^β) antiga funcao
        Q_it = γ * A_it - bens intermediarios necessarios 
    """
    return gamma * A_it

def Nit(Y_it, delta_d):
    return delta_d * Y_it

def Leontief(gamma, delta_d, Nit, Q_supply):
    """"
        Y_it_eff = min(N_it / δ_d, Q_it / γ)
    """
    Y_labor = Nit / delta_d
    Y_intermediate = Q_supply / gamma
    return np.minimum(Y_labor, Y_intermediate)

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

def Q_supply_from_upstream(A_d, upstream_ids, A_u, phi, beta):
    """
        Oferta efetiva de bens intermediários para cada firma i
        depende da saude financeira dos fornecedores
    """
    return sum(phi * (A_u[j] ** beta) for j in upstream_ids)
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

def rjz(self, B_upstream):
    

    p = self.params
    r_jz_list = np.zeros(p.N_u)

    for j in range(p.N_u):
        banks = self.bank_links_u[j]

        B_j = B_upstream[j]
        A_j = self.A_u[j]

        if len(banks) == 0 or A_j <= 0:
            r_jz_list[j] = 0.05
            continue

        ljt = B_j / A_j  # razão entre empréstimo bancário e patrimônio líquido da firma upstream j.

        r_banks = []
        for z in banks:
            A_z = self.A_z[z]

            term_bank = p.sigma * (A_z ** (-p.sigma))
            term_risk = p.theta * (ljt ** p.theta)
            r_banks.append(term_bank + term_risk)

        r_jz_list[j] = np.mean(r_banks)


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


## testes de novas funcoes

def Y_potential(A_it, beta, phi):
    """"
        Y_it_potencial = ϕ * A_it^β
        Produção potencial antes gargalos
    """
    return phi * (A_it ** beta)

def N_demand(Y_pot, delta_d):
    """"
        N_it_demand = δ_d * Y_it_potencial
        Demanda de trabalho para reduçao de Y potencial
    """
    return delta_d * Y_pot

def Q_demand(Y_pot, gamma):
    """"
        Q_it_demand = γ * Y_it_potencial
        Demanda de bens intermediarios para reduçao de Y potencial
    """
    return gamma * Y_pot

def q_upstream(A_j, phi, beta):
    """
        q_jt = ϕ * A_j^β
        Produçao de bens intermediarios de uma firma upstream j
    """
    return phi * (A_j ** beta)

def Q_supply_i(i, supplier_links, A_u, phi, beta):
    """
        Soma da produção dos fornecedores de i (downstream)
    """
    return sum(
        q_upstream(A_u[j], phi, beta)
        for j in supplier_links[i]
    )

def Leontief_new(Y_pot, N_it, Q_demand_it, Q_supply_it, delta_d, gamma):
    """
        Produçao efetiva com gargalos
    """
    Y_labor = N_it / delta_d
    Y_intermediate = Q_supply_it / gamma
    return np.minimum(Y_labor, Y_intermediate)


# ============================================================
# ECONOMIA
# ============================================================

class Economy:

    def __init__(self, params, supplier=None, bank_links=None, bank_links_u=None):
        self.params = params
        self.supplier = supplier
        self.bank_links = bank_links
        self.bank_links_u = bank_links_u

        self.A_d = np.ones(params.N_d)  # downstream começam com patrimônio líquido igual a 1
        self.A_u = np.ones(params.N_u)  # fornecedores começam com patrimônio líquido igual a 1
        self.A_z = np.ones(params.N_z)  # bancos começam com patrimônio líquido igual a 1


        self.history = {
            "Y": [],
            "Revenue": [],
            "A_d": [],
            "profits_d": [],
            "Bad debt": []}

    def production_downstream(self):
        p = self.params
        #A_safe = np.maximum(self.A_d, 1e-6)  # evita capital negativo
        Y_pot = Y_potential(self.A_d, p.beta, p.phi)
        
        Q_d = Q_demand(self.A_d, p.gamma)
        N_d = N_demand(Y_pot, p.delta_d)

        Q_s = np.zeros(p.N_d)
        for i in range(p.N_d):
            Q_s[i] = Q_supply_i(
                i,
                self.supplier,
                self.A_u,
                p.phi,
                p.beta
            )
        
        #producao efetiva (efeito do gargalo)
        Y_eff = np.array([
        Leontief_new(
            Y_pot[i],
            N_d[i],
            Q_d[i],
            Q_s[i],
            p.delta_d,
            p.gamma
        )
        for i in range(p.N_d)
    ])
        u = uit(p.N_d)
        revenue = u * Y_eff
        return Y_pot, Q_d, N_d, Y_eff, u, revenue

    def credit_demand(self, Y, N):
        p = self.params
        w_vector = p.w * np.array(N)
        return Bxt(w_vector, self.A_d)

    def profits_downstream(self, Y_eff, B_down, u_vec):
        p = self.params
        prices = uit(len(Y_eff))
        #r_b = [p.r_b] * len(B_down)
        
        rj_list = np.zeros(p.N_d)
        for i in range(p.N_d):
            upstream_j = self.supplier[i][0]  # escolhe primeiro supplier
            rj_list[i] = rj(self.A_u[upstream_j], p.alpha)

        profits = pi_it(
            p_jt_list=u_vec,
            Yit_list=Y_eff,
            riz_list= riz(self, B_down),#ajustar a funcao
            Bit_list=B_down,
            Qit_list=Qit(self.A_d, p.gamma),
            rj_list=rj_list
        )

        return profits
    
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
        new_A= self.A_d + profits

        BD = np.where(self.A_d < 0, -new_A, 0) #bad debt

        #correcao do patrimonio liquido apos bad debt
        self.A_d = new_A - BD 
        bankrupt = np.where(self.A_d <= 0)[0]
        if len(bankrupt) > 0:
            self.A_d[bankrupt] = np.random.uniform(0.8, 1.2, len(bankrupt))

        bankrupt_u = np.where(self.A_u <= 0)[0]
        if len(bankrupt_u) > 0:
            self.A_u[bankrupt_u] = np.random.uniform(0.8, 1.2, len(bankrupt_u))
        
        bankrupt_z = np.where(self.A_z <= 0)[0]
        self.A_z[bankrupt_z] = np.random.uniform(0.8, 1.2, len(bankrupt_z))

    def choose_with_noise(candidates, scores, sample_size, eps):
        """"
            candidatos: lista dos possiveis
            scores: sao os preços estocasticos
            sample_size: tamanho da relacao M(U) ou N(bancos)
            eps: prob de escolha aleatoria
        """

        if np.random.rand() < eps:
            # escolha aleatoria
            return np.random.choice(candidates, size=sample_size, replace=False)

        sample = np.random.choice(candidates, size=sample_size, replace=False)

        sample_scores = [scores[i] for i in sample]
        best = sample[np.argmin(sample_scores)]  # menor preço

        return best

    def update_supplier_links(self):
        p = self.params
        new_supplier = []

        for i in range(p.N_d):
            current = self.supplier[i][0]
            candidates = np.arrange(p.N_u)
            prices = p_jt(self.A_u, p.alpha)

            chosen = Economy.choose_with_noise(
                candidates=candidates,
                scores=prices,
                sample_size=p.M,
                eps=p.e
            )

            new_supplier.append([chosen])

        self.supplier = new_supplier


    def update_bank_links(self):
        p = self.params
        new_bank_links = []

        for i in range(p.N_d):
            candidates = np.arrange(p.N_z)
            prices = riz(self, B_down=np.zeros(p.N_d))  # placeholder para B_down

            chosen_banks = Economy.choose_with_noise(
                candidates=candidates,
                scores=prices,
                sample_size=p.Z,
                eps=p.e
            )

            new_bank_links.append([chosen_banks])

        self.bank_links = new_bank_links

    def simulate(self, T):
        for t in range(T):

            #producao downstream
            Y, Q, N, Y_eff, u_vec, revenue = self.production_downstream()

            #credito
            B_down = self.credit_demand(Y, N)

            #lucros
            profits = self.profits_downstream(Y_eff, B_down, u_vec)
        
            #atualizacao do patrimonio liquido
            self.update_A(profits)

            self.history["Y"].append(Y_eff.copy())
            self.history["Revenue"].append(revenue.copy())
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

def choose_preferred_upstream(A_u, M):
    """"
        Escolhe os upstream fornecedores upstream com maior patrimônio líquido A_u.
    """
    weights = A_u / np.sum(A_u)
    preferred_indices = np.random.choice(len(A_u), size=M, replace=False, p=weights)
    return preferred_indices

def choose_preferred_banks(A_z, Z):
    """"
        Escolhe Z bancos com probabilidade proporcional ao patrimonio liquido A_z
    """
    weights = A_z / np.sum(A_z)
    preferred_indices = np.random.choice(len(A_z), size=Z, replace=False, p=weights)
    return preferred_indices


if __name__ == "__main__":

    p = Params()
    econ = Economy(p, supplier = None, bank_links = None)
    #1. Downstream -> upstream
    supplier = [
        list(choose_preferred_upstream(econ.A_u, p.M))
        for _ in range(p.N_d)
    ]

    upstream_to_downstream = [[] for _ in range(p.N_u)]
    for i in range(p.N_d):
        for j in supplier[i]:
            upstream_to_downstream[j].append(i)

    # Banks -> D-U
    banks_links_d = [
        choose_preferred_banks(econ.A_z, p.Z)
        for _ in range(p.N_d)
    ]

    banks_links_u = [
        choose_preferred_banks(econ.A_z, p.Z)
        for _ in range(p.N_u)
    ]

    econ.supplier = supplier
    econ.bank_links = banks_links_d
    econ.bank_links_u = banks_links_u
    econ.simulate(T=1000)
    

# 1 model

rev_by_period = np.array(econ.history["Y"])
agg_revenue_t = rev_by_period.sum(axis=1)
log_rev = np.log10(np.maximum(agg_revenue_t, 1e-12))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(len(rev_by_period)),
    y=log_rev,
    mode='lines'
))
fig.update_yaxes(title="log(Y)")
fig.update_xaxes(title="t")
fig.show()

# 2 model

A_final = np.array(econ.history["A_d"][-1])
A_final = A_final[A_final > 0]

A_sorted = np.sort(A_final)[::-1]
ranks = np.arange(0, len(A_sorted) + 1)

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