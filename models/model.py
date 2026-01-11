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
    N = 5 # parametro relacao D e U -> Bancos (Z) rede formacao de empréstimo bancarios
    Z = 5 #banco por firma
    e = 0.01
    #p_jt = 0.4
    #r_b = 0.02


# ============================================================
# FUNÇÕES DE PRODUÇÃO DOWNSTREAM - BENS FINAIS
# ============================================================

def Yit(A_it, beta, phi):
    """"
        Y_it = ϕ * A_it^β
    """
    return (A_it ** beta) * phi

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

def rjt(At, alpha): #ok
    """"
        r_jt = α * A_t^(-α)
        rjt é a taxa de juros no crédito comercial
        o nível de rjt depende da condição financeira da firma upstream.
    """
    if alpha > 0:
        return alpha * At ** (-alpha)
    return 0

def calculate_loan_rate(A_bank, leverage_borrower, sigma, theta):
    """
    Taxa de juros bancária geral [cite: 177]
    r = sigma * A_z^(-sigma) + theta * (leverage)^theta
    """
    A_z_safe = np.maximum(A_bank, 1e-6)
    l_safe = np.maximum(leverage_borrower, 0) # Leverage não pode ser negativa
    
    term_bank = sigma * (A_z_safe ** (-sigma))
    term_risk = theta * (l_safe ** theta)
    return term_bank + term_risk

def Bxt(w_vector, A_vector):

    gap = w_vector - A_vector

    demanda_credito = np.where(gap > 0, gap, 0) #caso precise de dinheiro ele retorna 0
    #print(demanda_credito)

    return demanda_credito

def riz(self, B_down):
    """"
        r_izt = σ * A_zt^(-σ) + θ * B_xt^(θ)
        taxa de juros do empréstimo bancário para cada firma downstream
    """
    p = self.params

    r_iz_list = np.zeros(p.N_d)
    
    for i in range(p.N_d):
        banks = self.bank_links_d[i]
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
    """
        calcula a taxa de juros que os bancos cobram das firmas Upstream
        r_zt = sigma * A_zt^(-sigma) + theta * l_jt^(theta)
    """
    
    p = self.params
    r_jz_list = np.zeros(p.N_u)

    for j in range(p.N_u):
        banks = self.bank_links_u[j]
        B_j = B_upstream[j]
        A_j = self.A_u[j]

        if len(banks) == 0:
            r_jz_list[j] = 0.0
            continue
        if B_j == 0:
            r_jz_list[j] = 0.0
            continue

        net_worth = A_j if A_j > 1e-6 else 1e-6

        ljt = B_j / net_worth  #alavancagem da firma U (eq. 9 do artigo)

        r_banks = []
        for z in banks:
            A_z = self.A_z[z]

            bank_net_worth = A_z if A_z > 1e-6 else 1e-6

            term_bank = p.sigma * (bank_net_worth ** (-p.sigma))
            term_risk = p.theta * (ljt ** p.theta)

            r_banks.append(term_bank + term_risk)

        r_jz_list[j] = np.mean(r_banks)

    return r_jz_list


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

def pjt(At, Alpha): #ok
    if Alpha > 0:
        return 1 + (rjt(At, Alpha))
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
    total = 0
    for j in supplier_links[i]:
        #j = int(j) #blindagem robusta de indice
        total += q_upstream(A_u[j], phi, beta)
    return total

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

    def __init__(self, params, supplier=None, bank_links_d=None, bank_links_u=None):
        self.params = params
        self.supplier = supplier
        self.bank_links_d = bank_links_d
        self.bank_links_u = bank_links_u

        self.A_d = np.ones(params.N_d)  # downstream começam com patrimônio líquido igual a 1
        self.A_u = np.ones(params.N_u)  # fornecedores começam com patrimônio líquido igual a 1
        self.A_z = np.ones(params.N_z)  # bancos começam com patrimônio líquido igual a 1


        self.history = {
            "Y": [],
            "Revenue": [],
            "A_d": [],
            "A_u": [],
            "profits_d": [],
            "Bad debt": [],
            "deg_down": [],
            "deg_up": []
        }

    def step_production(self):
        p = self.params
        #A_safe = np.maximum(self.A_d, 1e-6)  # evita capital negativo
        A_d_safe = np.maximum(self.A_d, 1e-6)
        Y_d = p.phi * (A_d_safe ** p.beta)

        N_d_demand = p.delta_d * Y_d
        Q_d_demand = p.gamma * Y_d

        u_prices = np.random.uniform(0, 2, p.N_d)
        revenue_d = u_prices * Y_d

        Q_u_production = np.zeros(p.N_u)

        for i in range(p.N_d):
            u_id = self.supplier[i][0]
            Q_u_production[u_id] += Q_d_demand[i]
        
        N_u_demand = p.delta_u * Q_u_production
        
        return Y_d, Q_d_demand, N_d_demand, revenue_d, Q_u_production, N_u_demand
    
    def credit_demand_d(self, Y, N):
        p = self.params
        w_vector = p.w * np.array(N)
        return Bxt(w_vector, self.A_d)
    
    def step_credit_prices_and_rates(self, N_d_demand, N_u_demand):
        """
            Calcula preços (U) e taxas de juros (bancos) com base na saúde financeira
            e na demanda de crédito gerada pela produção

            Ref: Eq. 3 (Pág 9) Eq 6 (Pág 12) do artigo
        """

        p = self.params

        A_u_safe = np.maximum(self.A_u, 1e-6)

        # demanda de crédito comercial
        r_trade_u = p.alpha * (A_u_safe ** (-p.alpha))
        p_intermediate_vec = 1 + r_trade_u

        # GAP FINANCEIRO

        #downstream
        wage_bill_d = p.w * N_d_demand 
        B_d = np.maximum(wage_bill_d - self.A_d, 0)

        #upstream
        wage_bill_u = p.w * N_u_demand
        B_u = np.maximum(wage_bill_u - self.A_u, 0)

        #crédito bancario - calculo das taxas
        # Eq.6 = sigma * A_zt^(-sigma) + theta * l_it^(theta)

        r_bank_d = np.zeros(p.N_d)
        r_bank_u = np.zeros(p.N_u)

        #taxas para D
        for i in range(p.N_d):
            
            z_idx = self.bank_links_d[i]
            A_z_curr = self.A_z[z_idx[0]]

            #calculo de alavancagem
            #l = B / A
            A_d_curr = max(self.A_d[i], 1e-6)
            levarage_i = B_d[i] / A_d_curr

            base_rate = p.sigma * max(A_z_curr, 1e-6) ** (-p.sigma)
            risk_premium = p.theta * (levarage_i ** p.theta)

            r_bank_d[i] = base_rate + risk_premium

        # taxas para U

        for j in range(p.N_u):
            z_idx = self.bank_links_u[j]
            A_z_curr = self.A_z[z_idx[0]]

            #calculo de alavancagem
            #l = B / A
            A_u_curr = max(self.A_u[j], 1e-6)
            levarage_j = B_u[j] / A_u_curr

            base_rate = p.sigma * max(A_z_curr, 1e-6) ** (-p.sigma)
            risk_premium = p.theta * (levarage_j ** p.theta)

            r_bank_u[j] = base_rate + risk_premium
        
        return p_intermediate_vec, B_d, r_bank_d, B_u, r_bank_u, wage_bill_d, wage_bill_u
                            
    
    def step_profits_and_dynamics(self, Y_d, revenue_d, Q_d_demand, price_intermediate, B_d, r_bank_d, B_u,
                                  r_bank_u, Q_u_production, wage_bill_d, wage_bill_u):
        
        p = self.params

        costs_financial = (1 + r_bank_d) * B_d

        cost_trade_d = np.zeros(p.N_d)
        for i in range(p.N_d):
            u_idx = self.supplier[i][0]
            cost_trade_d[i] = price_intermediate[u_idx] * Q_d_demand[i]

        profits_d = revenue_d - wage_bill_d - (r_bank_d * B_d) - cost_trade_d

        A_d_pre_default = np.zeros(p.N_d)
        for i in range(p.N_d):
            equity_remanescente = max(self.A_d[i] - wage_bill_d[i], 0)

            trade_cost = cost_trade_d[i]
            bank_cost = (1 + r_bank_d[i]) * B_d[i]

            A_d_pre_default[i] = revenue_d[i] + equity_remanescente - bank_cost - trade_cost

        
        revenue_u = np.zeros(p.N_u)
        for i in range(p.N_u):
            #receita de vendas para downstream
            u_idx = self.supplier[i]
            revenue_u[u_idx] = cost_trade_d[i]

        A_u_pre_default = np.zeros(p.N_u)
        for j in range(p.N_u):
            equity_remanescente = max(self.A_u[j] - wage_bill_u[j], 0)
            bank_cost = (1 + r_bank_u[j]) * B_u[j]
            A_u_pre_default[j] = revenue_u[j] + equity_remanescente - bank_cost

        return A_d_pre_default, A_u_pre_default, cost_trade_d


    def credit_demand_u(self, Q_supply_list):
        """
            Cálculo da demanda de crédito bancário das firmas upstream
            B_jt = w * N_jt - A_jt
            Onde N_jt = delta_u * Q_jt (tecnologia leontief invertida.)
        """
        p = self.params
        B_up = np.zeros(p.N_u)
        N_up = np.zeros(p.N_u)

        for j in range(p.N_u):
            N_j = p.delta_u * Q_supply_list[j]
            N_up[j] = N_j
            

            wage_bill = p.w * N_j

            gap = wage_bill - self.A_u[j]
            B_up[j] = max(0, gap)

        return B_up, N_up

    def profits_downstream(self, Y_eff, B_down, u_vec, Q_d_effective, p_intermediate_u, r_bank_d):
        """
            Calcula o lucro das firmas Downstream (D)
            Pi_it = u_it * Y_it - (1 + r_zt)*B_it - (1 + r_jt)*Q_it
            Fonte: Equação na página 15 do artigo (Seção 2).
        """
        p = self.params
        profits = np.zeros(p.N_d)

        for i in range(p.N_d):
            u_id = self.supplier[i][0]


            revenue = u_vec[i] * Y_eff[i]
            cost_bank = (1 + r_bank_d[i]) * B_down[i]
            cost_trade = p_intermediate_u[self.supplier[i][0]] * Q_d_effective[i]

            profits[i] = revenue - cost_bank - cost_trade

        return profits
    
    def profits_upstream(self, Q_supply, B_up, p_intermediate_u, r_bank_u):
        """
            π_jt = (1 + r_trade_charges) * Q_jt - (1 + r_bank_charges) * B_jt
            Lucro operacional das firmas U
            (Receita de vendas de bens intermediarios para D) - (Custo do empréstimo bancário)
            Ref: Equação página 16
        """
        p = self.params
        profits = np.zeros(p.N_u)
        for j in range(p.N_u):

            revenue = p_intermediate_u[j] * Q_supply[j]
            cost_bank = (1 + r_bank_u[j]) * B_up[j]
            profits[j] = revenue - cost_bank

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

            banks = self.bank_links_d[i]
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


    def propagate_bad_debtt(self, A_d_new, A_u_new, B_d, costs_trade_d, B_u):
        p = self.params
        
        total_bad_debt = 0.0
        
        # --- Check Default D ---
        defaults_d = np.where(A_d_new < 0)[0]
        
        # Vetores de perda para U e Z
        loss_to_u = np.zeros(p.N_u)
        loss_to_z = np.zeros(p.N_z)
        
        for i in defaults_d:
            bad_debt_val = abs(A_d_new[i]) # O "buraco" no patrimônio
            total_bad_debt += bad_debt_val
            
            # Quem leva o calote? Fornecedor U e Banco Z
            u_idx = self.supplier[i]
            z_idx = self.bank_links_d[i]
            
            # Proporção da dívida
            debt_u = costs_trade_d[i]
            debt_z = B_d[i] # Simplificando: o calote é no principal+juros, mas usamos exposição nominal
            
            total_liabilities = debt_u + debt_z
            
            if total_liabilities > 0:
                share_u = debt_u / total_liabilities
                share_z = debt_z / total_liabilities
                
                loss_to_u[u_idx] += bad_debt_val * share_u
                loss_to_z[z_idx] += bad_debt_val * share_z
        
        # --- Atualiza A_u com as perdas de D ---
        A_u_final = A_u_new - loss_to_u
        
        # --- Check Default U ---
        defaults_u = np.where(A_u_final < 0)[0]
        
        for j in defaults_u:
            bad_debt_val = abs(A_u_final[j])
            total_bad_debt += bad_debt_val
            
            # U deve apenas para bancos
            z_idx = self.bank_links_u[j]
            loss_to_z[z_idx] += bad_debt_val
            
        # --- Atualiza Bancos ---
        # Lucro dos bancos = Juros de quem pagou - Bad Debt de quem quebrou
        # (Simplificação: apenas subtrai bad debt do patrimônio acumulado dos bancos)
        # O banco acumula os juros (lucro) na variação do A_z, mas aqui focamos no choque
        # Vamos assumir que os juros ganhos já entraram no A_z implicitamente? 
        # Melhor: A_z += (Juros Ganhos) - Loss.
        # Cálculo rápido de Juros Ganhos (aprox):
        # Para ser exato, precisaríamos somar juros de todos os não-falidos. 
        # Dado a complexidade, vamos focar no impacto negativo no A_z.
        self.A_z -= loss_to_z
        
        # --- Substituição de Agentes (Re-entry) [cite: 250, 251] ---
        # Firms falidas saem e entram novas com A ~ U(0.8, 1.2)
        # D firms
        self.A_d = A_d_new
        self.A_d[defaults_d] = np.random.uniform(0.8, 1.2, len(defaults_d))
        
        # U firms
        self.A_u = A_u_final
        self.A_u[defaults_u] = np.random.uniform(0.8, 1.2, len(defaults_u))
        
        # Bancos (se quebrarem)
        defaults_z = np.where(self.A_z < 0)[0]
        self.A_z[defaults_z] = np.random.uniform(0.8, 1.2, len(defaults_z))
        
        return total_bad_debt

    def update_A(self, profits):
        new_A = self.A_d + profits

        BD = np.where(self.A_d < 0, -new_A, 0) #bad debt

        #correcao do patrimonio liquido apos bad debt
        self.A_d = new_A - BD 
        bankrupt = np.where(self.A_d <= 0)[0]
        total_bed_debt = np.sum(np.abs(new_A[bankrupt]))
        if len(bankrupt) > 0:
            self.A_d[bankrupt] = np.random.uniform(0.8, 1.2, len(bankrupt))

        bankrupt_u = np.where(self.A_u <= 0)[0]
        if len(bankrupt_u) > 0:
            self.A_u[bankrupt_u] = np.random.uniform(0.8, 1.2, len(bankrupt_u))
        
        bankrupt_z = np.where(self.A_z <= 0)[0]
        self.A_z[bankrupt_z] = np.random.uniform(0.8, 1.2, len(bankrupt_z))
        return total_bed_debt

    def choose_with_noise(candidates, scores, sample_size, eps):
        """"
            candidatos: lista dos possiveis
            scores: sao os preços estocasticos
            sample_size: tamanho da relacao M(U) ou N(bancos)
            eps: prob de escolha aleatoria
        """

        if np.random.rand() < eps:
            return list(np.random.choice(candidates, size=sample_size, replace=False))
        
        sample = np.random.choice(candidates, size=sample_size, replace=False)
        sample_scores = scores[sample]
        best = int(sample[np.argmin(sample_scores)])  # menor preço

        return [best]

    def update_supplier_links(self):
        p = self.params

        current_price_u = 1 + rjt(self.A_u, p.alpha)

        for i in range(p.N_d):
            # regra do ruido
            if np.random.rand() < p.e:
                # escolha aleatória
                self.supplier[i] = [np.random.randint(0, p.N_u)]

            else:
                current_u_idx = self.supplier[i][0]
                current_price = current_price_u[current_u_idx]

                candidates = np.random.choice(p.N_u, size=p.M, replace=False)
                candidates_prices = current_price_u[candidates]

                min_candidate_idx = np.argmin(candidates_prices)
                best_candidate_price = candidates_prices[min_candidate_idx]
                best_candidate_id = candidates[min_candidate_idx]

                if best_candidate_price < current_price:
                    self.supplier[i] = [best_candidate_id]
                else:
                    self.supplier[i] = [current_u_idx]
               


    def update_bank_links_d(self):
        p = self.params

        Az_safe = np.maximum(self.A_z, 1e-6)
        bank_base_rates = p.sigma * (Az_safe ** (-p.sigma))

        for i in range(p.N_d):
            # regra do ruido
            if np.random.rand() < p.e:
                # escolha aleatória
                self.bank_links_d[i] = [np.random.randint(0, p.N_z)]

            else:

                if len(self.bank_links_d[i]) > 0:
                    current_bank_idx = self.bank_links_d[i][0]
                else:
                    current_bank_idx = np.random.randint(0, p.N_z)

                current_rate = bank_base_rates[current_bank_idx]

                candidates = np.random.choice(np.arange(p.N_z), size=p.N, replace=False)
                candidates_rates = bank_base_rates[candidates]

                min_idx = np.argmin(candidates_rates)
                best_candidate_rate = candidates_rates[min_idx]
                best_candidate_id = candidates[min_idx]

                if best_candidate_rate < current_rate:
                    self.bank_links_d[i] = [best_candidate_id]
                else:
                    self.bank_links_d[i] = [current_bank_idx]


    def update_bank_links_u(self):
        p = self.params
        
        Az_safe = np.maximum(self.A_z, 1e-6)
        bank_base_rates = p.sigma * (Az_safe ** (-p.sigma))

        for j in range(p.N_u):
            # regra do ruido
            if np.random.rand() < p.e:
                # escolha aleatória
                self.bank_links_u[j] = [np.random.randint(0, p.N_z)]

            else:

                if len(self.bank_links_u[j]) > 0:
                    current_bank_idx = self.bank_links_u[j][0]
                else:
                    current_bank_idx = np.random.randint(0, p.N_z)

                current_rate = bank_base_rates[current_bank_idx]

                candidates = np.random.choice(np.arange(p.N_z), size=p.N, replace=False)
                candidates_rates = bank_base_rates[candidates]

                min_idx = np.argmin(candidates_rates)
                best_candidate_rate = candidates_rates[min_idx]
                best_candidate_id = candidates[min_idx]

                if best_candidate_rate < current_rate:
                    self.bank_links_u[j] = [best_candidate_id]
                else:
                    self.bank_links_u[j] = [current_bank_idx]


    def update_financial_positions_and_propagate(self, profits_d, profits_u, B_down, Q_d_effective, p_intermediate_u, r_bank_d_list, B_up, r_bank_u_list):
        """
        Atualiza patrimônios e propaga falências (Bad Debt) pela rede.
        Ordem: Downstream -> (impacta) -> Upstream -> (impacta) -> Bancos
        """
        p = self.params
        
        # -------------------------------------------------------
        # 1. ATUALIZAÇÃO DOWNSTREAM (D) E IDENTIFICAÇÃO DE CALOTE
        # -------------------------------------------------------
        self.A_d += profits_d
        
        # Identifica quem quebrou (A < 0)
        default_d_idx = np.where(self.A_d < 0)[0]
        
        # Vetores para acumular perdas repassadas
        loss_to_upstream = np.zeros(p.N_u)
        loss_to_banks = np.zeros(p.N_z)
        
        total_bad_debt_d = 0.0

        for i in default_d_idx:
            bad_debt = abs(self.A_d[i])
            total_bad_debt_d += bad_debt
            
            # Recupera credores
            u_id = self.supplier[i][0] # Fornecedor
            bank_ids = self.bank_links_d[i] # Lista de bancos
            
            # Dívida com Fornecedor = (1 + r_trade) * Q (valor monetário dos bens)
            debt_u = p_intermediate_u[u_id] * Q_d_effective[i]
            
            # Dívida com Banco = (1 + r_bank) * B (empréstimo)
            debt_z = (1 + r_bank_d_list[i]) * B_down[i]
            
            total_debt = debt_u + debt_z
            
            if total_debt > 0:
                # Rateio do prejuízo (Proporcional à exposição)
                share_u = debt_u / total_debt
                share_z = debt_z / total_debt
                
                # Aplica o prejuízo nos acumuladores
                loss_to_upstream[u_id] += bad_debt * share_u
                
                if len(bank_ids) > 0:
                    per_bank_loss = (bad_debt * share_z) / len(bank_ids)
                    for z in bank_ids:
                        loss_to_banks[z] += per_bank_loss

        # -------------------------------------------------------
        # 2. ATUALIZAÇÃO UPSTREAM (U) COM CONTÁGIO
        # -------------------------------------------------------
        
        # U recebe seus lucros operacionais MENOS o calote de D
        self.A_u += profits_u 
        self.A_u -= loss_to_upstream # <--- O CONTÁGIO ACONTECE AQUI [cite: 58, 62]
        
        # Identifica firmas U que quebraram (seja por operação ruim ou por calote de D)
        default_u_idx = np.where(self.A_u < 0)[0]
        
        total_bad_debt_u = 0.0
        
        for j in default_u_idx:
            bad_debt = abs(self.A_u[j])
            total_bad_debt_u += bad_debt
            
            # U só deve para Bancos (neste modelo simplificado)
            bank_ids_u = self.bank_links_u[j]
            
            # Como U só tem dívida bancária relevante para falência aqui, 
            # o banco absorve tudo ou proporcional ao empréstimo se houver
            if len(bank_ids_u) > 0:
                per_bank_loss = bad_debt / len(bank_ids_u)
                for z in bank_ids_u:
                    loss_to_banks[z] += per_bank_loss

        # -------------------------------------------------------
        # 3. ATUALIZAÇÃO BANCOS (Z)
        # -------------------------------------------------------
        profits_z = np.zeros(p.N_z)
        
        for i in range(p.N_d):
            banks = self.bank_links_d[i]
            if len(banks) == 0:
                continue

            interest = r_bank_d_list[i] * B_down[i]
            per_bank = interest / len(banks)


            for z in banks:
                profits_z[z] += per_bank

        
        for j in range(p.N_u):
            banks = self.bank_links_u[j]
            if len(banks) == 0:
                continue

            interest = r_bank_u_list[j] * B_up[j]
            per_bank = interest / len(banks)

            for z in banks:
                profits_z[z] += per_bank


        self.A_z += profits_z
        self.A_z -= loss_to_banks 

        # -------------------------------------------------------
        # 4. REENTRADA (SUBSTITUIÇÃO DE AGENTES FALIDOS)
        # -------------------------------------------------------
        # Conforme o artigo: agentes falidos saem e entram novos com patrimônio aleatório [cite: 250, 251]
        
        # Reset Downstream
        if len(default_d_idx) > 0:
            self.A_d[default_d_idx] = np.random.uniform(0.8, 1.2, len(default_d_idx))
            
        # Reset Upstream
        if len(default_u_idx) > 0:
            self.A_u[default_u_idx] = np.random.uniform(0.8, 1.2, len(default_u_idx))
            
        # Reset Bancos (Opcional, mas recomendado para evitar colapso total da simulação)
        default_z_idx = np.where(self.A_z < 0)[0]
        if len(default_z_idx) > 0:
            self.A_z[default_z_idx] = np.random.uniform(0.8, 1.2, len(default_z_idx))

        # Retorna o total de bad debt para plotar o gráfico (Figura 3 do artigo)
        return total_bad_debt_d + total_bad_debt_u
    

    def get_bank_degrees(self):
        """
            Calculo do grau (numero de clientes de cada banco)
        """
        degrees = np.zeros(self.params.N_z)

        #soma de conexoes para firmas upstream
        if self.bank_links_u is not None:
            for i in range(len(self.bank_links_u)):
                for z_idx in self.bank_links_u[i]:
                    degrees[z_idx] += 1

        if self.bank_links_d is not None:
            for k in range(len(self.bank_links_d)):
                for z_idx in self.bank_links_d[k]:
                    degrees[z_idx] += 1

        return degrees

    

    def simulate(self, T):
        for t in range(1, T + 1):
            if t % 100 == 0: 
                print(f"Time period: {t}")

            #producao downstream
            Y_d, Q_d_demand, N_d_demand, revenue_d, Q_u_production, N_u_demand = self.step_production()

            p_inter, B_d, r_b_d, B_u, r_b_u, wb_d, wb_u = self.step_credit_prices_and_rates(
                N_d_demand=N_d_demand, 
                N_u_demand=N_u_demand
            )

            A_d_new, A_u_new, cost_trade_d = self.step_profits_and_dynamics(
                Y_d=Y_d,
                revenue_d=revenue_d,
                Q_d_demand=Q_d_demand,
                price_intermediate=p_inter,
                B_d=B_d,
                r_bank_d=r_b_d,
                B_u=B_u,
                r_bank_u=r_b_u,
                Q_u_production=Q_u_production,
                wage_bill_d=wb_d,
                wage_bill_u=wb_u
            )

            bd = self.propagate_bad_debtt(A_d_new, A_u_new, B_d, cost_trade_d, B_u)
            
            #como o mercado se organiza
            down_deg, up_deg = degree_distribution_new(self.supplier, self.params.N_d, self.params.N_u)
            
            #historico
            #A_total = self.A_d.sum() + self.A_u.sum() + self.A_z.sum()
            self.history["Y"].append(Y_d.copy())
            self.history["Revenue"].append(revenue_d.copy())
            self.history["A_d"].append(self.A_d.copy())
            self.history["A_u"].append(self.A_u.copy())
            #self.history["profits_d"].append(profits_d)
            self.history["deg_down"].append(down_deg)
            self.history["deg_up"].append(up_deg)
            self.history["Bad debt"].append(bd)

            # atualizacao da rede
            self.update_supplier_links()
            self.update_bank_links_d()
            self.update_bank_links_u()

# ============================================================
# REDES
# ============================================================

def degree_distribution_new(supplier, N_d, M):
    down_deg = np.array([len(supplier[i]) for i in range(N_d)])
    up_deg = np.zeros(M)
    for i in range(N_d):
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
    preferred_indice = np.random.choice(len(A_u), size=M, replace=False, p=weights)
    return preferred_indice

def choose_preferred_banks(A_z, Z):
    """"
        Escolhe Z bancos com probabilidade proporcional ao patrimonio liquido A_z
    """
    weights = A_z / np.sum(A_z)
    preferred_indice = np.random.choice(len(A_z), size=Z, replace=False, p=weights)
    return preferred_indice


if __name__ == "__main__":

    p = Params()

    #1. Downstream -> upstream
    supplier = [
        [np.random.randint(0, p.N_u)]
        for _ in range(p.N_d)
    ]

    # Banks -> Downstream
    banks_links_d = [
        [np.random.randint(0, p.N_z)]
        for _ in range(p.N_d)
    ]
    
    # Banks -> Downstream
    banks_links_u = [
        [np.random.randint(0, p.N_z)]
        for _ in range(p.N_u)
    ]
    econ = Economy(p, supplier = supplier, bank_links_d = banks_links_d, bank_links_u = banks_links_u)
    econ.simulate(T=1000)

    def run_monte_carlo(n_simulations=10, T=1000):
        print(f"--- Iniciando Monte Carlo ({n_simulations} simulações de T={T}) ---")
        
        all_Y = np.zeros((n_simulations, T))
        all_Bad_Debt = np.zeros((n_simulations, T))

        for sim in range(n_simulations):
            if (sim + 1) % 1 == 0: 
                print(f"Executando simulação {sim + 1}/{n_simulations}...")
            
            p = Params()
            
            # 2. Inicialização Aleatória (Fundamental para Monte Carlo)
            supplier = [[np.random.randint(0, p.N_u)] for _ in range(p.N_d)]
            banks_links_d = [[np.random.randint(0, p.N_z)] for _ in range(p.N_d)]
            banks_links_u = [[np.random.randint(0, p.N_z)] for _ in range(p.N_u)]

            # 3. Instancia e Roda
            econ = Economy(p, supplier=supplier, bank_links_d=banks_links_d, bank_links_u=banks_links_u)
            econ.simulate(T=T)
            
            # 4. Extração e Tratamento de Dados
            # Y: O history guarda um vetor de N_d firmas por tempo. Precisamos somar (Agregado).
            y_history = np.array(econ.history["Y"]) 
            # Soma axis=1 para ter o Y total da economia naquele tempo
            agg_y_t = y_history.sum(axis=1)
            
            bad_debt_history = np.array(econ.history["Bad debt"])
            
            # 5. Armazenamento (Garantindo que o tamanho bata com T)
            # Cortamos [:T] caso a simulação tenha gerado um passo extra ou ajuste de índice
            limit = min(T, len(agg_y_t))
            all_Y[sim, :limit] = agg_y_t[:limit]
            
            limit_bd = min(T, len(bad_debt_history))
            all_Bad_Debt[sim, :limit_bd] = bad_debt_history[:limit_bd]

        return all_Y, all_Bad_Debt

# 1 model

rev_by_period = np.array(econ.history["Y"])
agg_revenue_t = rev_by_period.sum(axis=1) #soma por periodo das firmas
log_rev = np.log10(np.maximum(agg_revenue_t, 1e-12))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(len(rev_by_period[rev_by_period > 0]) + 1),
    y=log_rev,
    mode='lines'
))
fig.update_yaxes(title="log(Y)")
fig.update_xaxes(title="t")
fig.update_layout(title="(A) - Agg production of Downstream firms")

fig.show()

# 2 model

A_final = np.array(econ.history["A_d"][-1])
A_final = A_final[A_final > 0]
if len(A_final) > 0:
        A_sorted = np.sort(A_final)[::-1]
        rank = np.arange(1, len(A_final) + 1)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=np.log10(A_sorted), y=np.log10(rank), mode="markers"))
        fig2.update_layout(title="(B) Distribuição de Tamanho das Firmas", 
                           xaxis_title="log(Patrimônio Líquido)", yaxis_title="log(Rank)")
        fig2.show()


# 3 model

down_deg, up_deg = degree_distribution_new(econ.supplier, econ.params.N_d, econ.params.N_u)

active_up_deg = up_deg[up_deg > 0]

up_sorted = np.sort(active_up_deg)[::-1]
rank = np.arange(1, len(active_up_deg) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(up_sorted),
    y=np.log10(rank),
    mode="markers"
))
fig.update_layout(
    title="(c) Distribuição de Grau — Upstream (Clientes por Fornecedor)",
    xaxis_title="log(Number of links)",
    yaxis_title="log(rank)"
)
fig.show()


# 4 model
bank_deg = econ.get_bank_degrees()
print(bank_deg)
bank_deg = bank_deg[bank_deg > 0]

deg_sorted = np.sort(bank_deg)[::-1]
rank = np.arange(1, len(deg_sorted) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(deg_sorted),
    y=np.log10(rank),
    mode="markers",
    name="Bancos"
)) 

fig.update_xaxes(title="log(degree) - Number os clients")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(D) - Degree Distribution - Banks vs (D + U clients)")
fig.show()

# 5 model

bad_debt_series = np.array(econ.history["Bad debt"])
time_steps = np.arange(len(bad_debt_series)) + 1

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_steps,
    y=bad_debt_series,
    mode="lines"
))
fig.update_xaxes(title="t")
fig.update_yaxes(title="Bad debt")
fig.update_layout(title="Bad debt with Banks (D + U)")
fig.show()

# model simulation - monte carlo
results_Y, results_BD = run_monte_carlo(n_simulations=10, T=1000)

mean_Y = np.mean(results_Y, axis=0)
std_Y = np.std(results_Y, axis=0)

# simulacao producao agg log
fig = go.Figure()
t_index = np.arange(1, 1001)
for k in range(min(10, len(results_Y))):
    fig.add_trace(go.Scatter(
        x=t_index, 
        y=np.log10(np.maximum(results_Y[k, :], 1e-12)),
        mode='lines',
        line=dict(width=1, color='rgba(0,0,255,0.2)'), # Azul transparente
        showlegend=False
    ))

fig.add_trace(go.Scatter(
    x=t_index,
    y=np.log10(np.maximum(mean_Y, 1e-12)),
    mode='lines',
    name='Média (Monte Carlo)',
    line=dict(width=4, color='red')
))

fig.update_layout(
    title=f"Monte Carlo: Produção Agregada (100 simulações)",
    xaxis_title="Tempo (t)",
    yaxis_title="Bad debt (Valor Monetário)"
)
fig.show()


# ============================================================
# FIGURA 4: Probabilidade de Eventos Extremos (Avalanches)
# ============================================================

if 'results_BD' in locals():
    # "Achatamos" a matriz para um único vetor longo (pool de dados)
    bd_data = results_BD.flatten() 
else:
    bd_data = np.array(econ.history["Bad debt"])

median_bd = np.median(results_BD)
bd_prime = np.abs(bd_data - median_bd)
sigma_bd = np.std(bd_data)

x_values = np.linspace(0, 10, 100) 
prob_y = []

for x in x_values:
    threshold = x * sigma_bd
    count_extremes = np.sum(bd_prime > threshold)
    probability = count_extremes / len(bd_prime)
    prob_y.append(probability)

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=x_values,
    y=prob_y,
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Prob(BD\' > xσ)'
))

fig4.update_layout(
    title="Figura 4: Agregado de má dívida - Probabilidade de Eventos Extremos",
    xaxis_title="x (Múltiplos do Desvio Padrão)",
    yaxis_title="log(Probabilidade)",
    yaxis_type="log"
)

fig4.show()

# %%
print("Downstream degrees:", np.unique(down_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))
# %%
