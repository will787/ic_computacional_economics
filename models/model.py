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

    def production_downstream(self):
        p = self.params
        #A_safe = np.maximum(self.A_d, 1e-6)  # evita capital negativo
        Y_star = p.phi * self.A_d ** p.beta
        N_demand = p.delta_d * Y_star
        Q_demand = p.gamma * Y_star
        u = uit(size=p.N_d)

        revenue = u * Y_star
        return Y_star, Q_demand, N_demand, u, revenue
    
    def credit_demand_d(self, Y, N):
        p = self.params
        w_vector = p.w * np.array(N)
        return Bxt(w_vector, self.A_d)
    
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
        new_supplier = []

        for i in range(p.N_d):
            current = self.supplier[i][0]
            candidates = np.arange(p.N_u)
            prices = pjt(self.A_u, p.alpha)

            chosen = Economy.choose_with_noise(
                candidates=candidates,
                scores=prices,
                sample_size=p.M,
                eps=p.e
            )
            self.supplier[i] = chosen

    def update_bank_links_d(self):
        p = self.params
        new_bank_links = []

        for i in range(p.N_d):
            candidates = np.arange(p.N_z)
            prices = p.sigma * (self.A_z ** (-p.sigma))

            chosen_banks = Economy.choose_with_noise(
                candidates=candidates,
                scores=prices,
                sample_size=p.Z,
                eps=p.e
            )
            self.bank_links_d[i] = chosen_banks

    def update_bank_links_u(self):
        p = self.params
        new_bank_links = []

        for j in range(p.N_u):
            candidates = np.arange(p.N_z)
            prices = p.sigma * (self.A_z ** (-p.sigma))

            chosen_banks = Economy.choose_with_noise(
                candidates=candidates,
                scores=prices,
                sample_size=p.Z,
                eps=p.e
            )

            self.bank_links_u[j] = chosen_banks


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
        for t in range(T):
            if t % 100 == 0: print(f"Time period: {t}")

            if t > 0:
                    self.supplier = supplier
                    self.bank_links_d = banks_links_d
                    self.bank_links_u = banks_links_u


            #producao downstream
            Y_star, Q_d_demand, N, u_vec, revenue = self.production_downstream()
            Y_eff = Y_star.copy()

            Q_demand_u = np.zeros(self.params.N_u)
            for i in range(self.params.N_d):
                u_id = self.supplier[i][0]
                Q_demand_u[u_id] += Q_d_demand[i]

            #producao potencial de upstream
            Q_potencial_u = q_upstream(self.A_u, self.params.phi, self.params.beta)
            Q_supply_u = np.minimum(Q_potencial_u, Q_demand_u)
            
            #contagem quantas clientes firmas upstream tem
            u_clients_counts = np.zeros(self.params.N_u)
            for i in range(self.params.N_d):
                if len(self.supplier[i]) > 0:
                    u_id = self.supplier[i][0]
                    u_clients_counts[u_id] += 1

            # racionamento
            for i in range(self.params.N_d):
                if len(self.supplier[i]) > 0:
                    u_id = self.supplier[i][0]

                    n_clients = max(1, u_clients_counts[u_id])
                    Q_supply_i = Q_supply_u[u_id] / n_clients

                    Y_eff[i] = min(
                        Y_star[i],
                        N[i] / self.params.delta_d,
                        Q_supply_i / self.params.gamma,
                    )

            B_up, N_up = self.credit_demand_u(Q_supply_u)
            r_bank_u_vector = rjz(self, B_up)

            r_trade_vector_u = np.zeros(self.params.N_u)
            for j in range(self.params.N_u):
                r_trade_vector_u[j] = rjt(self.A_u[j], self.params.alpha)

            p_intermediate_u = 1 + r_trade_vector_u

            r_trade_vector_d = np.zeros(self.params.N_d)
            for i in range(self.params.N_d):
                u_id = self.supplier[i][0]
                r_trade_vector_d[i] = r_trade_vector_u[u_id]
            
            p_intermediate_d = 1 + r_trade_vector_d

            B_down = self.credit_demand_d(Y_eff, N)
            r_bank_d_vector = riz(self, B_down)

            Q_d_effective = np.zeros(self.params.N_d)
            for i in range(self.params.N_d):
                u_id = self.supplier[i][0]
                n_clients = max(1, len(supplier[u_id]))
                Q_d_effective[i] = Q_supply_u[u_id] / n_clients

            profits_d = self.profits_downstream(Y_eff, B_down, u_vec, Q_d_effective, p_intermediate_d, r_bank_d_vector)
            profits_u = self.profits_upstream(Q_supply_u, B_up, p_intermediate_u, r_bank_u_vector)

            #atualizacao do patrimonio liquido
            bad_debt_t = self.update_financial_positions_and_propagate(
                profits_d,
                profits_u,
                B_down,
                Q_d_effective,
                r_trade_vector_d,
                r_bank_d_vector,
                B_up,
                r_bank_u_vector
            )

            #como o mercado se organiza
            down_deg, up_deg = degree_distribution_new(self.supplier, self.params.N_d, self.params.N_u)
            
            #historico
            #A_total = self.A_d.sum() + self.A_u.sum() + self.A_z.sum()

            self.history["Y"].append(Y_eff.copy())
            self.history["Revenue"].append(revenue.copy())
            self.history["A_d"].append(self.A_d.copy())
            self.history["A_u"].append(self.A_u.copy())
            self.history["profits_d"].append(profits_d)
            self.history["deg_down"].append(down_deg)
            self.history["deg_up"].append(up_deg)
            self.history["Bad debt"].append(bad_debt_t)

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
    econ = Economy(p, supplier = None, bank_links_d = None, bank_links_u = None)

    #1. Downstream -> upstream
    supplier = [
        list(choose_preferred_upstream(econ.A_u, p.M))
        for _ in range(p.N_d)
    ]

    # Banks -> Downstream
    banks_links_d = [
        list(choose_preferred_banks(econ.A_z, p.Z))
        for _ in range(p.N_d)
    ]
    
    # Banks -> Downstream
    banks_links_u = [
        choose_preferred_banks(econ.A_z, p.Z)
        for _ in range(p.N_u)
    ]
    econ.supplier = supplier
    econ.bank_links_d = banks_links_d
    econ.bank_links_u = banks_links_u
    econ.simulate(T=1000)
    

# 1 model

rev_by_period = np.array(econ.history["Y"])
agg_revenue_t = rev_by_period.sum(axis=1) #soma por periodo das firmas
log_rev = np.log10(np.maximum(agg_revenue_t, 1e-12))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(len(rev_by_period[rev_by_period > 0])),
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
A_sorted = np.sort(A_final)[::-1]
rank = np.arange(1, len(A_final) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(A_sorted),
    y=np.log10(rank),
    mode="markers"
))
fig.update_xaxes(title="log(A)")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(B) - Degree Distribution - Downstream size (in terms PL)")
fig.show()


# 3 model

down_deg = econ.history["deg_down"][-1]
up_deg = econ.history["deg_up"][-1]

d_sorted = np.sort(down_deg)[::-1]
rank = np.arange(1, len(d_sorted) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.log10(d_sorted[d_sorted > 0]),
    y=np.log10(rank[:len(d_sorted[d_sorted > 0])]),
    mode="markers"
))
fig.update_xaxes(title="log(number of links)")
fig.update_yaxes(title="log(rank)")
fig.update_layout(title="(C) - Degree Distribution of Network - Downstream vs Upstream")
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
time_steps = np.arange(len(bad_debt_series))

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


# %%
print("Downstream degrees:", np.unique(down_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))
print("Upstream degrees:", np.unique(up_deg, return_counts=True))
# %%
