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
    A_d_safe = np.maximum(A_it, 1e-6)
    return phi * (A_d_safe ** beta)


def uit(size):
    """"
        Equação pag. 8 - preço estocástico do bem final
        uit ~ U(0, 2) - distribuição uniforme entre 0 e 2
        size = número de firmas downstream
        preço estocastico, exogeno e aleatorio do bem final
    """
    return np.random.uniform(0, 2, size)


def Qit(Yit, gamma):
    """"
        Equação pag. 8 - demanda por bens intermediários
        
        Q_it = γ * (ϕ * A_it^β)
        Q_it = gamma * Y_it
    """
    return gamma * Yit


def Nit(Y_it, delta_d):
    """"
        Equação pag. 8 - demanda por trabalho
       
        N_it = δ_d * (ϕ * A_it^β)
        N_it = δ_d * Y_it
    """
    return delta_d * Y_it

def Leontief(gamma, delta_d, Nit, Q_supply):
    """
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
        soma = sum(Yit(A_d[i], params.beta, params.phi) for i in S_j)
        Q_u[j] = params.gamma * soma
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
        Equação pag. 9 - rjt é a taxa de juros no crédito comercial
        r_jt = α * A_t^(-α)
        o nível de rjt depende da condição financeira da firma upstream.
    """
    return alpha * (At ** (-alpha))

def rxt(A_bank, leverage_borrower, sigma, theta):
    """"
        Pág 12 -Taxa de juros bancária geral
        r = sigma * A_z^(-sigma) + theta * (leverage)^theta

        A_zt = patrimônio líquido do banco z no tempo t
        lxt = índice de alavancagem da firma tomadora do empréstimo bancário
        sigma e theta = parâmetros positivos
    """
    A_z_safe = np.maximum(A_bank, 1e-6)
    l_safe = np.maximum(leverage_borrower, 0) 

    base_rate = sigma * max(A_z_safe, 1e-6) ** (-sigma)
    risk_premium = theta * (leverage_borrower ** theta)
    return base_rate + risk_premium

def Bxt(wage_bill_u, A_vector):
    """"
        B_xt = max(w_t * N_xt - A_xt, 0)
        cálculo do gap financeiro (demanda de crédito bancário)
        
        wage_bill = w_t * N_xt
        w_t = constante uniforme entre as firmas
        N_xt = demanda de trabalho da firma x em tempo t
        A_xt = patrimônio líquido da firma x em tempo t
    """
    return np.maximum(wage_bill_u - A_vector, 0)


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

        l_it = lxt(B_i,  A_i)

        r_banks = []
        for z in banks: 
            A_z = self.A_z[z]

            term_bank = p.sigma * (A_z ** (-p.sigma))
            term_risk = p.theta * (l_it ** p.theta)
            r_banks.append(term_bank + term_risk)

        r_iz_list[i] = np.mean(r_banks)
    
    return r_iz_list

def lxt(B_t, A_t):
    """
        Equação pag. 13 - indice de alavancagem
        l_it = B_it / A_it
        Define a razão entre o empréstimo bancário e o patrimônio líquido da firma downstream i.
        Define a razão entre o empréstimo bancário e o patrimônio líquido da firma upstream j.
        indice de alavancagem: relação entre crédito demanda (Bit) e patrimônio líquido (Ait).
    """
    return B_t / A_t if A_t > 1e-6 else 0
# ============================================================
# PREÇOS
# ============================================================

def pjt(At, Alpha):
    """"
        Equação pag. 9 - cálculo do preço do bem intermediário que a firma j cobra de seus consumidores.
        p_jt = 1 + r_jt = 1 + α * A_t^(-α)
        preço do bem intermediário fornecido pela firma upstream j
    """
    if Alpha > 0:
        return 1 + (rjt(At, Alpha))
    return 0

# ============================================================
# LUCROS
# ============================================================

def pi_jt(p_jt_list, Qjt_list,  rjzt_list, Bjt_list, Qit_list, rj_list):
    profits = []
    for p_jt, Qjt, rjzt, Bjt, Qi, r_jt in zip(
        p_jt_list, Qjt_list, rjzt_list, Bjt_list, Qit_list, rj_list
    ):
        profit = p_jt * Qjt - (1 + rjzt_list) * Bjt - (1 + rj_list) * Qi
        profits.append(profit)
    return np.array(profits)


def pi_jt(rjt, Qjt, rjzt):
    profit = (1 + rjt) * Qjt - (1 + rjzt) * Qjt
    return profit

def pi_zt(rizt, bit, rjzt, bjt):
    profit = (1 + rizt) * bit - (1 + rjzt) * bjt
    return profit



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
        """
            Cálculo da produção downstream, demanda por trabalho e bens intermediários
        """
        p = self.params
        A_d_safe = np.maximum(self.A_d, 1e-6)
        Y_d = Yit(A_d_safe, p.beta, p.phi)
        N_d_demand = Nit(Y_d, p.delta_d)
        Q_d_demand = Qit(Y_d, p.gamma)

        u_prices = uit(size=p.N_d)
        revenue_d = u_prices * Y_d

        Q_u_production = np.zeros(p.N_u)

        for i in range(p.N_d):
            u_id = self.supplier[i][0]
            Q_u_production[u_id] += Q_d_demand[i]
        
        N_u_demand = p.delta_u * Q_u_production
        
        return Y_d, Q_d_demand, N_d_demand, revenue_d, Q_u_production, N_u_demand
    
    def step_credit_prices_and_rates(self, N_d_demand, N_u_demand):
        """
            Calcula preços (U) e taxas de juros (bancos) com base na saúde financeira
            e na demanda de crédito gerada pela produção

            Equação pag. 13 - Lxt, Rxt, Btz
        """

        p = self.params

        A_u_safe = np.maximum(self.A_u, 1e-6)

        # demanda de crédito comercial
        r_trade_u = rjt(A_u_safe, p.alpha)
        p_intermediate_vec = pjt(A_u_safe, p.alpha)

        # GAP FINANCEIRO

        #downstream
        wage_bill_d = p.w * N_d_demand 
        B_d = Bxt(wage_bill_d, self.A_d)

        #upstream
        wage_bill_u = p.w * N_u_demand
        B_u = Bxt(wage_bill_u, self.A_u)
        
        #crédito bancário - calculo das taxas
        r_bank_d = np.zeros(p.N_d)
        r_bank_u = np.zeros(p.N_u)

        #taxas para D
        for i in range(p.N_d):
            
            z_idx = self.bank_links_d[i]
            A_z_curr = self.A_z[z_idx[0]]

            #calculo de alavancagem
            #l = B / A
            A_d_curr = max(self.A_d[i], 1e-6)
            leverage_i = lxt(B_d[i], A_d_curr)

            r_bank_d[i] = rxt(A_z_curr, leverage_i, p.sigma, p.theta)
        
        # taxas para U
        for j in range(p.N_u):
            z_idx = self.bank_links_u[j]
            A_z_curr = self.A_z[z_idx[0]]

            #calculo de alavancagem
            #l = B / A
            A_u_curr = max(self.A_u[j], 1e-6)
            leverage_j = lxt(B_u[j], A_u_curr)

            r_bank_u[j] = rxt(A_z_curr, leverage_j, p.sigma, p.theta)
        
        return p_intermediate_vec, B_d, r_bank_d, B_u, r_bank_u, wage_bill_d, wage_bill_u

    def calculate_profits(self, revenue_d, wage_bill_d, wage_bill_u, B_d, r_bank_d, B_u, r_bank_u, Q_d_demand, price_intermediate):
        """"
            Calcula os lucros das firmas downstream, upstream e dos bancos
            Equação pag. 16 - lucro  (pi_jt, pi_it, pi_zt)
        """
        p = self.params

        bank_interest_d = r_bank_d * B_d  #juros
        bank_repayment_d = (1 + r_bank_d) * B_d #principal + juros

        cost_trade_d = np.zeros(p.N_d)
        for i in range(p.N_d):
            u_idx = self.supplier[i][0]
            cost_trade_d[i] = price_intermediate[u_idx] * Q_d_demand[i]

        profits_d = revenue_d - wage_bill_d - bank_interest_d - cost_trade_d

        A_d_pre_default = np.zeros(p.N_d)
        for i in range(p.N_d):
            equity_remanescente = max(self.A_d[i] - wage_bill_d[i], 0)
            A_d_pre_default[i] = revenue_d[i] + equity_remanescente - bank_repayment_d[i] - cost_trade_d[i]


        revenue_u = np.zeros(p.N_u)
        for i in range(p.N_d):
            u_idx = self.supplier[i][0]
            revenue_u[u_idx] += cost_trade_d[i] #U recebe pagamento de D

        A_u_pre_default = np.zeros(p.N_u)
        bank_interest_u = r_bank_u * B_u #juros

        for j in range(p.N_u):
            equity_remanescente = max(self.A_u[j] - wage_bill_u[j], 0) #pagamento de custos
            bank_repayment_u = (1 + r_bank_u[j]) * B_u[j] # ( 1 + rjzt) * B_jt
            A_u_pre_default[j] = revenue_u[j] + equity_remanescente - bank_repayment_u

        profits_z = np.zeros(p.N_z)

        #juros de D ->Z
        for i in range(p.N_d):
            if len(self.bank_links_d[i]) > 0:
                z_idx = self.bank_links_d[i]
                profits_z[z_idx] += bank_interest_d[i]

        #juros de U ->Z
        for j in range(p.N_u):
            if len(self.bank_links_u[j]) > 0:
                z_idx = self.bank_links_u[j]
                profits_z[z_idx] += bank_interest_u[j]
        
        self.A_z += profits_z

        return A_u_pre_default, A_d_pre_default, profits_z, cost_trade_d     


    def update_net_worth_pre_default(self, profits_d, profits_u, profits_z):   
        """"
            Atualiza o patrimônio líquido antes da propagação do calote
            A_{t+1} = A_t + Pi_t
        """
        A_d_new = self.A_d + profits_d
        A_u_new = self.A_u + profits_u
        self.A_z += profits_z

        return A_d_new, A_u_new

    def step_profits_and_dynamics(self, Y_d, revenue_d, Q_d_demand, price_intermediate, B_d, 
                                  r_bank_d, B_u, r_bank_u, Q_u_production, wage_bill_d, wage_bill_u):
        """"
            revenue_u = cust_trade_d (receita recebida por vendas da downstream para upstream)
            pi_jt = cost_trade - bank_cost
        """

        p = self.params

        bank_interest_d = r_bank_d * B_d  #juros
        bank_repayment_d = (1 + r_bank_d) * B_d #principal + juros

        #pagamento por bens intermediarios Ddownstream -> Upstream
        cost_trade_d = np.zeros(p.N_d)
        for i in range(p.N_d):
            u_idx = self.supplier[i][0]
            cost_trade_d[i] = price_intermediate[u_idx] * Q_d_demand[i]

        profits_d = revenue_d - wage_bill_d - (r_bank_d * B_d) - cost_trade_d

        # receita e atualizacao Downstream
        A_d_pre_default = np.zeros(p.N_d)
        for i in range(p.N_d):
            equity_remanescente = max(self.A_d[i] - wage_bill_d[i], 0)
            A_d_pre_default[i] = revenue_d[i] + equity_remanescente - bank_repayment_d[i] - cost_trade_d[i]

        revenue_u = np.zeros(p.N_u)
        for i in range(p.N_u):
            #recebimentos de D
            u_idx = self.supplier[i][0]
            revenue_u[u_idx] = cost_trade_d[i] #U recebe pagamento de D

        A_u_pre_default = np.zeros(p.N_u)
        bank_interest_u = r_bank_u * B_u #juros

        # receita e atualizacao Upstream
        for j in range(p.N_u):
            equity_remanescente = max(self.A_u[j] - wage_bill_u[j], 0) #pagamento de custos
            bank_repayment_u = (1 + r_bank_u[j]) * B_u[j] # ( 1 + rjzt) * B_jt
            A_u_pre_default[j] = revenue_u[j] + equity_remanescente - bank_repayment_u

        # lucros banco - juros recebidos de D e U
        profits_z = np.zeros(p.N_z)
        for i in range(p.N_d):
            if len(self.bank_links_d[i]) > 0:
                z_idx = self.bank_links_d[i][0]
                profits_z[z_idx] += r_bank_d[i] * B_d[i]

        for j in range(p.N_u):
            if len(self.bank_links_u[j]) > 0:
                z_idx = self.bank_links_u[j][0]
                profits_z[z_idx] += r_bank_u[j] * B_u[j]
        
        self.A_z += profits_z

        return A_d_pre_default, A_u_pre_default, cost_trade_d


    def propagate_bad_debt(self, A_d_new, A_u_new, B_d, costs_trade_d, B_u):
        p = self.params
        
        total_bad_debt = 0.0
        
        # --- Check Default D --- Falencia das firmas downstream
        defaults_d = np.where(A_d_new < 0)[0]
        
        # Vetores de perda para U e Z
        loss_to_u = np.zeros(p.N_u)
        loss_to_z = np.zeros(p.N_z)
        
        for i in defaults_d:
            bad_debt_val = abs(A_d_new[i])
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

    def update_supplier_links(self):
        p = self.params

        current_price_u = pjt(self.A_u, p.alpha)

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
                wage_bill_d=wb_d,
                wage_bill_u=wb_u,
                B_d=B_d,
                r_bank_d=r_b_d,
                B_u=B_u,
                r_bank_u=r_b_u,
                Q_u_production=Q_u_production,
                Q_d_demand=Q_d_demand,
                price_intermediate=p_inter
            )


            bd = self.propagate_bad_debt(A_d_new, A_u_new, B_d, cost_trade_d, B_u)
            
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
time_steps = np.arange(len(agg_revenue_t)) + 1
log_rev = np.log10(np.maximum(agg_revenue_t, 1e-12))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_steps,
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
        fig2.update_layout(
            title="(B) Distribuição de Tamanho das Firmas", 
            xaxis_title="log(Patrimônio Líquido)",
            yaxis_title="log(Rank)")
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
fig.update_layout(title="(D) - Grau de Distribuição - Banks vs (D + U clients)")
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
        line=dict(width=1, color='rgba(0,0,255,0.2)'),
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
    yaxis_title="Log (Y)"
)
fig.show()


# ============================================================
# FIGURA 4: Probabilidade de Eventos Extremos (Avalanches)
# ============================================================

bd_data = np.array(econ.history["Bad debt"])

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

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=x_values,
    y=prob_y,
    mode='markers',
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
