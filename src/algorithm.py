import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky


def stratified_sampling_linear_projection(mean, Sigma, v, K, x=None):
    """
    Generates K samples from N(0, Sigma) stratified along the direction determined by v.
    
    Args:
    - mean (array): The mean vector of the distribution.
    - Sigma (2D array): The covariance matrix of the distribution.
    - v (array): The vector along which stratification is done.
    - K (int): The number of stratified samples to generate.
    - x (optionnal) :  si on donne une valeur pour x alors c'est qu'on conditionne à une valeur particulière de x. Sinon on sample sur tout la strat Si. Taille K
    
    Returns:
    - samples (2D array): The generated stratified samples. (K, de N(0,sigma), stratifié sur la direction donnée par v)
    """
    d = len(mean)  # Dimension of the normal distribution
    # Normalize v so that v^T Sigma v = 1
    v = v / np.sqrt(v.T @ Sigma @ v)
    
    # Generate K stratified samples for the standard normal distribution along v
    U = np.random.uniform(0, 1, K)     # Uniformly distributed samples in (0,1)
    V = (np.arange(K) + U) / K         # Stratified samples in (0,1)
    
    X = norm.ppf(V)                    # Inverse CDF (quantile function) to get stratified samples for N(0, 1)
    
    # Compute the matrix A for the conditional distribution of xi given x
    A = cholesky(Sigma, lower=True)    # Cholesky factorization
    A_minus_v_Sigma_v_T_A = A - np.outer(Sigma @ v, v.T @ A)
    
    # Generate K samples from the conditional distribution of xi given X
    Z = np.random.randn(K, d)  # Z ~ N(0, I) in d dimensions 
    if x is not None:

        xi_samples = Sigma @ v * x[:, None] + (A_minus_v_Sigma_v_T_A @ Z.T).T  # Conditional samples 
        #xi_samples = Sigma @ v @ x + (A_minus_v_Sigma_v_T_A @ Z.T).T

    else:
        xi_samples = Sigma @ v * X[:, None] + (A_minus_v_Sigma_v_T_A @ Z.T).T 
        #xi_samples = Sigma @ v @ X + (A_minus_v_Sigma_v_T_A @ Z.T).T 
        
    
    return xi_samples + mean            # Add the mean to each sample


def tirer_K_nb(borne_inf, borne_sup, K):
    #vérifier comment tirer les nonbres uniformes sur -inf, a et sur a, +inf. Plusieurs manieres de faire je suis pas sur 
    if borne_inf == -np.inf:
        # Pour l'intervalle (-∞, b)

        u = np.random.uniform(0, 1, K)
        x_neg_inf = borne_sup - np.log(1 / u)
        return x_neg_inf

    elif borne_sup == np.inf:
        u = np.random.uniform(0, 1, K)
        x_pos_inf = borne_inf + np.log(1 / (1 - u))
        return x_pos_inf

    else:
        out = np.random.uniform(borne_inf , borne_sup, K)
        return out 
    

def quantiles_normaux(I):
    """Args : I 
    return : les quantiles normaux, affiches ci-dessus;"""
   
    quantiles = [norm.ppf(k / I) for k in range(1, I)]
    
    return quantiles


def tirer_y_strat_nu(mu, i, mean, Sigma, K, S):
    """

    Args : 
        - mu : l'orientation de la strat 
        - i : indice appartenant à 0....I-1 !! et pas 1,....I ATTENTION 
        - mean : moyenne de la v.a. Y 
        - sigma : variance de la v.a. Y 
        - S : nos strats 
    
    Outputs : tire K variables suivant l'indice i pour la strat donnée """

    return  stratified_sampling_linear_projection(mean, Sigma, mu, K, tirer_K_nb(S[i][0],S[i][1], K))


def compute_m(q, M):

    I = len(q)
    m = np.zeros(I)
    cumulative_sum = np.cumsum(q)

    m[0] = int(M*cumulative_sum[0])

    for i in range(1, I):

        m[i] = max(int(M*cumulative_sum[i]) - int(M*cumulative_sum[i-1]), 1)
    
    return np.array(m, dtype=int)


def strats_Si(I):
    # Calcul de Si pour chaque ik de 1 à I
    results = [(norm.ppf((ik - 1) / I), norm.ppf(ik / I)) for ik in range(1, I + 1)]
    return results



class Algorithm():

    def __init__(self,
                 M,
                 I,
                 d,
                 phi,
                 mean,
                 cov,
                 K: int=1000):

        self.M = M
        self.I = I
        self.d = d
        self.phi = phi
        self.phi2 = lambda x: phi(x)**2

        self.mean = mean
        self.cov = cov
        self.strats = strats_Si(I)


        self.mu = np.ones(d)/np.sqrt(d)
        self.p = np.ones(I)/I

        sigma = np.zeros(I)
        for i in range(I):
            samples = tirer_y_strat_nu(self.mu, i, mean, cov, K, self.strats)
            sigma[i] = np.sqrt(np.mean(np.apply_along_axis(self.phi2, 1, samples))
                                - np.mean(np.apply_along_axis(phi, 1, samples))**2)
        self.sigma = sigma
        
        self.q = (self.p * self.sigma)/np.sum(self.p * self.sigma)
        self.m = compute_m(self.q, M)

    
    def compute_nu_and_grad_nu_estimates(self):

        nu_f_phi = np.zeros((self.I))
        nu_f_phi2 = np.zeros((self.I))

        grad_nu_f = np.zeros((self.I, self.d))
        grad_nu_f_phi = np.zeros((self.I, self.d))
        grad_nu_f_phi2 = np.zeros((self.I, self.d))

        for i in range(self.I):

            samples = tirer_y_strat_nu(self.mu, i, self.mean, self.cov, self.m[i], self.strats)
            nu_f_phi[i] = self.p[i]*np.mean(np.apply_along_axis(self.phi, 1, samples))
            nu_f_phi2[i] = self.p[i]*np.mean(np.apply_along_axis(self.phi2, 1, samples))

            grad_nu_f[i] = - np.mean(samples, axis=0)/np.abs(self.mu)
            grad_nu_f_phi[i] = - np.mean(samples*np.apply_along_axis(self.phi, 1, samples)[:, np.newaxis]/np.abs(self.mu), axis=0)
            grad_nu_f_phi2[i] = - np.mean(samples*np.apply_along_axis(self.phi2, 1, samples)[:, np.newaxis]/np.abs(self.mu), axis=0)
        
        return nu_f_phi, nu_f_phi2, grad_nu_f, grad_nu_f_phi, grad_nu_f_phi2


    def compute_grad_V_estimate(self, nu_f_phi, nu_f_phi2, grad_nu_f, 
                                grad_nu_f_phi, grad_nu_f_phi2):
        
        grad_V = 0

        for i in range(self.I):

            if self.p[i]*self.sigma[i] != 0:

                term_to_add = (grad_nu_f[i] * nu_f_phi2[i]
                                + self.p[i] * grad_nu_f_phi2[i]
                                - 2 * nu_f_phi[i] * grad_nu_f_phi[i]) / (2 * self.p[i] * self.sigma[i])
                
                grad_V += term_to_add

        return grad_V
    

    def update_mu_sigma_q_p(self, gamma):

        (nu_f_phi, nu_f_phi2, grad_nu_f, 
         grad_nu_f_phi, grad_nu_f_phi2) = self.compute_nu_and_grad_nu_estimates()
        
        grad_V = self.compute_grad_V_estimate(nu_f_phi, nu_f_phi2, grad_nu_f, 
                                               grad_nu_f_phi, grad_nu_f_phi2)


        # Update mu
        mu_tilde = self.mu - gamma * grad_V
        U, _, _ = np.linalg.svd(mu_tilde[:, np.newaxis])
        self.mu = U[:, :1].T[0]

        # Update sigma 
        self.sigma = np.sqrt(nu_f_phi2/self.p
                        - ((nu_f_phi)/self.p)**2)
        
        # Update q
        self.q = (self.p * self.sigma)/np.sum(self.p * self.sigma)
        
        # Update p
        self.p = np.ones(self.I)/self.I

        # Update m
        self.m = compute_m(self.q, self.M)

    
    def compute_V(self):

        return np.sum(self.p * self.sigma)

    def compute_algorithm_iteration(self,
                                    N: int=1000,
                                    callback: bool=True):
        
        if callback:

            V = np.zeros(N+1)
            V[0] = self.compute_V()
            mus = [self.mu]

        for n in range(1, N+1):
            
            gamma = 1 / n 

            self.update_mu_sigma_q_p(gamma)
            
            if callback:
                
                V[n] = self.compute_V()
                mus.append(self.mu)
        
        return V, mus