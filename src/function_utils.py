#module contenant les différentes fonctions utiles. 
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
from scipy.stats import mvn
from scipy.stats import norm
from scipy.linalg import cholesky

#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
def payoff_function(x):
    #fonction phi
    return x**2

#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------Calcul des strats Si------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------


def calculate_strata_boundaries(i_tuple, I, d):
    """
    préliminaire : calcul des strats. 
    Calcul les strats pour une loi normale multivariée de dimension d. Dans ce code on suppose que la covariance est l'identitée. 

    :param i_tuple: m-uplet (i1, ..., im) representing indices of the strata
    :param I: Positive integer used in the calculation of strata boundaries
    :param d: Dimension of the multivariate normal distribution
    :return: Strata boundaries Si as a product of tuples
    """
    Si = []
    
    # Compute the CDF for each component of the tuple i_tuple
    for ik in i_tuple:
        # Compute the CDF for the lower and upper bounds
        lower_bound = norm.ppf((ik - 1) / I)
        upper_bound = norm.ppf(ik / I)
        
        # As we're considering multivariate normal distribution, we assume independence
        # and hence the product of intervals across dimensions forms the stratum.
        Si.append(((lower_bound, upper_bound),) * d)
    
    # Si should be a tuple of tuples representing the Cartesian product of intervals across dimensions
    return tuple(Si)


#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.A.i - mise à jour de nu -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

def update_nu_hat(pi, Y, payoff=payoff_function):
    # étape 2.a.i. 
    #pour chaque i on doit mettre à jour cette fonction nu. On a donc 
    #   i). On donne en input pi et Y (liste de réalisation, de taille Mi), choisi en avance, on renvoie ensuite pour phi et phi2 la valeur nu

    Mi = len(Y) #Y est une liste de v.a. samplés suivant la distribution P(Y|Y in S)
    sum_payoff_1 = np.sum(payoff(Y))
    sum_payoff_2 = np.sum(payoff(Y)**2) #on prend aussi le carré du payoff 
    nu_hat_phi= (pi/ Mi) * sum_payoff_1 #le pi est donnée, le Mi on l'a déduit 
    nu_hat_phi2= (pi/ Mi) * sum_payoff_2
    return nu_hat_phi, nu_hat_phi2 #on renvoie les deux mu_hat_1, mu_hat_2 -> l'idée


#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.A.ii - tirage suivant la distribution/calcul du gradient de nu -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------



def stratified_sampling_linear_projection(mu, Sigma, v, K,s=None):
    """Cette fonction sert à tirer Y conditionnellement à mu.T*Y = s. Utile pour le problème. 

    Generates K samples from N(0, Sigma) stratified along the direction determined by v.
    
    Args:
    - mu (array): The mean vector of the distribution.
    - Sigma (2D array): The covariance matrix of the distribution.
    - v (array): The vector along which stratification is done.
    - K (int): The number of stratified samples to generate.
    - x (optionnal) :  si on donne une valeur pour x alors c'est qu'on conditionne à une valeur particulière de x. Sinon on sample sur tout la strat Si. Taille K
    
    Returns:
    - samples (2D array): The generated stratified samples. (K, de N(0,sigma), stratifié sur la direction donnée par v)
    """
    d = len(mu)  # Dimension of the normal distribution
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
    if s is not None:

        xi_samples = Sigma @ v * s[:, None] + (A_minus_v_Sigma_v_T_A @ Z.T).T  # Conditional samples 
        #xi_samples = Sigma @ v @ x + (A_minus_v_Sigma_v_T_A @ Z.T).T

    else:
        xi_samples = Sigma @ v * X[:, None] + (A_minus_v_Sigma_v_T_A @ Z.T).T 
        #xi_samples = Sigma @ v @ X + (A_minus_v_Sigma_v_T_A @ Z.T).T 
        
    
    return xi_samples + mu             # Add the mean to each sample

#comment calculer le gradient de nu ? ?? 



#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.A.iii - calcul du gradient_V à partir des calculs précédents ----------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------


def calculer_gradient_V(grad_mu,grad_mu_phi2,mu_phi,mu_phi2, p_i, sigma_i):
    """
    Important : avoir les listes qui contiennent tout. On doit stocker ts ls objets ds ds listes 
    Calcule le gradient de V par rapport à mu en utilisant les gradients donnés et les valeurs pour f, phi, pi et sigma_i.
    
    """
    I = len(p_i)  # Nombre de termes dans la somme
    gradient_V = np.zeros_like(grad_mu)  # Initialisation du gradient de V(mu) à zéro

    for i in range(I):
        if p_i[i] * sigma_i[i] != 0:  # Calculer seulement si le produit pi * sigma_i est non nul
            term_1 = grad_mu[i] * mu_phi2[i]
            term_2 = grad_mu_phi2[i]*p_i[i]
            common_term = 2*(mu_phi[i]*grad_mu[i])

            gradient_V += (term_1 + term_2 - common_term) / (2 * p_i[i] * sigma_i[i])

    return gradient_V


#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.b - mise à jour de la direction de stratifaction -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------




def update_direction(mu_t, gradient_V, gamma_t, m):
   #2.b. 
   #Mise à jour de la direction de stratifaction. (ittération). On donne aussi m le nombre de composantes singulières que l'on garde.
   
    mu_hat = mu_t - gamma_t * gradient_V

    # Perform singular value decomposition
    U, s, Vt = np.linalg.svd(mu_hat)

    # Define mu_(t+1) as the orthogonal matrix found by keeping the m left singular vectors
    mu_t_plus_1 = U[:, :m]

    return mu_t_plus_1

#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.c - mise à jour de standard deviation, allocation et nouveaux Mi -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------


def calculate_sigma_hat(nu_hat_phi2, nu_hat_phi, pi, mu):
   #2.c.i. 
   #on calcul juste pour tous les i l'estimateur de la standard deviation sigma_hat sur chaque strat. 
    term1 = nu_hat_phi2 /pi 
    term2 = (nu_hat_phi / pi ) ** 2
    sigma_hat = (term1 - term2) ** 0.5
    return sigma_hat



def calculate_q_t_plus_1(p_i, sigma_hat_i,pis, sigma_hats):
    #2.c.ii.
    #On calcul le vecteur d'allocation pour chaque i ici. On donne le pi, sigma_hat_i, et surtout la liste de tous les sigma hats et de tous les pis
    denominator = sum([p_j * sigma_hat_j for p_j,  sigma_hat_j in zip(pis, sigma_hats)])
    q_t_plus_1 = (p_i *  sigma_hat_i) / denominator 
    return q_t_plus_1



def calculate_M_i(q_t_plus_1_list, i, M):
    #2.c;ii.
    #on calcul ici les nouveaux Mi, pour chaque i.. On se base sur la valeur de qi précédente que l'on a pu calculer. 
    #les Mi sont fonctions des q_i
    sum_q_j_up_to_i = sum(q_t_plus_1_list[:i])
    sum_q_j_less_than_i = sum(q_t_plus_1_list[:i-1])
    M_i = int(M * sum_q_j_up_to_i) - int(M * sum_q_j_less_than_i)
    return M_i


#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.d - mise à jour des probabilités -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

# à vérifier cette fonction. Mais petit draf. Pour chaque strat on calcul cette proba, et il nous faut un vecteur mu. 

def calculate_probability(mu,mean, sigma, Si): 
    #2.d. 
    #On doit mettre à jour les probas, du coup on doit calculer la proba d'être dans Si. On appelle sigma le covariance de Y. 
    #mean est la moyenne de y

    #param Si: A list of tuples representing the bounds in each dimension of Si. Dimension d.
    lower_bounds, upper_bounds = zip(*Si)

    # The mean vector after the transformation mu^T Y is the product of mu and the means of Y, which is zero.
    # Thus, the transformed mean is zero.
    transformed_mean = np.zeros_like(mean)

    # The covariance matrix after the transformation mu^T Y is mu * sigma * mu^T.
    transformed_cov = np.outer(mu, mu) * sigma 

    # Calculate the probability using the mvn (multivariate normal) cumulative distribution function.
    prob, _ = mvn.mvnun(lower_bounds, upper_bounds, transformed_mean, transformed_cov)
    return prob



#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.e - Calculer une moyenne stratifiée ds quantités d'intérêt -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_s_squared(pis, sigma_hats, M):
    #2.e.
    #on calcul les estimateurs stratifiés. On donne des listes pis, sigma_hats en input. 
    s_squared = (1 / M) * sum((pi * sigma_hat) ** 2 for pi, sigma_hat in zip(pis, sigma_hats))
    return s_squared



def calculate_epsilon_t_plus_1(s_squared_values, nu_hat_phi_values):
    #2.e
    #current fit de l'estimateur stratifié, et dernière étape de l'algo. 
  
    sum_inverse_s_squared = sum(1/s_squared for s_squared in s_squared_values)
    sum_product_inverse_s_nu_hat = sum(
        1/s_squared_values[tau-1] * sum(nu_hat_phi_values[tau-1])
        for tau in range(1, len(s_squared_values) + 1)
    )
    epsilon_t_plus_1 = sum_inverse_s_squared**(-1) - sum_product_inverse_s_nu_hat
    return epsilon_t_plus_1