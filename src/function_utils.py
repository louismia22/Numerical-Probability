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

    Pour call la function : calculate_strata_boundaries((1,2,3,4,8),10,3) par exemple -> renvoie (((-inf, -1.2815515655446004),
  (-inf, -1.2815515655446004),
  (-inf, -1.2815515655446004)),...
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

    Mi = len(Y) # Y est une liste de v.a. samplés suivant la distribution P(Y|Y in S)
    sum_payoff_1 = np.sum(payoff(Y))
    sum_payoff_2 = np.sum(payoff(Y)**2) #on prend aussi le carré du payoff 
    nu_hat_phi= (pi/ Mi) * sum_payoff_1 #le pi est donnée, le Mi on l'a déduit 
    nu_hat_phi2= (pi/ Mi) * sum_payoff_2
    return nu_hat_phi, nu_hat_phi2 #on renvoie les deux mu_hat_1, mu_hat_2 -> l'idée


#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.A.ii - tirage suivant la distribution/calcul du gradient de nu -----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------



def stratified_sampling_linear_projection(mu, Sigma, v, K, s=None):
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

# comment calculer le gradient de nu ? ?? 




#--------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------ Étape 2.A.iii - calcul du gradient_V à partir des calculs précédents ----------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------


def calculer_gradient_V(grad_mu, grad_mu_phi2, mu_phi, mu_phi2, p_i, sigma_i):
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
    term2 = (nu_hat_phi/pi)**2
    sigma_hat = np.sqrt((term1 - term2))
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


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------
#------------------------Algorithm 1 INITIALISATION FUNC  ----------------------------------------
#------------------------------------------------------------------------------------

#le code suivant fonctionne. Il faut checker qu'il se comporte bien comme on veut.
# pour l'output du code : containing the number of draws and probabilities for each stratum... 


def calculate_probability(mu, sigma, Si):
    """
    Calculate the probability of being within the bounds specified by Si for the transformed variable mu^T Y.
    
    :param mu: np.array, the direction vector for stratification
    :param sigma: np.array, the covariance matrix of Y
    :param Si: tuple of tuples, each inner tuple contains the lower and upper bounds for a single dimension
    :return: float, probability of being within Si
    """
    lower_bounds, upper_bounds = zip(*Si)

    # Compute the variance of the projection
    transformed_var = np.dot(mu.T, np.dot(sigma, mu))

    # Mean of the projection is zero as mean of Y is assumed to be zero
    transformed_mean = 0

    # Probability calculation adjusted for a univariate normal
    lower_prob = norm.cdf(lower_bounds, loc=transformed_mean, scale=np.sqrt(transformed_var))
    upper_prob = norm.cdf(upper_bounds, loc=transformed_mean, scale=np.sqrt(transformed_var))
    return upper_prob - lower_prob

def calculate_strata_boundaries(index, I, d):
    """
    Calculate boundaries for each stratum in a univariate normal distribution.
    Here d should be 1 because we are projecting onto a line.
    
    :param index: int, index of the stratum
    :param I: int, total number of divisions per dimension
    :param d: int, number of dimensions
    :return: tuple of tuples, where each tuple is a pair (lower_bound, upper_bound)
    """
    lower_bound = norm.ppf((index - 1) / I)
    upper_bound = norm.ppf(index / I)
    return tuple((lower_bound, upper_bound) for _ in range(d))

# Assuming you adjust your initialization call accordingly


# Initialization procedure
def initialization(mu_0, M, sigma):
    """
    Initialize the stratification procedure.

    :param mu_0: np.array, initial stratification directions
    :param M: int, total number of draws
    :param sigma: np.array, covariance matrix of Y
    :return: dict, containing the number of draws and probabilities for each stratum
    """
    m = len(mu_0)  # Number of strata
    M_0 = np.full(m, M // m)  # Evenly distribute initial draws
    M_0[:M % m] += 1  # Distribute any remaining draws equally
    
    # Calculate strata boundaries and probabilities
    strata_bounds = [calculate_strata_boundaries(i, 10, len(mu_0)) for i in range(1, m + 1)]
    print(strata_bounds)
    probabilities = [calculate_probability(mu_0, sigma, Si) for Si in strata_bounds]

    return {"M_0": M_0, "probabilities": probabilities}

# Example usage
#mu_0 = np.array([0.5, 0.4,0.1])
#sigma = np.eye(3)  # Identity matrix for covariance, assuming independence and unit variance
#initialization_result = initialization(mu_0, 100, sigma)
#print(initialization_result) 

# returns ok 




#--------------------------------------------------------------------------------------------------------------
#------------------------Algorithm 2 Computation of Gradient Estimates ----------------------------------------
#--------------------------------------------------------------------------------------------------------------

#partie la plus importante 





#--------------------------------------------------------------------------------------------------------------
#------------------------Algorithm 3 Update of Allocation Policy and Probabilities ----------------------------------------
#--------------------------------------------------------------------------------------------------------------


#3.c.i
def calculate_sigma_hat(nu_hat_phi2, nu_hat_phi, pi, mu):
    term1 = nu_hat_phi2 / pi
    term2 = (nu_hat_phi / pi) ** 2
    sigma_hat_squared = term1 - term2  # This should always be non-negative.
    if sigma_hat_squared < 0:  # Check for negative due to floating point precision issues.
        sigma_hat_squared = max(sigma_hat_squared, 0)
    return sigma_hat_squared ** 0.5

#3.c.ii
def calculate_q_t_plus_1(p_i, sigma_hat_i, pis, sigma_hats):
    denominator = sum([p_j * sigma_hat_j for p_j, sigma_hat_j in zip(pis, sigma_hats)])
    if denominator == 0:  # Check for zero denominator.
        return 0
    return (p_i * sigma_hat_i) / denominator

#3.c.ii calcul juste à partir de la liste q_t_plus_1
def calculate_M_i(q_t_plus_1_list, i, M):
    sum_q_j_up_to_i = sum(q_t_plus_1_list[:i])
    sum_q_j_less_than_i = sum(q_t_plus_1_list[:i-1])
    if not isinstance(sum_q_j_up_to_i, complex) and not isinstance(sum_q_j_less_than_i, complex):
        return int(M * sum_q_j_up_to_i) - int(M * sum_q_j_less_than_i)
    return 0  # Return 0 or some error handling if complex numbers are detected.


def update_allocation_and_probabilities(nu_hat_phi2, nu_hat_phi, pis, mu, M,I,d):
    """
    Update the allocation policy and probabilities.
    
    :param nu_hat_phi2: List of estimated V_i^(t+1)(phi^2) for each stratum
    :param nu_hat_phi: List of estimated V_i^(t+1)(phi) for each stratum
    :param pis: List of probabilities p_i(mu^(t)) for each stratum
    :param mu: Current mu vector (not directly used in this function)
    :param M: Total number of draws
    :return: Tuple (updated q list, updated M_i list, updated probabilities list)
    """
    m = len(pis)
    sigma_hats = [calculate_sigma_hat(nu_hat_phi2[i], nu_hat_phi[i], pis[i], mu) for i in range(m)]
    q_t_plus_1_list = [calculate_q_t_plus_1(pis[i], sigma_hats[i], pis, sigma_hats) for i in range(m)]
    
    M_i_list = [calculate_M_i(q_t_plus_1_list, i+1, M) for i in range(m)]
    
    # Assuming update_probabilities function is defined elsewhere to update p_i(mu^(t+1))
    updated_probabilities = update_probabilities(mu, sigma_hats, I,d)  # You need to implement this function
   
    return q_t_plus_1_list, M_i_list, updated_probabilities


#on calcul juste les probas (en theorie)
def update_probabilities(mu, sigma, I, d):
    """
    Update probabilities based on new mu values and potentially updated sigma or other model parameters.
    
    :param mu: np.array, new direction vector for stratification after updates
    :param sigma: np.array, possibly updated covariance matrix of Y
    :param I: int, total number of divisions per dimension used in stratification
    :param d: int, dimension of the problem
    :return: list, updated probabilities for each stratum
    """
    m = len(mu)  # Assuming the number of strata is determined by the length of mu
    strata_bounds = [calculate_strata_boundaries(i, I, d) for i in range(1, m + 1)]
    updated_probabilities = [calculate_probability(mu, sigma, Si) for Si in strata_bounds]

    return updated_probabilities

# Example use:



# Example parameters
#nu_hat_phi2 = [100, 200, 300, 400]  # Placeholder for V_i^(t+1)(phi^2)
#nu_hat_phi = [10, 20, 30, 40]       # Placeholder for V_i^(t+1)(phi)
#pis = [0.1, 0.2, 0.3, 0.4]          # Placeholder for p_i(mu^(t))
#mu = [0.5, 0.5, 0.5, 0.5]           # Placeholder for mu^(t)
#M = 1000                            # Total number of draws

#q_t_plus_1_list, M_i_list, updated_probabilities = update_allocation_and_probabilities(nu_hat_phi2, nu_hat_phi, pis, mu, M)
#print("Updated q_t_plus_1_list:", q_t_plus_1_list)
#print("Updated M_i_list:", M_i_list)
#print("Updated probabilities:", updated_probabilities)