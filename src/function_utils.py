#module contenant les différentes fonctions utiles. 
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
from scipy.stats import mvn
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
def payoff_function(x):
    #fonction phi
    return x**2


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

# à vérifier cette fonction. Mais petit draf. 

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