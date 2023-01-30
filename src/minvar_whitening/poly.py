import numpy as np
from scipy.linalg import fractional_matrix_power
import sympy as sym
import math

def sigma_jk(sigma, j, k):
    """Returns a vector of the matrix sigma to the matrix power i for i in range (j, ..., j + k -1)

    Input:
     - sigma: covariance matrix of the data
     - j: user defined value. Constraint 1, j = 1/2. Constraint 2, j = 1.
     - k: k - 1 is the degree of the polynomial to be calculated

    Output:
     - Sigma_{(j,k)} is a vector of sigma to powers between j and j + k - 1
    
    """ 

    sigma = np.real(sigma)
    if type(j) == int:
        sigma_jk_list = [np.linalg.matrix_power(sigma, j)]
    else:
        sigma_jk_list = [fractional_matrix_power(sigma, j)]
    
    for i in range(1, k):
        sigma_jk_list.append(sigma_jk_list[i-1] @ sigma)
        
    return sigma_jk_list


def s_j(sigma, j):
    """
    S_{j} = trace(Sigma^{j}). 

    Input: 
     - sigma: covariance matrix of data
     - j: scalar, can be int or float
    
    Output:
     - S_{j} = trace(Sigma^{j}): scalar
    """
    if type(j) == int:
        return np.trace(np.linalg.matrix_power(sigma, j))
    else:
        return np.trace(fractional_matrix_power(sigma, j))

def M_matrix(sigma, k):
    """
    Creates the matrix M used to calculate the minimal-variance polynomial approximation. 
    
    M_{k}_ij = s_{i+j-1} for i, j in {1, ..., k}.

    Input: 
     - sigma: covariance matrix of data
     - j: scalar, can be int or float

    Output:
     - M: matrix of traces
    """
    M = np.zeros([k, k])
    
    for i in range(k):
        for j in range(k):
            M[i, j] = s_j(sigma, i+j+1)
            
    return M

def theta_vector(sigma, k):
    """
    Creates the vector of coefficients to be used in the minimal-variance polynomial approximation when using constraint 1. 

    Input: 
     - sigma: covariance matrix of data
     - j: scalar, can be int or float

    Output:
     - theta: vector of coefficients  
    """
    M = M_matrix(sigma, k)
    frac_power_matrices = sigma_jk(sigma, 1/2, k)
    S = np.array([np.trace(i) for i in frac_power_matrices])
    s0 = sigma.shape[0]

    MinvS = np.linalg.pinv(M) @ S
    
    denominator = S.T @ MinvS
    
    omega = s0/denominator
    
    theta = omega * MinvS
    
    return np.array(theta, dtype=float)


def A_matrix(sigma, k):
    """
    Create the minimal-variance polynomial approximation to Sigma^{-1/2} for decorrelation. 

    Input:
     - sigma: covariance matrix of data
     - k: integer scalar > 1, k - 1 = degree of polynomial
    Output:
     - A: matrix which approximates Sigma^{-1/2}
    """
    theta = theta_vector(sigma, k)

    matrix = np.identity(sigma.shape[0])
    A = np.zeros_like(matrix)
    for i in range(k):
        A += theta[i] * matrix
        matrix = matrix @ sigma
        
    return np.array(A, dtype = float)


def A_matrix_poly(sigma, k, t):
    """
    Create the minimal-variance polynomial to Sigma^{-1/2} for decorrelation, and return the polynomial. 

    Input:
     - sigma: covariance matrix of data
     - k: integer scalar > 1, k - 1 = degree of polynomial
     - t: the variable to write the polynomial in. Recommended to use sym.MatrixSymbol.

    Output:
     - A: polynomial which, when t=Sigma, returns polynomial approximation to Sigma^{-1/2}
    """
    theta = theta_vector(sigma, k)

    A = 0 
    for i in range(k):
        A += theta[i] * (t**i)
        
    return A
    
def analytical_c(sigma, polynomial, weight_power=1):
    """
    Used when correcting polynomial after it has been found, in the case of many degenerate attributes.

    Input:
     - sigma: covariance matrix of data
     - polynomial: the polynomial given by the A_matrix_poly
     - weight_power: the power of the eigenvalues used as weights 

    Output:
     - c*: multiplying the polynomial by this scalar will improve the fit of the polynomial to the larger eigenvalues.
    """ 

    eigs = np.linalg.eigvals(sigma)
    eigs = eigs[eigs > 10e-20].astype(float)
    len_eigs_nonzero = len(eigs)
    
    weights = eigs**weight_power

    eigs05 = eigs**(-0.5)
    
    t = sym.Symbol('t')
    polynomial = sym.lambdify(t, polynomial)
    rho_eigs = polynomial(eigs)
        
    numerator = sum([weights[i] * eigs05[i] * rho_eigs[i] for i in range(len_eigs_nonzero)])
    denominator = sum([weights[i] * (rho_eigs[i]**2) for i in range(len_eigs_nonzero)])
    
    return numerator/denominator 

def adjustment_value(sigma, k, weight_power=1):
    """
    Finds the adjustment value for you using the analytical_c function, without the need to calculate the polynomial oneself. 

    Input:
     - sigma: covariance matrix of data
     - k: k-1 is the degree of the polynomial to be found
     - weight_power: the power of the eigenvalues used as weights 

    Output:
     - c*: multiplying the polynomial by this scalar will improve the fit of the polynomial to the larger eigenvalues.
    """
    d = sigma.shape[0]
    
    t = sym.Symbol('t')
    A_poly = A_matrix_poly(sigma, k, t)
    c = analytical_c(sigma, A_poly, weight_power)
    
    return c

def apply_whitening(sigma, X, k, adjust=False, weight_power=1, return_all=True, dtype=np.float64):
    """
    Find the polynomial whitening matrix A and perform the whitening.

    Input:
     - sigma: covariance matrix of data
     - X: the data to be whitened
     - k: k-1 is the degree of the polynomial to be found
     - weight_power: the power of the eigenvalues used as weights 

    Output:
     - X: data whitened by the polynomial whitening matrix A
     - A: the polynomial whitening matrix
     - sigma: the covariance matrix of the data whitened by the polynomial whitening matrix
    """
    A = A_matrix(sigma, k)
    
    if adjust == True:
        c = adjustment_value(sigma, k, weight_power)
    else:
        c = 1 
        
    X = np.array(c*A @ (X - np.mean(X, axis = 1)[:, np.newaxis]), dtype=dtype)
    sigma = np.array(np.cov(X), dtype=dtype)

    if return_all == False:
        return X
    else: 
        return X, A, sigma
