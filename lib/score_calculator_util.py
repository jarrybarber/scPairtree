import numpy as np
from numba import njit


@njit
def _M1_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00, scaling_factor=0):
    return np.exp(  \
                np.log(2) + \
                n11*np.log( (1-beta_i)*(1-beta_j)*phi_j +     (1-beta_i)*alpha_j*(phi_i - phi_j) +         alpha_i*alpha_j*(1-phi_i) ) + \
                n10*np.log(     (1-beta_i)*beta_j*phi_j + (1-beta_i)*(1-alpha_j)*(phi_i - phi_j) +     alpha_i*(1-alpha_j)*(1-phi_i) ) + \
                n01*np.log(     beta_i*(1-beta_j)*phi_j +         beta_i*alpha_j*(phi_i - phi_j) +     (1-alpha_i)*alpha_j*(1-phi_i) ) + \
                n00*np.log(         beta_i*beta_j*phi_j +     beta_i*(1-alpha_j)*(phi_i - phi_j) + (1-alpha_i)*(1-alpha_j)*(1-phi_i) ) - \
                scaling_factor
    )

@njit
def _M2_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00, scaling_factor=0):
    return np.exp(   \
                np.log(2) + \
                n11*np.log( (1-beta_i)*(1-beta_j)*phi_i +     (1-beta_j)*alpha_i*(phi_j - phi_i) +         alpha_i*alpha_j*(1-phi_j) ) + \
                n10*np.log(     (1-beta_i)*beta_j*phi_i +         beta_j*alpha_i*(phi_j - phi_i) +     alpha_i*(1-alpha_j)*(1-phi_j) ) + \
                n01*np.log(     beta_i*(1-beta_j)*phi_i + (1-beta_j)*(1-alpha_i)*(phi_j - phi_i) +     (1-alpha_i)*alpha_j*(1-phi_j) ) + \
                n00*np.log(         beta_i*beta_j*phi_i +     beta_j*(1-alpha_i)*(phi_j - phi_i) + (1-alpha_i)*(1-alpha_j)*(1-phi_j) ) - \
                scaling_factor
    )

@njit
def _M3_integrand(phi, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00, scaling_factor=0):
    return np.exp(   \
                n11*np.log( (1-beta_i)*(1-beta_j)*phi +         alpha_i*alpha_j*(1-phi) ) + \
                n10*np.log(     (1-beta_i)*beta_j*phi +     alpha_i*(1-alpha_j)*(1-phi) ) + \
                n01*np.log(     beta_i*(1-beta_j)*phi +     (1-alpha_i)*alpha_j*(1-phi) ) + \
                n00*np.log(         beta_i*beta_j*phi + (1-alpha_i)*(1-alpha_j)*(1-phi) ) - \
                scaling_factor
    )

@njit
def _M4_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00, scaling_factor=0):
    return np.exp(   \
                np.log(2) + \
                n11*np.log(     (1-beta_i)*alpha_j*phi_i +        alpha_i*(1-beta_j)*phi_j +         alpha_i*alpha_j*(1-phi_i-phi_j) ) + \
                n10*np.log( (1-beta_i)*(1-alpha_j)*phi_i +            alpha_i*beta_j*phi_j +     alpha_i*(1-alpha_j)*(1-phi_i-phi_j) ) + \
                n01*np.log(         beta_i*alpha_j*phi_i +    (1-alpha_i)*(1-beta_j)*phi_j +     (1-alpha_i)*alpha_j*(1-phi_i-phi_j) ) + \
                n00*np.log(     beta_i*(1-alpha_j)*phi_i +        (1-alpha_i)*beta_j*phi_j + (1-alpha_i)*(1-alpha_j)*(1-phi_i-phi_j) ) - \
                scaling_factor
    )

@njit
def _M5_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00, scaling_factor=0):
    return np.exp(   \
                (n11 + n10)*np.log( (1-beta_i)*phi_i +     alpha_i*(1-phi_i) ) + \
                (n01 + n00)*np.log(     beta_i*phi_i + (1-alpha_i)*(1-phi_i) ) + \
                (n11 + n01)*np.log( (1-beta_j)*phi_j +     alpha_j*(1-phi_j) ) + \
                (n10 + n00)*np.log(     beta_j*phi_j + (1-alpha_j)*(1-phi_j) ) - \
                scaling_factor
    )



@njit
def _M1_logged_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.log(2) + \
        n11*np.log( (1-beta_i)*(1-beta_j)*phi_j +     (1-beta_i)*alpha_j*(phi_i - phi_j) +         alpha_i*alpha_j*(1-phi_i) ) + \
        n10*np.log( (1-beta_i)*beta_j*phi_j     + (1-beta_i)*(1-alpha_j)*(phi_i - phi_j) +     alpha_i*(1-alpha_j)*(1-phi_i) ) + \
        n01*np.log( beta_i*(1-beta_j)*phi_j     +         beta_i*alpha_j*(phi_i - phi_j) +     (1-alpha_i)*alpha_j*(1-phi_i) ) + \
        n00*np.log( beta_i*beta_j*phi_j         +     beta_i*(1-alpha_j)*(phi_i - phi_j) + (1-alpha_i)*(1-alpha_j)*(1-phi_i) ) 

@njit
def _M2_logged_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.log(2) + \
        n11*np.log( (1-beta_i)*(1-beta_j)*phi_i +     (1-beta_j)*alpha_i*(phi_j - phi_i) +         alpha_i*alpha_j*(1-phi_j) ) + \
        n10*np.log( (1-beta_i)*beta_j*phi_i     +         beta_j*alpha_i*(phi_j - phi_i) +     alpha_i*(1-alpha_j)*(1-phi_j) ) + \
        n01*np.log( beta_i*(1-beta_j)*phi_i     + (1-beta_j)*(1-alpha_i)*(phi_j - phi_i) +     (1-alpha_i)*alpha_j*(1-phi_j) ) + \
        n00*np.log( beta_i*beta_j*phi_i         +     beta_j*(1-alpha_i)*(phi_j - phi_i) + (1-alpha_i)*(1-alpha_j)*(1-phi_j) )  

@njit
def _M3_logged_integrand(phi, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        n11*np.log( (1-beta_i)*(1-beta_j)*phi +         alpha_i*alpha_j*(1-phi) ) + \
        n10*np.log( (1-beta_i)*beta_j*phi     +     alpha_i*(1-alpha_j)*(1-phi) ) + \
        n01*np.log( beta_i*(1-beta_j)*phi     +     (1-alpha_i)*alpha_j*(1-phi) ) + \
        n00*np.log( beta_i*beta_j*phi         + (1-alpha_i)*(1-alpha_j)*(1-phi) )

@njit
def _M4_logged_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.log(2) + \
        n11*np.log( (1-beta_i)*alpha_j*phi_i     +        alpha_i*(1-beta_j)*phi_j +         alpha_i*alpha_j*(1-phi_i-phi_j) ) + \
        n10*np.log( (1-beta_i)*(1-alpha_j)*phi_i +            alpha_i*beta_j*phi_j +     alpha_i*(1-alpha_j)*(1-phi_i-phi_j) ) + \
        n01*np.log( beta_i*alpha_j*phi_i         +    (1-alpha_i)*(1-beta_j)*phi_j +     (1-alpha_i)*alpha_j*(1-phi_i-phi_j) ) + \
        n00*np.log( beta_i*(1-alpha_j)*phi_i     +        (1-alpha_i)*beta_j*phi_j + (1-alpha_i)*(1-alpha_j)*(1-phi_i-phi_j) )

@njit
def _M5_logged_integrand(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        (n11 + n10)*np.log( (1-beta_i)*phi_i + alpha_i*(1-phi_i) ) + \
        (n01 + n00)*np.log( beta_i*phi_i + (1-alpha_i)*(1-phi_i) ) + \
        (n11 + n01)*np.log( (1-beta_j)*phi_j + alpha_j*(1-phi_j) ) + \
        (n10 + n00)*np.log( beta_j*phi_j + (1-alpha_j)*(1-phi_j) )





@njit
def _M1_logged_integrand_jacobian(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.array([
        n11*    (1-beta_i-alpha_i)*alpha_j/( (1-beta_i)*(1-beta_j)*phi_j +     (1-beta_i)*alpha_j*(phi_i - phi_j) +         alpha_i*alpha_j*(1-phi_i) ) + \
        n10*(1-beta_i-alpha_i)*(1-alpha_j)/(     (1-beta_i)*beta_j*phi_j + (1-beta_i)*(1-alpha_j)*(phi_i - phi_j) +     alpha_i*(1-alpha_j)*(1-phi_i) ) + \
        n01*    (beta_i+alpha_i-1)*alpha_j/(     beta_i*(1-beta_j)*phi_j +         beta_i*alpha_j*(phi_i - phi_j) +     (1-alpha_i)*alpha_j*(1-phi_i) ) + \
        n00*(beta_i+alpha_i-1)*(1-alpha_j)/(         beta_i*beta_j*phi_j +     beta_i*(1-alpha_j)*(phi_i - phi_j) + (1-alpha_i)*(1-alpha_j)*(1-phi_i) )
        ,
        n11*(1-beta_i)*(1-alpha_j-beta_j)/( (1-beta_i)*(1-beta_j)*phi_j +     (1-beta_i)*alpha_j*(phi_i - phi_j) +         alpha_i*alpha_j*(1-phi_i) ) + \
        n10*(1-beta_i)*(alpha_j+beta_j-1)/(     (1-beta_i)*beta_j*phi_j + (1-beta_i)*(1-alpha_j)*(phi_i - phi_j) +     alpha_i*(1-alpha_j)*(1-phi_i) ) + \
        n01*    beta_i*(1-alpha_j-beta_j)/(     beta_i*(1-beta_j)*phi_j +         beta_i*alpha_j*(phi_i - phi_j) +     (1-alpha_i)*alpha_j*(1-phi_i) ) + \
        n00*    beta_i*(alpha_j+beta_j-1)/(         beta_i*beta_j*phi_j +     beta_i*(1-alpha_j)*(phi_i - phi_j) + (1-alpha_i)*(1-alpha_j)*(1-phi_i) )
        ])

@njit
def _M2_logged_integrand_jacobian(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.array([
        n11*(1-beta_j)*(1-alpha_i-beta_i)/( (1-beta_i)*(1-beta_j)*phi_i +     (1-beta_j)*alpha_i*(phi_j - phi_i) +         alpha_i*alpha_j*(1-phi_j) ) + \
        n10*    beta_j*(1-alpha_i-beta_i)/( (1-beta_i)*beta_j*phi_i     +         beta_j*alpha_i*(phi_j - phi_i) +     alpha_i*(1-alpha_j)*(1-phi_j) ) + \
        n01*(1-beta_j)*(alpha_i+beta_i-1)/( beta_i*(1-beta_j)*phi_i     + (1-beta_j)*(1-alpha_i)*(phi_j - phi_i) +     (1-alpha_i)*alpha_j*(1-phi_j) ) + \
        n00*    beta_j*(alpha_i+beta_i-1)/( beta_i*beta_j*phi_i         +     beta_j*(1-alpha_i)*(phi_j - phi_i) + (1-alpha_i)*(1-alpha_j)*(1-phi_j) )
        ,
        n11*   (1-alpha_j-beta_j)*alpha_i /( (1-beta_i)*(1-beta_j)*phi_i +     (1-beta_j)*alpha_i*(phi_j - phi_i) +         alpha_i*alpha_j*(1-phi_j) ) + \
        n10*   (alpha_j+beta_j-1)*alpha_i /( (1-beta_i)*beta_j*phi_i     +         beta_j*alpha_i*(phi_j - phi_i) +     alpha_i*(1-alpha_j)*(1-phi_j) ) + \
        n01*(1-alpha_j-beta_j)*(1-alpha_i)/( beta_i*(1-beta_j)*phi_i     + (1-beta_j)*(1-alpha_i)*(phi_j - phi_i) +     (1-alpha_i)*alpha_j*(1-phi_j) ) + \
        n00*(alpha_j+beta_j-1)*(1-alpha_i)/( beta_i*beta_j*phi_i         +     beta_j*(1-alpha_i)*(phi_j - phi_i) + (1-alpha_i)*(1-alpha_j)*(1-phi_j) )
        ])

@njit
def _M3_logged_integrand_jacobian(phi, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        n11*((1-beta_i)*(1-beta_j) -    alpha_i *   alpha_j ) / ((1-beta_i)*(1-beta_j)*phi +         alpha_i*alpha_j*(1-phi)) + \
        n10*((1-beta_i)*   beta_j  -    alpha_i *(1-alpha_j)) / ((1-beta_i)*beta_j*phi     +     alpha_i*(1-alpha_j)*(1-phi)) + \
        n01*(   beta_i *(1-beta_j) - (1-alpha_i)*   alpha_j ) / (beta_i*(1-beta_j)*phi     +     (1-alpha_i)*alpha_j*(1-phi)) + \
        n00*(   beta_i *   beta_j  - (1-alpha_i)*(1-alpha_j)) / (beta_i*beta_j*phi         + (1-alpha_i)*(1-alpha_j)*(1-phi))

@njit
def _M4_logged_integrand_jacobian(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.array([
        n11*(1-alpha_i-beta_i)*   alpha_j /( (1-beta_i)*alpha_j*phi_i     +        alpha_i*(1-beta_j)*phi_j +         alpha_i*alpha_j*(1-phi_i-phi_j) ) + \
        n10*(1-alpha_i-beta_i)*(1-alpha_j)/( (1-beta_i)*(1-alpha_j)*phi_i +            alpha_i*beta_j*phi_j +     alpha_i*(1-alpha_j)*(1-phi_i-phi_j) ) + \
        n01*(alpha_i+beta_i-1)*   alpha_j /( beta_i*alpha_j*phi_i         +    (1-alpha_i)*(1-beta_j)*phi_j +     (1-alpha_i)*alpha_j*(1-phi_i-phi_j) ) + \
        n00*(alpha_i+beta_i-1)*(1-alpha_j)/( beta_i*(1-alpha_j)*phi_i     +        (1-alpha_i)*beta_j*phi_j + (1-alpha_i)*(1-alpha_j)*(1-phi_i-phi_j) )
        ,
        n11*(1-alpha_j-beta_j)*   alpha_i /( (1-beta_i)*alpha_j*phi_i     +        alpha_i*(1-beta_j)*phi_j +         alpha_i*alpha_j*(1-phi_i-phi_j) ) + \
        n10*(alpha_j+beta_j-1)*   alpha_i /( (1-beta_i)*(1-alpha_j)*phi_i +            alpha_i*beta_j*phi_j +     alpha_i*(1-alpha_j)*(1-phi_i-phi_j) ) + \
        n01*(1-alpha_j-beta_j)*(1-alpha_i)/( beta_i*alpha_j*phi_i         +    (1-alpha_i)*(1-beta_j)*phi_j +     (1-alpha_i)*alpha_j*(1-phi_i-phi_j) ) + \
        n00*(alpha_j+beta_j-1)*(1-alpha_i)/( beta_i*(1-alpha_j)*phi_i     +        (1-alpha_i)*beta_j*phi_j + (1-alpha_i)*(1-alpha_j)*(1-phi_i-phi_j) )
        ])

@njit
def _M5_logged_integrand_jacobian(phi_i, phi_j, alpha_i, alpha_j, beta_i, beta_j, n11, n10, n01, n00):
    return \
        np.array([
        (n11 + n10)*(1-alpha_i-beta_i) / ((1-beta_i)*phi_i +     alpha_i*(1-phi_i)) + \
        (n01 + n00)*(alpha_i+beta_i-1) / (    beta_i*phi_i + (1-alpha_i)*(1-phi_i)) 
        ,
        (n11 + n01)*(1-alpha_j-beta_j) / ((1-beta_j)*phi_j +     alpha_j*(1-phi_j)) + \
        (n10 + n00)*(alpha_j+beta_j-1) / (    beta_j*phi_j + (1-alpha_j)*(1-phi_j)) 
        ])
