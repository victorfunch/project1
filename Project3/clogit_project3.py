import numpy as np
from scipy.stats import genextreme

def q(theta, y, x): 
    '''q: Criterion function, passed to estimation.estimate().
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        y: (N,) vector of outcomes (integers in 0, 1, ..., J)

    Returns
        (N,) vector. 
        
    '''
    return (-1)*(loglikelihood(theta, y, x))

def starting_values(y, x): 
    '''starting_values(): returns a "reasonable" vector of parameters from which to start estimation
    Returns
        theta0: (K,) vector of starting values for estimation
    '''
    N,J,K = x.shape
    theta = np.zeros((K,))
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    return theta

def util(theta, x, MAXRESCALE:bool=True): 
    '''util: compute the deterministic part of utility, v, and max-rescale it
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        MAXRESCALE (optional): bool, we max-rescale if True (the default)
    
    Returns
        v: (N,J) matrix of (deterministic) utility components
    '''
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    N,J,K = x.shape 

    # deterministic utility 
    v = x @ theta

    if MAXRESCALE:
        v -= v.max(axis=1, keepdims=True)
    
    return v 

def loglikelihood(theta, y, x): 
    '''loglikelihood()
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
        y: (N,) vector of outcomes (integers in 0, 1, ..., J)
    
    Returns
        ll_i: (N,) vector of loglikelihood contributions
    '''
    assert theta.ndim == 1 
    N,J,K = x.shape 

    # deterministic utility 
    v = util(theta, x, MAXRESCALE=True)
        
    s = np.exp(v) / np.sum(np.exp(v), axis=1, keepdims=True)
    
    s = np.fmax(s, 1e-8)    # truncate below at 0.00000001 
    s = np.fmin(s, 1.-1e-8) # truncate above at 0.99999999
    
    # likelihood 
    ll_i = np.sum(y*np.log(s), axis=1)

    assert ll_i.ndim == 1 # we should return an (N,) vector 

    return ll_i


def choice_prob(theta, x):
    '''choice_prob(): Computes the (N,J) matrix of choice probabilities 
    Args. 
        theta: (K,) vector of parameters 
        x: (N,J,K) matrix of covariates 
    
    Returns
        ccp: (N,J) matrix of probabilities 
    '''
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    N, J, K = x.shape
    
    # deterministic utility 
    v = util(theta, x, MAXRESCALE=True)
    
    # denominator 
    denom = np.exp(v).sum(axis=1).reshape((-1,1))
    assert denom.ndim == 2 # denom must be (N,1) so we can divide an (N,J) matrix with it without broadcasting errors
    
    # Conditional choice probabilites
    ccp = np.exp(v) / denom
    
    return ccp


def sim_data(N: int, theta: np.ndarray, J: int) -> tuple:
    """Takes input values N and J to specify the shape of the output data. The
    K dimension is inferred from the length of theta. Creates a y column vector
    that are the choice that maximises utility, and a x matrix that are the 
    covariates, drawn from a random normal distribution.

    Args:
        N (int): Number of households.'
        J (int): Number of choices.
        theta (np.ndarray): The true value of the coefficients.

    Returns:
        tuple: y,x
    """
    assert theta.ndim == 1, f'theta should have ndim == 1, got {theta.ndim}'
    
    K = theta.size
    
    # 1. draw explanatory variables 
    x = np.random.normal(size=(N,J,K))

    # 2. draw error term 
    uni = np.random.uniform (size=(N,J))
    e = genextreme.ppf(uni, c=0)

    # 3. deterministic part of utility (N,J)
    v = x @ theta # (N,J) matrix of "observable utilities"

    # 4. full utility 
    u = v + e # (unobserved)
    
    # 5. chosen alternative
    # Find which choice that maximises value: this is the discrete choice 
    y = u.argmax(axis=1)
    assert y.ndim == 1 # y must be 1D
    
    return y,x