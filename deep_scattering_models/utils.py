"""Module with auxiliary non-scattering related functions."""

import numpy as np

def cwishrnd(M, looks_num=15):
    """Add wishard noise...

    Parameters
    ----------
    M: ``numpy.ndarray``        
        Square Hermitian definite positive array.
    looks_num : ``int``, default: 15          
        Number of looks wished to simulate.          

    Returns
    -------
    ``numpy.ndarray``      
        M matrix of input with wishard noise added.
    
    """
    # Set looks limit
    MAX_LOOKS = 81

    # Check square matrix (Add Hermitian and positive definite)
    n, m = np.shape(M)  
    if n != m: 
        raise ValueError('Array shape mismatch. M must be a square array')
    
    #elif not(np.allclose(M, np.matrix(M).H, rtol=1e-05, atol=1e-08)):
    #    raise ValueError('M must be hermitian')    
    
    # Cholesky Decomposition and ...
    d = np.linalg.cholesky(M) 
    
    if (looks_num <= MAX_LOOKS + n): 
        # Random defaul Generator and random matrix
        rng = np.random.default_rng()
        rand_real = rng.standard_normal((looks_num, d.shape[0]))
        rand_img = rng.standard_normal((looks_num, d.shape[0]))

        # ...
        x = np.matmul((rand_real + 1j*rand_img)/np.sqrt(2), d.T) 

        return np.matmul(np.conj(x.T), x)

    else:
      raise ValueError(
          'Number of looks "look_num" must not exceed number of rows in '
          f'M plus {MAX_LOOKS}: {MAX_LOOKS + n}'
          )