import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    r_k = np.ones(data.shape[1]).T
    mu_k = 0.0

    for i in range(num_steps):
        tmp_vector = data@r_k 
        r_k = (tmp_vector) / np.linalg.norm(tmp_vector)
        mu_k = (np.dot(r_k, tmp_vector)) / (np.dot(r_k, r_k))
    
    return float(mu_k),r_k 