import numpy as np

def kl_divergence(p, q, eps=1e-12):
    
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Add epsilon to prevent log(0)
    q = q + eps

    # Mask where p > 0 (since p*log(p/q)=0 when p=0)
    mask = p > 0

    # Compute KL divergence
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return float(kl)