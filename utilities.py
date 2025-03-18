import numpy as np
import scipy.stats as stats

def std(y):
    y = np.asarray(y).flatten()
    m_y = np.mean(y)
    sd_y = np.sqrt(np.var(y, ddof=0) * (len(y) - 1) / len(y))
    return (y - m_y) / sd_y

def g(v):  # normalizes a vector
    return v / np.sqrt(np.sum(v ** 2))

def qhaar(q, n, lower_tail=True, stoperr=False, known_sigma=False, sigma_hat=1):
    """
    CDF for the first (or any) element of a random vector distributed uniformly on S_{n-1},
    i.e., the (n-1)-dimensional unit spherical shell in ambient dimension n.
    """
    if known_sigma:
        return stats.norm.cdf(q * sigma_hat, loc=0, scale=1) if lower_tail else 1 - stats.norm.cdf(q * sigma_hat, loc=0, scale=1)
    
    if abs(q) > 1 and stoperr:
        raise ValueError("Impossible Haar quantile")
    
    if q >= 1:
        p = 1
    elif q <= -1:
        p = 0
    elif q == 0:
        p = 0.5
    elif q < 0:
        p = stats.t.sf(np.sqrt((n - 1) / (1 / q**2 - 1)), df=n - 1)
    else:
        p = 1 - stats.t.sf(np.sqrt((n - 1) / (1 / q**2 - 1)), df=n - 1)
    
    return p if lower_tail else 1 - p
