import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LassoCV
import cvxpy as cp

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

def beta_x(x, y, X, ind, lamb):
    n, p = X.shape
    Z = np.column_stack((np.ones(n), np.delete(X, ind, axis=1)))
    beta = cp.Variable(p)
    objective = cp.Minimize(cp.sum_squares(y - x * X[:, ind] - Z @ beta) / (2 * n) + lamb * cp.norm(beta[1:], 1))
    problem = cp.Problem(objective)
    problem.solve()
    return np.round(beta.value, 9)

def beta_full(y, X, ind, lamb):
    n, p = X.shape
    Z = np.column_stack((np.ones(n), X))
    beta = cp.Variable(p + 1)
    objective = cp.Minimize(cp.sum_squares(y - Z @ beta) / (2 * n) + lamb * cp.norm(beta[1:], 1))
    problem = cp.Problem(objective)
    problem.solve()
    return np.round(beta.value, 9)

def l_cdf_adjusted(x, y, X, ind, gamma, lambda_cv, lambda_val, tail='left', smoothed=False):
    n, p = X.shape
    Z = np.column_stack((np.ones(n), np.delete(X, ind, axis=1)))
    proj = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    y_hat = proj @ y
    sigma_hat = np.sqrt(np.sum(((np.eye(n) - proj) @ (y - X[:, ind] * gamma)) ** 2))
    second_denom_term = np.sqrt(np.sum(((np.eye(n) - proj) @ X[:, ind]) ** 2))
    
    beta_x_val = beta_x(0, y - gamma * X[:, ind], X, ind, lambda_cv)
    mid = -np.sum(X[:, ind] * (y_hat - gamma * (proj @ X[:, ind]) - Z @ beta_x_val)) / (sigma_hat * second_denom_term)
    u1 = np.sum(X[:, ind] * (y - y_hat - gamma * (np.eye(n) - proj) @ X[:, ind])) / (sigma_hat * second_denom_term)
    
    if smoothed and x == 0:
        dist_from_mid = abs(mid - u1)
        v_x = mid - dist_from_mid if tail == 'left' else mid + dist_from_mid
    else:
        v_x = (-np.sum(X[:, ind] * (y_hat - gamma * (proj @ X[:, ind]) - x * X[:, ind] - Z @ beta_x_val)) + n * lambda_cv * np.sign(x)) / (sigma_hat * second_denom_term)
    
    beta_0 = beta_x(0, y, X, ind, lambda_val)
    v1 = (-np.sum(X[:, ind] * (y_hat + gamma * (np.eye(n) - proj) @ X[:, ind] - Z @ beta_0)) - n * lambda_val) / (sigma_hat * second_denom_term)
    v2 = (-np.sum(X[:, ind] * (y_hat + gamma * (np.eye(n) - proj) @ X[:, ind] - Z @ beta_0)) + n * lambda_val) / (sigma_hat * second_denom_term)
    
    denom = 1 - (qhaar(v2, n - p, True) - qhaar(v1, n - p, True))
    numer_1 = qhaar(v_x, n - p, True)
    numer_2 = qhaar(min(v_x, v2), n - p, True) - qhaar(v1, n - p, True) if v_x > v1 else 0
    cond_prob = (numer_1 - numer_2) / denom
    
    return 1 - cond_prob if tail == 'right' else cond_prob
