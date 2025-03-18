import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LassoCV

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

def l_cdf_glmnet(x, y, X, ind, lambda_val, lambda_cv, lasso_obj=None, adjusted=False, return_both=False, return_dropprob=False):
    """ CDF calculation using LassoCV (equivalent to glmnet in R) """
    n, p = X.shape
    y_std = std(y)
    Z = np.column_stack((np.ones(n), np.delete(X, ind, axis=1)))
    proj = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    y_std_hat = proj @ y_std
    sigma_hat_std = np.sqrt(np.sum((y_std - y_std_hat) ** 2))
    second_denom_term = np.sqrt(np.sum(((np.eye(n) - proj) @ X[:, ind]) ** 2))
    
    if lasso_obj is None:
        lasso_obj = [LassoCV(cv=5).fit(X, y_std), LassoCV(cv=5).fit(np.delete(X, ind, axis=1), y_std)]
    
    beta_x = lasso_obj[1].predict(np.delete(X, ind, axis=1))
    v_x = (-np.sum(X[:, ind] * (y_std_hat - X[:, ind] - Z @ beta_x)) + n * lambda_cv * np.sign(x)) / (sigma_hat_std * second_denom_term)
    
    beta_0 = lasso_obj[0].predict(np.delete(X, ind, axis=1))
    v1 = (-np.sum(X[:, ind] * (y_std_hat - Z @ beta_0)) - n * lambda_val) / (sigma_hat_std * second_denom_term)
    v2 = (-np.sum(X[:, ind] * (y_std_hat - Z @ beta_0)) + n * lambda_val) / (sigma_hat_std * second_denom_term)
    
    denom = 1 - (qhaar(v2, n - p, True) - qhaar(v1, n - p, True))
    uncond_prob = qhaar(v_x, n - p, True)
    
    if not adjusted:
        return uncond_prob
    
    beta_hat = lasso_obj[0].predict(X)
    cond_prob = (qhaar(v_x, n - p, True) - qhaar(min(v_x, v2), n - p, True) + qhaar(v1, n - p, True)) / denom
    
    if return_both:
        return uncond_prob, cond_prob
    if return_dropprob:
        return cond_prob, 1 - denom
    return cond_prob

def l_test(y, X, ind, lambda_val=-1, lambda_cv=-1, lasso_obj=None, adjusted=False, smoothed=True, return_both=False):
    """ Performs the L-test with optional adjustments """
    n, p = X.shape
    if p <= 2:
        raise ValueError("The dimension needs to be at least 3")
    
    y_std = std(y)
    Z = np.column_stack((np.ones(n), np.delete(X, ind, axis=1)))
    proj = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    y_std_hat = proj @ y_std
    sigma_hat_std = np.sqrt(np.sum((y_std - y_std_hat) ** 2))
    
    if lasso_obj is None:
        lasso_obj = [LassoCV(cv=5).fit(X, y_std), LassoCV(cv=5).fit(np.delete(X, ind, axis=1), y_std)]
    
    x = abs(lasso_obj[0].coef_[ind])
    
    if x == 0:
        uncond_pval = 1
        if not adjusted:
            return uncond_pval
    
    pval_right = 1 - l_cdf_glmnet(x, y, X, ind, lambda_val, lambda_cv, lasso_obj, adjusted)
    pval_left = l_cdf_glmnet(-x, y, X, ind, lambda_val, lambda_cv, lasso_obj, adjusted)
    pval = pval_left + pval_right
    
    return pval
