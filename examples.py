### examples.py
import numpy as np
from utilities import normalize
from l_testing import l_test, l_ci
from adjusted_l_testing import beta_x

np.random.seed(1)
n, p, s, A = 100, 50, 5, 2.3

# Generate random design matrix X with normalized columns
X = np.random.randn(n, p)
X = np.apply_along_axis(normalize, 0, X)

# Generate sparse beta vector
beta = np.zeros(p)
rand_ind = np.random.choice(p, s, replace=False)
j = rand_ind[0]  # Index to test
beta[rand_ind] = (1 - 2 * np.random.binomial(1, 0.5, s)) * A

# Generate response variable y
y = X @ beta + np.random.randn(n)

# Perform different L-tests
pval_l = l_test(y, X, j)  # l-test for H_j: beta_j = 0
pval_l_2_3 = l_test(y - 2.3 * X[:, j], X, j)  # Testing H_j(2.3): beta_j = 2.3
pval_l_lambda = l_test(y, X, j, lambda_cv=0.01)  # l-test with supplied lambda
pval_l_adjusted = l_test(y, X, j, adjusted=True, lambda_val=0.01)  # Adjusted l-test

# Generate gamma range for confidence interval testing
gamma_range = np.linspace(beta[j] - 10, beta[j] + 10, 100)

# Compute confidence intervals
ci_l = l_ci(y, X, j, gamma_range, coverage=0.95)  # l-CI

# Placeholder for adjusted CI (function to be implemented)
def l_ci_adjusted(y, X, ind, gamma_range, coverage=0.95, lambda_val=0.01):
    # This function should be implemented similar to l_ci with adjustments
    return l_ci(y, X, ind, gamma_range, coverage)

ci_l_adjusted = l_ci_adjusted(y, X, j, gamma_range, coverage=0.95, lambda_val=0.01)  # Post-selection l-CI

# Print results
print("P-Value for L-Test:", pval_l)
print("P-Value for L-Test with Beta_j=2.3:", pval_l_2_3)
print("P-Value for L-Test with Lambda CV=0.01:", pval_l_lambda)
print("Adjusted P-Value for L-Test:", pval_l_adjusted)
print("Confidence Interval (l-CI):", ci_l)
print("Adjusted Confidence Interval (l-CI):", ci_l_adjusted)
