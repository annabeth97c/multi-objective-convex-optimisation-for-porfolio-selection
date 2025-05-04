import numpy as np
import matplotlib.pyplot as plt

from filtered_dataset import FilteredDataset
from optimisation_algorithms import *
from visualisation import *

filtered_dataset_csv = 'filtered_stock_prices_with_returns.csv'

us_stocks_dataset = FilteredDataset(filtered_dataset_csv)
us_stocks_price_change_data = us_stocks_dataset.load_dataset()
n = us_stocks_dataset.get_stocks_count()

#--- USER: replace these with your actual data and solver calls ---#
price_change_data = us_stocks_price_change_data  # your (T × n) array
# n = price_change_data.shape[1]
mu = price_change_data.mean(axis=0)
cov = np.cov(price_change_data, rowvar=False)

# placeholder solver interfaces; replace with your actual functions
def cvx_weights(P, lam, chebyshev=False):
    # returns weight vector for CVXPY solve
    return constrained_convex_problem_cvxpy(P, lam, n, chebyshev)[0]

def pgd_weights(P, lam, chebyshev=False, epsilon = 1e-3):
    # returns weight vector for your PGD implementation
    pgd = ProjectedGradientDescent(lam, chebyshev, epsilon)
    pgd.set_data(P)
    return pgd.run()[0]

# sweep lambda
lambdas = np.linspace(0.01, 0.99, 21)

# collect (variance, return) for each method
results = {
    ('CVXPY', 'Linear Scalarization'): [],
    ('CVXPY', 'Chebyshev Scalarization'): [],
    ('PGD', 'Linear Scalarization'): [],
    ('PGD', 'Chebyshev Scalarization'): []
}

for lam in lambdas:
    # CVXPY
    w = cvx_weights(price_change_data, lam, chebyshev=False)
    w = np.array(w)
    results[('CVXPY','Linear Scalarization')].append((w.T @ cov @ w, mu.dot(w)))
    w = cvx_weights(price_change_data, lam, chebyshev=True)
    w = np.array(w)
    results[('CVXPY', 'Chebyshev Scalarization')].append((w.T @ cov @ w, mu.dot(w)))
    # PGD combinations
    for cheb in [False, True]:
        w = pgd_weights(price_change_data, lam, chebyshev=cheb, epsilon=1e-3)
        w = np.array(w)
        key = ('PGD', 'Chebyshev Scalarization' if cheb else 'Linear Scalarization')
        results[key].append((w.T @ cov @ w, mu.dot(w)))

# Pareto filter
def is_pareto(points):
    mask = np.ones(len(points), dtype=bool)
    for i, pi in enumerate(points):
        for j, pj in enumerate(points):
            if (pj[0] <= pi[0] and pj[1] >= pi[1]) and (pj[0] < pi[0] or pj[1] > pi[1]):
                mask[i] = False
                break
    return mask

# plot
# plt.figure(figsize=(8,6))
# styles = {
#     ('CVXPY','Linear Scalarization'):     ('o-', 'red'),
#     ('CVXPY','Chebyshev Scalarization'):    ('s--', 'blue'),
#     ('PGD','Linear Scalarization'): ('^-.', 'green'),
#     ('PGD','Chebyshev Scalarization'): ('*-.', 'orange')
# }

# for key, pts in results.items():
#     if key[1] == 'Linear Scalarization':
#         arr = np.array(pts)
#         mask = is_pareto(arr)
#         style, color = styles[key]
#         plt.plot(arr[mask,0], arr[mask,1], style, color=color, label="{} {}".format(*key))

# plt.xlabel('Portfolio Variance', fontsize=14)
# plt.ylabel('Expected Return', fontsize=14)
# plt.title('Linear Scalarisation Pareto Front Comparison', fontsize=16)
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(8,6))
styles = {
    ('CVXPY','Linear Scalarization'):     ('o-', 'red'),
    ('CVXPY','Chebyshev Scalarization'):    ('s--', 'blue'),
    ('PGD','Linear Scalarization'): ('^-.', 'green'),
    ('PGD','Chebyshev Scalarization'): ('*-.', 'orange')
}

for key, pts in results.items():
    if key[1] == 'Linear Scalarization':
        arr = np.array(pts)
        mask = is_pareto(arr)
        style, color = styles[key]
        plt.plot(arr[mask,0], arr[mask,1], style, color=color, label=f"{key[0]} {key[1]}")

# enlarge axis labels
plt.xlabel('Portfolio Variance', fontsize=16)
plt.ylabel('Expected Return', fontsize=16)

# enlarge tick labels
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# enlarge title
plt.title('Linear Scalarisation Pareto Front Comparison', fontsize=18)

# larger legend text
plt.legend(loc='lower right', fontsize=14)

plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
styles = {
    ('CVXPY','Linear Scalarization'):     ('o-', 'red'),
    ('CVXPY','Chebyshev Scalarization'):    ('s--', 'blue'),
    ('PGD','Linear Scalarization'): ('^-.', 'green'),
    ('PGD','Chebyshev Scalarization'): ('*-.', 'orange')
}

for key, pts in results.items():
    if key[1] == 'Chebyshev Scalarization':
        arr = np.array(pts)
        mask = is_pareto(arr)
        style, color = styles[key]
        plt.plot(arr[mask,0], arr[mask,1], style, color=color, label=f"{key[0]} {key[1]}")

# enlarge axis labels
plt.xlabel('Portfolio Variance', fontsize=16)
plt.ylabel('Expected Return', fontsize=16)

# enlarge tick labels
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# enlarge title
plt.title('Chebyshev Scalarisation Pareto Front Comparison', fontsize=18)

# larger legend text
plt.legend(loc='lower right', fontsize=14)

plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8,6))
# styles = {
#     ('CVXPY','Linear Scalarization'):     ('o-', 'red'),
#     ('CVXPY','Chebyshev Scalarization'):    ('s--', 'blue'),
#     ('PGD','Linear Scalarization'): ('^-.', 'green'),
#     ('PGD','Chebyshev Scalarization'): ('*-.', 'orange')
# }

# for key, pts in results.items():
#     if key[1] == 'Chebyshev Scalarization':
#         arr = np.array(pts)
#         mask = is_pareto(arr)
#         style, color = styles[key]
#         plt.plot(arr[mask,0], arr[mask,1], style, color=color, label="{} {}".format(*key))

# plt.xlabel('Portfolio Variance', fontsize=14)
# plt.ylabel('Expected Return', fontsize=14)
# plt.title('Chebyshev Scalarisation Pareto Front Comparison', fontsize=16)
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
