import numpy as np
import matplotlib.pyplot as plt

from filtered_dataset import FilteredDataset
from optimisation_algorithms import constrained_convex_problem_cvxpy, ProjectedGradientDescent

# --- load data ---
filtered_dataset_csv = 'filtered_stock_prices_with_returns.csv'
us_stocks_dataset = FilteredDataset(filtered_dataset_csv)
P = us_stocks_dataset.load_dataset()       # shape (T, n)
T, n = P.shape

# --- sweep parameters ---
lambdas = np.linspace(0.01, 0.99, 21)
variants = [
    ('CVXPY', False, 'CVXPY – Linear'),
    ('CVXPY', True,  'CVXPY – Chebyshev'),
    ('PGD',   False, 'PGD – Linear'),
    ('PGD',   True,  'PGD – Chebyshev'),
]

# --- containers ---
wealth_curves = { label: [] for (_, _, label) in variants }
solve_times   = { label: [] for (_, _, label) in variants }

# --- main loop: solve, time & wealth ---
for lam in lambdas:
    for solver, cheb, label in variants:
        if solver == 'CVXPY':
            w, t = constrained_convex_problem_cvxpy(P, lam, n, chebyshev=cheb)
        else:
            pgd = ProjectedGradientDescent(lam, chebyshev=cheb, epsilon=1e-3)
            pgd.set_data(P)
            w, t = pgd.run()
        solve_times[label].append(t)
        
        daily_ret = P.dot(w)
        wealth    = np.cumprod(1 + daily_ret)
        wealth_curves[label].append(wealth)

# --- 1) 2×2 grid of cumulative-wealth plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()
ncols = 2

for idx, (label, curves) in enumerate(wealth_curves.items()):
    ax = axes[idx]
    arr = np.stack(curves, axis=0)      # shape (n_lambdas, T)
    finals = arr[:, -1]
    best_i = np.argmax(finals)
    
    for i, w in enumerate(arr):
        if i == best_i:
            ax.plot(w, linewidth=2.0, alpha=1.0, color='red',
                    label=f'w1={lambdas[i]:.2f} (best)')
        else:
            ax.plot(w, linewidth=1.0, alpha=0.3, label=f'w1={lambdas[i]:.2f}')
    ax.set_title(label, fontsize=12)
    ax.grid(True)
    
    # only label bottom row and left col
    row, col = divmod(idx, ncols)
    if row == 1: ax.set_xlabel('Time Step', fontsize=10)
    if col == 0: ax.set_ylabel('Wealth', fontsize=10)

# shared legend
handles, lbls = axes[-1].get_legend_handles_labels()
fig.legend(handles, lbls, loc='upper left', ncol=7, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.suptitle('Cumulative Wealth over Time for All Variants', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- compute aggregate metrics ---
total_times   = {lbl: sum(ts) for lbl, ts in solve_times.items()}

print("Total times : ", total_times)
best_returns  = {
    lbl: max(curves[i][-1] for i in range(len(curves)))
    for lbl, curves in wealth_curves.items()
}

print("Best returns : ", best_returns)

labels = list(total_times.keys())
times  = [total_times[l] for l in labels]
rets   = [best_returns[l]   for l in labels]


# --- 2) Bar: total solve time by variant ---
plt.figure(figsize=(8,5))
plt.bar(labels, times)
plt.title('Total Solve Time by Variant', fontsize=14)
plt.ylabel('Total Solve Time (s)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- 3) Bar: best cumulative return by variant ---
plt.figure(figsize=(8,5))
plt.bar(labels, rets)
plt.title('Best Cumulative Wealth by Variant', fontsize=14)
plt.ylabel('Wealth (final value)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- 4) Scatter: solve time vs best return ---
plt.figure(figsize=(8,5))
plt.scatter(times, rets, s=50)
for x, y, lbl in zip(times, rets, labels):
    plt.annotate(lbl, (x, y), textcoords='offset points', xytext=(5,5))
plt.title('Solve Time vs Best Cumulative Wealth', fontsize=14)
plt.xlabel('Total Solve Time (s)', fontsize=12)
plt.ylabel('Best Cumulative Wealth', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
