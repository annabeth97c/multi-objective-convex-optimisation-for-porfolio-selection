from filtered_dataset import FilteredDataset
from optimisation_algorithms import *
from visualisation import *

filtered_dataset_csv = 'filtered_stock_prices_with_returns.csv'

us_stocks_dataset = FilteredDataset(filtered_dataset_csv)
us_stocks_price_change_data = us_stocks_dataset.load_dataset()

lambda1 = 0.5

stock_weights_cvxpy, time_cvxpy = constrained_convex_problem_cvxpy(us_stocks_price_change_data, lambda1, us_stocks_dataset.get_stocks_count())
stock_weights_cvxpy_chebyshev, time_cvxpy_chebyshev = constrained_convex_problem_cvxpy(us_stocks_price_change_data, lambda1, us_stocks_dataset.get_stocks_count(), True)

projected_grad_desc = ProjectedGradientDescent(lambda1, False)
projected_grad_desc.set_data(us_stocks_price_change_data)
stock_weights_pgd, time_pgd = projected_grad_desc.run()

projected_grad_desc_chebyshev = ProjectedGradientDescent(lambda1, True)
projected_grad_desc_chebyshev.set_data(us_stocks_price_change_data)
stock_weights_pgd_chebyshev, time_pgd_chebyshev = projected_grad_desc_chebyshev.run()

print("Time CVXPY ", time_cvxpy)
print("Time CVXPY Chebyshev", time_cvxpy_chebyshev)
print("Time Projected Gradient Descent : Weighted Sum ", time_pgd)
print("Time Projected Gradient Descent : Chebyshev ", time_pgd_chebyshev)

visualise(us_stocks_price_change_data, stock_weights_cvxpy, "CVXPY", stock_weights_pgd, "Projected Gradient Descent : Weighted Sum")

visualise(us_stocks_price_change_data, stock_weights_cvxpy_chebyshev, "CVXPY : Chebyshev", stock_weights_pgd_chebyshev, "Projected Gradient Descent : Chebyshev")
