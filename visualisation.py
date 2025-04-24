import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def visualise(P, x1, algo1, x2, algo2):

	x1 = np.array(x1)
	x2 = np.array(x2)

	# equal-weight portfolio for comparison
	equal_weights = np.ones_like(x1) / len(x1)

	# compute daily portfolio returns
	port1_returns = P.dot(x1)
	port2_returns = P.dot(x2)
	equal_returns = P.dot(equal_weights)


	# compute cumulative returns (wealth index)
	cum_port1 = np.cumprod(1 + port1_returns)
	cum_port2 = np.cumprod(1 + port2_returns)
	cum_equal = np.cumprod(1 + equal_returns)

	# final wealth levels
	final_wealth_custom1 = cum_port1[-1]
	final_wealth_custom2 = cum_port2[-1]
	final_wealth_equal  = cum_equal[-1]

	# total returns
	total_return_custom1 = final_wealth_custom1 - 1
	total_return_custom2 = final_wealth_custom2 - 1
	total_return_equal  = final_wealth_equal  - 1

	# print out
	print(f"{algo1} portfolio final wealth: {final_wealth_custom1:.4f}")
	print(f" → Total return: {total_return_custom1:.2%}")
	print(f"{algo2} portfolio final wealth: {final_wealth_custom2:.4f}")
	print(f" → Total return: {total_return_custom2:.2%}")
	print(f"Equal-weight portfolio final wealth: {final_wealth_equal:.4f}")
	print(f" → Total return: {total_return_equal:.2%}")

	# comparison
	difference1 = total_return_custom1 - total_return_equal
	print(f"Difference in total return 1 : {difference1:.2%}")
	difference2 = total_return_custom2 - total_return_equal
	print(f"Difference in total return 2 : {difference2:.2%}")

	# plot the results
	plt.figure()
	plt.plot(cum_port1, label=f'{algo1} Weights')
	plt.plot(cum_port2, label=f'{algo2} Weights')
	plt.plot(cum_equal, label='Equal Weights')
	plt.xlabel('Time Step')
	plt.ylabel('Cumulative Return')
	plt.title('Portfolio Cumulative Returns Over Time')
	plt.legend()
	plt.tight_layout()
	plt.show()

def get_return_metrics(P, x1, algo1):

	print("P stats:", P.min(), P.max(), P.mean())

	x1 = np.array(x1)
	print("X1 sum ", x1.sum())

	# equal-weight portfolio
	equal_weights = np.ones_like(x1) / len(x1)
	print("equal_weights sum ", equal_weights.sum())

	# compute daily portfolio returns
	port1_returns = P.dot(x1)
	equal_returns = P.dot(equal_weights)

	# compute cumulative returns (wealth index)
	cum_port1 = np.cumprod(1 + port1_returns)
	cum_equal = np.cumprod(1 + equal_returns)

	# final wealth levels
	final_wealth_custom1 = cum_port1[-1]
	final_wealth_equal  = cum_equal[-1]

	# total returns
	total_return_custom1 = final_wealth_custom1 - 1
	total_return_equal  = final_wealth_equal  - 1

	# print out
	print(f"{algo1} portfolio final wealth: {final_wealth_custom1:.4f}")
	print(f" → Total return: {total_return_custom1:.2%}")
	print(f"Equal-weight portfolio final wealth: {final_wealth_equal:.4f}")
	print(f" → Total return: {total_return_equal:.2%}")

	# comparison
	difference1 = total_return_custom1 - total_return_equal
	print(f"Difference in total return 1 : {difference1:.2%}")