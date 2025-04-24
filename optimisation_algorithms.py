import numpy as np
import cvxpy as cp
import math
import time

def constrained_convex_problem_cvxpy(P: np.array, lambda1: float, n: int, chebyshev: bool = False) -> tuple:
	mu = np.mean(P, axis=0)

	# Covariance matrix (Σ) based on the price changes
	cov_matrix = np.cov(P, rowvar=False)

	# Define variables
	x = cp.Variable(n)  # Portfolio allocation (n stocks)

	# Objective function
	expected_return = -mu.T@x  # The negative of the expected return

	# Variance term (quadratic term with the covariance matrix)
	variance_term = cp.quad_form(x, cov_matrix)

	# Constraints
	constraints = [
		# lambda1 + lambda2 == 1,  # lambda1 and lambda2 sum to 1
		# lambda1 >= 0,             # lambda1 >= 0
		# lambda2 >= 0,             # lambda2 >= 0
		cp.sum(x) == 1,           # Portfolio weights sum to 1
		x >= 0,                   # No short-selling (weights >= 0)
	]

	if not chebyshev:
		# Objective to minimize (scalarized)
		objective = cp.Minimize(lambda1 * expected_return + (1 - lambda1) * variance_term)
	else:
		minus_mu = -1*mu
		f_star_return = minus_mu.min()
		f_star_variance = cp.Variable(n)
		f_star_variance_problem = cp.Problem(
		        cp.Minimize(cp.quad_form(f_star_variance, cov_matrix)),
		        [f_star_variance >= 0, cp.sum(f_star_variance) == 1]
		    )
		f_star_variance_problem.solve()
		f_star_variance = f_star_variance_problem.value

		t = cp.Variable()
		constraints += [ lambda1* (-mu.T @ x - f_star_return) <= t, (1-lambda1)* (cp.quad_form(x, cov_matrix) - f_star_variance) <= t]
		objective = cp.Minimize(t)

	# Formulate the problem
	problem = cp.Problem(objective, constraints)

	before_stamp = time.time()
	# Solve the problem
	problem.solve()
	timer = time.time() - before_stamp

	return x.value.tolist(), timer


class ProjectedGradientDescent:
	"""docstring for ProjectedGradientDescent"""
	def __init__(self, lambda1, chebyshev=False, epsilon=1e-6):
		super(ProjectedGradientDescent, self).__init__()

		self.chebyshev = chebyshev
		self.lambda1 = lambda1
		self.lambda2 = 1 - lambda1
		self.epsilon = epsilon

	def set_data(self, P: np.array):
		self.mu = np.mean(P, axis=0)

		# Calculate the covariance matrix (Σ) for the stock price changes
		self.cov_matrix = np.cov(P, rowvar=False)

		# Lipschitz constant L = 2*(λ2)*λ_max(Σ)
		eigs       = np.linalg.eigvalsh(self.cov_matrix)
		lambda_max = eigs[-1]
		self.L     = 2 * (self.lambda2) * lambda_max

		# choose the optimal step-size
		self.learning_rate = 1.0 / self.L

		self.max_iter = math.ceil((self.L * 2)/self.epsilon) + 100

		self.f_star_return = np.min(-self.mu)  # Ideal expected return (best possible return)
		self.f_star_variance = np.min(np.diag(self.cov_matrix))  # Ideal variance (best possible variance)

	def compute_chebyshev_gradient(self, x):

		# Gradient of expected return (negative of mu)
		grad_expected_return = -self.mu
		
		# Gradient of the variance term (2 * cov_matrix @ x)
		grad_variance = 2 * self.cov_matrix @ x

		# Calculate the deviations from the ideal point
		dev_f1 = -self.mu @ x - self.f_star_return
		dev_f2 = x.T @ self.cov_matrix @ x - self.f_star_variance

		# Calculate weighted absolute deviations from the ideal point
		weighted_dev1 = self.lambda1 * abs(dev_f1)
		weighted_dev2 = self.lambda2 * abs(dev_f2)

		# Pick the index with the larger deviation and form its subgradient
		if weighted_dev1 > weighted_dev2:
			return self.lambda1 * np.sign(dev_f1) * grad_expected_return
		elif weighted_dev2 > weighted_dev1:
			return self.lambda2 * np.sign(dev_f2) * grad_variance
		else:
			# If both are same, we get the average of the subgradients
			sub_grad1 = self.lambda1 * np.sign(f1 - self.f_star_return) * grad_expected_return
			sub_grad2 = self.lambda2 * np.sign(f2 - self.f_star_variance) * grad_variance
			return 0.5 * (grad1 + grad2)

		# return self.lambda1*np.sign(dev_f1)*grad_expected_return + self.lambda2*np.sign(dev_f2)*grad_variance

	# Define the gradient of the objective function
	def compute_gradient(self, x):
		# Gradient of expected return (negative of mu)
		grad_expected_return = -self.mu
		
		# Gradient of the variance term (2 * cov_matrix @ x)
		grad_variance = 2 * self.cov_matrix @ x
		
		# Combined gradient (weighted sum)
		grad = self.lambda1 * grad_expected_return + self.lambda2 * grad_variance
		return grad

	# Project the portfolio weights onto the feasible set
	# def project(self, x: np.ndarray) -> np.ndarray:
	#     # Project weights onto the feasible set (no short-selling and sum to 1)
	#     # Clip values to make sure they are between 0 and 1
	#     x_proj = np.clip(x, 0, 1)
	    
	#     # Normalize to ensure the weights sum to 1 (budget constraint)
	#     x_proj /= np.sum(x_proj)
	    
	#     return x_proj

	# Project the portfolio weights onto the feasible set
	def project(self, x: np.ndarray) -> np.ndarray:
	    """
	    Duchi's Euclidean projection of x onto the simplex:
	        { w >= 0, sum(w)=1 }.
	    """
	    # sort x in descending order
	    u = np.sort(x)[::-1]
	    # compute cumulative sums of u
	    cssv = np.cumsum(u)
	    # find rho = the last index where u[i] > (cssv[i] - z)/(i+1)
	    rho = np.nonzero(u > (cssv - 1) / (np.arange(1, len(x)+1)))[0][-1]
	    # compute the threshold theta
	    theta = (cssv[rho] - 1) / (rho + 1)
	    # project
	    w = np.maximum(x - theta, 0)
	    return w

	# Projected Gradient Descent Algorithm
	def run(self) -> tuple:
		# Initialize the portfolio weights
		x = np.ones(len(self.mu)) / len(self.mu)  # Start with equal weights
		
		before_stamp = time.time()
		# Iteratively update portfolio weights
		for iter in range(self.max_iter):

			if not self.chebyshev:
				# Compute the weighted sum gradient of the objective function
				grad = self.compute_gradient(x)
			else:
				# Compute the chebyshev gradient of the objective function
				grad = self.compute_chebyshev_gradient(x)
			# Update the portfolio weights using gradient descent
			x_new = x - self.learning_rate * grad
			
			# Project the updated weights back onto the feasible set
			x_new = self.project(x_new)
			
			# Check for convergence (if the change in weights is small enough)
			if np.linalg.norm(x_new - x) < self.epsilon:
				print(f"Converged after {iter+1} iterations")
				break
			
			# Update x for the next iteration
			x = x_new

		timer = time.time() - before_stamp
		
		return x.tolist(), timer