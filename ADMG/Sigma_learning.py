import torch
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn
import pandas as pd

torch.set_default_dtype(torch.float64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


def SPDLogCholesky(M: torch.tensor, d):
    """
    Use LogCholesky decomposition that map a matrix M to a SPD Sigma matrix.
    """
    # Take strictly lower triangular matrix
    M_strict = M.tril(diagonal=-1)
    # Make matrix with exponentiated diagonal
    D = M.diag()
    # Make the Cholesky decomposition matrix
    L = M_strict + torch.diag(torch.exp(D))
    # Invert the Cholesky decomposition
    Sigma = torch.matmul(L, L.t()) #+ 1e-6 * torch.eye(d) # numerical stability
    return Sigma


def reverse_SPDLogCholesky(d, Sigma: torch.tensor):
    """
    Reverse the LogCholesky decomposition that map the SPD Sigma matrix to the matrix M.
    """
    # Compute the Cholesky decomposition
    cov = torch.rand(d, d) * 0.1 - 0.05
    # print("cov: ", cov)
    # tril_indices = torch.tril_indices(d, d)
    # cov = cov[tril_indices[0], tril_indices[1]]
    cov = torch.triu(cov, diagonal=1)
    # print("cov: ", cov)
    # Sigma = torch.eye(d)
    # Sigma_init = cov + cov.T + torch.diag(Sigma.diag())
    # Sigma_init = torch.diag(Sigma.diag())
    Sigma_init = cov + cov.T + torch.eye(d)
    # Sigma_init = torch.tensor([[1, 0, 0, 0],    # Variance of X is 1, covariance between X and Y is 0.8
    #             [0, 1, 0, 0],
    #             [0, 0, 1, 0.6],
    #             [0, 0, 0.6, 1]])
    L = torch.linalg.cholesky(Sigma_init)
    # Take strictly lower triangular matrix
    M_strict = L.tril(diagonal=-1)
    # Take the logarithm of the diagonal
    D = torch.diag(torch.log(L.diag()))
    # Return the log-Cholesky parametrization
    M = M_strict + D
    return M

class Sigma_RKHSDagma(nn.Module):

    def __init__(self, data, gamma = 1):
        super(Sigma_RKHSDagma, self).__init__() # inherit nn.Module
        if isinstance(data, pd.DataFrame):
            self.x = torch.tensor(data.values, dtype=torch.float64).to(device)
        elif isinstance(data, torch.Tensor):
            self.x = data.to(device) # data matrix [n, d]  
        else:
            raise ValueError("Input data must be a pandas DataFrame or a torch.Tensor")
        self.d = self.x.shape[1]
        self.n = self.x.shape[0]
        Sigma = torch.eye(self.d)
        # self.Sigma = torch.tensor([[1, 0.6],    # Variance of X is 1, covariance between X and Y is 0.8
        #           [0.6, 1]], dtype=torch.float64)
        self.M = reverse_SPDLogCholesky(self.d, Sigma)
        self.M = nn.Parameter(self.M)
        alpha = torch.zeros(self.d, self.d)
        self.alpha = nn.Parameter(alpha)
        # self.alpha = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)

    def forward(self):
        Sigma = SPDLogCholesky(self.M, d=self.d)
        output = self.x @ self.alpha
        return Sigma, output
    
    def mle_loss(self, x, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        tmp = torch.linalg.solve(Sigma, (x - x_est).T)
        mle = torch.trace((x - x_est)@tmp)/self.n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle
    
class Sigma_discovery:

    def __init__(self, x, model: nn.Module, admg_class, verbose: bool = False, dtype: torch.dtype = torch.float64):
        self.model = model
        self.x = x
    def fit(self, lr: torch.float64 = .005, lambda2: torch.float64 = .005, max_iter=2500):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=lambda2)
        for i in range(max_iter):
            optimizer.zero_grad()
            Sigma_prior, x_est_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            obj = mle_loss_prior
            obj.backward()
            optimizer.step()
            Sigma_posterior, x_est_posterior = self.model.forward()
            mle_loss_posterior = self.model.mle_loss(self.x, x_est_posterior, Sigma_posterior)
            diff = torch.abs(mle_loss_prior - mle_loss_posterior)
            if diff < 1e-12:
                break
            if i % 100 == 0:
                print(f"Step {i}: mle = {mle_loss_posterior.item()}")
                print(f"Step {i}: Sigma = {Sigma_posterior}")
        return self.model.alpha, Sigma_posterior
    
if __name__ == "__main__":
    # Sample data generation: let's assume X and X_hat are from some known distributions for the sake of example
    n_samples = 200  # Number of samples
    dim = 2  # Dimension of the normal vectors

    # Random data for X and X_hat
    True_Sigma = np.array([[1, 0.6],    # Variance of X is 1, covariance between X and Y is 0.8
                  [0.6, 1]])   # Variance of Y is 1, covariance between Y and X is 0.8
    alpha = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples)
    epsilon = torch.tensor(epsilon, dtype=torch.float64)
    X_hat = torch.randn(n_samples, dim)
    X = X_hat @ alpha + epsilon
    eq_model2 = Sigma_RKHSDagma(X_hat, gamma = 1)
    model2 = Sigma_discovery(x=X, model=eq_model2, admg_class = "ancestral", verbose=True)
    alpha, Sigma = model2.fit()
    empirical_covariance = np.cov(epsilon, rowvar=False)
    print("Empirical Covariance Matrix:", empirical_covariance)
    print("estimated alpha: ", alpha)
    print("estimated Sigma: ", Sigma)
    
