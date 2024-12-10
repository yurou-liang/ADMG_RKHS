import torch
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

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
        self.gamma = gamma
        # initialize coefficients alpha
        alpha = torch.zeros(self.d, self.n)
        # alpha = torch.rand(self.d, self.n)
        self.alpha = nn.Parameter(alpha) 
        # initialize coefficients beta
        self.beta = nn.Parameter(torch.zeros(self.d, self.d, self.n))
        # self.beta = torch.zeros(self.d, self.d, self.n)
        #self.beta = nn.Parameter(torch.rand(self.d, self.d, self.n))
        # initialize the symmetric bidirected adjacency matrix with 0 on the diagonal, entries are uniformly picked from [-0.1, 0.1]
        self.I = torch.eye(self.d)
        # self.L = torch.rand(self.d, self.d) * 0.1 - 0.1
        # Sigma = torch.cov(self.x.T)
        Sigma = torch.eye(self.d)

        self.M = reverse_SPDLogCholesky(self.d, Sigma)
        self.M = nn.Parameter(self.M)
        # self.L = nn.Parameter(self.L)
        # self.Sigma = self.L @ self.L.T + 1e-6*self.I
        # self.Wii = torch.diag(torch.diag(self.Sigma))
        # self.W2 = self.Sigma - self.Wii
        # x: [n, d]; K: [d, n, n]: K[j, i, l] = k(x^i, x^l) without jth coordinate; grad_K1: [n, n, d]: gradient of k(x^i, x^l) wrt x^i_{k}; 
        # grad_K2: [n, n, d]: gradient of k(x^i, x^l) wrt x^l_{k}; mixed_grad: [n, n, d, d] gradient of k(x^i, x^l) wrt x^i_{a} and x^l_{b}

        self.omega = torch.ones(self.d, self.d)
        self.omega.fill_diagonal_(0)
        # Compute pairwise squared Euclidean distances using broadcasting
        self.diff = self.x.unsqueeze(1) - self.x.unsqueeze(0) # [n, n, d]
        self.sq_dist = torch.einsum('jk, ilk -> jil', self.omega, self.diff**2) # [d, n, n]

        # Compute the Gaussian kernel matrix
        self.K = torch.exp(-self.sq_dist / (self.gamma ** 2)) 

        # Compute the gradient of K wrt x
        self.grad_K1 = -2 / (self.gamma ** 2) * torch.einsum('jil, ila -> jila', self.K, self.diff) # [d, n, n, d] 
        self.identity_mask = torch.eye(self.d, dtype=torch.bool)
        self.broadcastable_mask = self.identity_mask.view(self.d, 1, 1, self.d)
        self.expanded_mask = self.broadcastable_mask.expand(-1, self.n, self.n, -1)
        self.grad_K1[self.expanded_mask] = 0.0
        self.grad_K2 = -self.grad_K1

        # Compute the second order gradient of K wrt x
        self.outer_products_diff = torch.einsum('ila, ilb->ilab', self.diff, self.diff)  # Outer product of the differences [n, n, d, d]
        self.mixed_grad = (-4 / self.gamma**4) * torch.einsum('jil, ilab -> jilab', self.K, self.outer_products_diff) # Apply the formula for a != b [d, n, n, d, d]
        # Diagonal elements (i == j) need an additional term
        self.K_expanded = torch.einsum('jil,ab->jilab', self.K, self.I) #[d, n, n, d, d]
        self.mixed_grad += 2/self.gamma**2 * self.K_expanded
        self.expanded_identity_mask1 = self.identity_mask.view(self.d, 1, 1, 1, self.d).expand(self.d, self.n, self.n, self.d, self.d)
        self.expanded_identity_mask2 = self.identity_mask.view(self.d, 1, 1, self.d, 1).expand(self.d, self.n, self.n, self.d, self.d)
        # Zero out elements in A where the mask is True
        self.mixed_grad[self.expanded_identity_mask1] = 0
        self.mixed_grad[self.expanded_identity_mask2] = 0

    def forward(self):
        Sigma = SPDLogCholesky(self.M, d=self.d)
        output1 = torch.einsum('jl, jil -> ij', self.alpha, self.K) # [n, d]
        output2 = torch.einsum('jal, jila -> ijl', self.beta, self.grad_K2) # [n, d, n]
        output2 = torch.sum(output2, dim = 2) # [n, d]
        output = output1 + output2 
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
    def fit(self, lr: torch.float64 = .05, lambda2: torch.float64 = .005, max_iter=2500):
        # self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=lambda2)
        # optimizer_alpha_beta = optim.Adam([self.model.alpha, self.model.beta], lr=lr, betas=(.99,.999), weight_decay=lambda2)
        # optimizer_Sigma = optim.Adam([self.model.M], lr=0.003, betas=(.99,.999), weight_decay=lambda2)

        obj_prev = 1e16####
        for i in range(max_iter):
            optimizer.zero_grad()
            Sigma_prior, x_est_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            obj = mle_loss_prior
            obj.backward()
            optimizer.step()

            # optimizer_Sigma.zero_grad()
            # Sigma_prior, x_est_prior = self.model.forward()
            # mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            # obj = mle_loss_prior
            # obj.backward()
            # optimizer_Sigma.step()
            
            Sigma_posterior, x_est_posterior = self.model.forward()
            mle_loss_posterior = self.model.mle_loss(self.x, x_est_posterior, Sigma_posterior)
            diff = torch.abs(mle_loss_prior - mle_loss_posterior)
            if diff < 1e-12:
                break
            if i % 100 == 0:
                print(f"Step {i}: mle = {mle_loss_posterior.item()}")
                print(f"Step {i}: Sigma = {Sigma_posterior}")
        return x_est_posterior, Sigma_posterior
    
if __name__ == "__main__":
    # Sample data generation: let's assume X and X_hat are from some known distributions for the sake of example
    n_samples = 200  # Number of samples
    dim = 2  # Dimension of the normal vectors

    # Random data for X and X_hat
    True_Sigma = np.array([[1, 0.6],    # Variance of X is 1, covariance between X and Y is 0.8
                  [0.6, 1]])   # Variance of Y is 1, covariance between Y and X is 0.8
    epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples) #[n, d]
    epsilon = torch.tensor(epsilon, dtype=torch.float64)
    X = torch.randn(n_samples, dim)
    X_inverse = X[:, [1, 0]]
    X_hat = 5*torch.sin(X_inverse)
    X_true = X_hat + epsilon
    eq_model2 = Sigma_RKHSDagma(X, gamma = 1)
    model2 = Sigma_discovery(x=X_true, model=eq_model2, admg_class = "ancestral", verbose=True)
    x_est, Sigma = model2.fit()
    y_hat = x_est[:, 1].detach().numpy()
    empirical_covariance = np.cov(epsilon, rowvar=False)
    print("Empirical Covariance Matrix:", empirical_covariance)
    print("estimated Sigma: ", Sigma)

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X_inverse.detach().numpy()[:, 1], X_true.detach().numpy()[:, 1], label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(X_inverse.detach().numpy()[:, 1], x_est.detach().numpy()[:, 1], label='y_est', color='red', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")
