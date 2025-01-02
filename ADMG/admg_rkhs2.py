import torch
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import typing
from tqdm.auto import tqdm
import copy

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
    Sigma_init = cov + cov.T + torch.eye(d)#torch.tensor([[0.1, 0], [0,1]])
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

def cycle_loss(W: torch.tensor, s=1):
    """
    Compute the loss, h_acyc, due to directed cycles in the induced graph of W.

    :param W: numpy matrix.
    :return: float corresponding to penalty on directed cycles.
    Use trick when computing the trace of a product
    """
    # d = W.size(0)
    # s = torch.tensor(s)
    # A = s*torch.eye(d) - W*W
    # sign, logabsdet = torch.linalg.slogdet(A)
    # h = -logabsdet + d * torch.log(s)
    # return h

    d = len(W)
    M = torch.eye(d) + W * W/d
    E = torch.matrix_power(M, d - 1)
    return torch.sum(E.T * M) - d

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
        # Sigma = torch.eye(self.d)

        # self.M = reverse_SPDLogCholesky(self.d, Sigma)
        # self.M = nn.Parameter(self.M)

        self.L = nn.Parameter(torch.ones(self.d))
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
        # Sigma = SPDLogCholesky(self.M, d=self.d)
        Sigma = torch.diag(self.L)
        output1 = torch.einsum('jl, jil -> ij', self.alpha, self.K) # [n, d]
        output2 = torch.einsum('jal, jila -> ijl', self.beta, self.grad_K2) # [n, d, n]
        output2 = torch.sum(output2, dim = 2) # [n, d]
        output = output1 + output2 
        # output[:, 0] = 0
        return Sigma, output
    
    def fc1_to_adj(self) -> torch.Tensor: # [d, d]
        """
        return the directed weighted adjacency matrix W1
        """
        weight1 = torch.einsum('jl, jilk -> kij', self.alpha, self.grad_K1) # [d, n, d]
        weight2 = torch.einsum('jal, jilka -> kij', self.beta, self.mixed_grad) # [d, n, d]
        weight = weight1 + weight2
        weight = torch.sum(weight ** 2, dim = 1)/self.n # [d, d]
        help = torch.tensor(1e-16)
        W1 = torch.sqrt(weight+help)
        return W1

    
    def mle_loss(self, x, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        tmp = torch.linalg.solve(Sigma, (x - x_est).T)
        mle = torch.trace((x - x_est)@tmp)/self.n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle
    
    def mse(self, x, x_est: torch.tensor): # [1, 1]
      squared_loss = 0.5 / self.n * torch.sum((x_est - x) ** 2)
      return squared_loss

    def complexity_reg(self, lambda1, tau):
        """
        parameter:
        tau: penalty for sparsity termn and function complexity term together
        lambda1: addtional penalty for function complexity term 

        return: function complexity penalty
        """
        temp1 = torch.einsum('ji, jil -> jl', self.alpha, self.K) #[d, n]
        temp1 = (self.alpha*temp1).sum() 
        temp2 = torch.einsum('jal, jila -> ji', self.beta, self.grad_K2) #[d, n]
        temp2 = (self.alpha * temp2).sum()
        temp3 = torch.einsum('jbl, jilab -> jai', self.beta, self.mixed_grad) #[d, d, n]
        temp3 = (self.beta * temp3).sum()
        regularized = lambda1*tau*(temp1 + temp2 + temp3)
        return regularized
    
    def sparsity_reg(self, weight: torch.tensor, tau):
        """
        weight: weighted adjacency matrix

        return: sparsity penalty 
        """
        sparsity = torch.sum(weight)
        return 2*tau*sparsity
    
class Sigma_discovery:

    def __init__(self, x, model: nn.Module, admg_class, verbose: bool = False, dtype: torch.dtype = torch.float64, lambda1=1e-3, tau=1e-4):
        self.model = model
        self.x = x
        self.vprint = print if verbose else lambda *a, **k: None
    # def minimize(self, lr: torch.float64 = .03, lambda2: torch.float64 = .005, max_iter=2500, lambda1=0.5e-3, tau=1):
    def minimize(self, 
            max_iter: float, 
            lr: float, 
            lambda1: float, 
            tau: float,
            lambda2: float, 
            mu: float, 
            s: float,
            lr_decay: float = False, 
            tol: float = 1e-12, 
            pbar: typing.Optional[tqdm] = None,
            t = None
    ): 
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        # optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=lambda2)
        optimizer_alpha_beta = optim.Adam([self.model.alpha, self.model.beta], lr=lr, betas=(.99,.999), weight_decay=lambda2)
        optimizer_Sigma = optim.Adam([self.model.L], lr=0.003, betas=(.99,.999), weight_decay=lambda2)

        obj_prev = 1e16####
        for i in range(max_iter):
            optimizer_alpha_beta.zero_grad()
            W1 = self.model.fc1_to_adj()
            h_val = cycle_loss(W1, s=1)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            Sigma_prior, x_est_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            mse_loss_prior = self.model.mse(self.x, x_est_prior)
            residual_norm = torch.norm(self.x - x_est_prior, dim=1).mean()
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(W1, tau)
            score = mle_loss_prior + sparsity_reg + complexity_reg 
            obj = mu * score + h_val
            obj.backward()
            optimizer_alpha_beta.step()

            optimizer_Sigma.zero_grad()
            W1 = self.model.fc1_to_adj()
            h_val = cycle_loss(W1, s=1)
            Sigma_prior, x_est_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            mse_loss_prior = self.model.mse(self.x, x_est_prior)
            residual_norm = torch.norm(self.x - x_est_prior, dim=1).mean()
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(W1, tau)

            score = mle_loss_prior + sparsity_reg + complexity_reg 
            obj = mu * score + h_val
            obj.backward()
            optimizer_Sigma.step()
            
            Sigma_posterior, x_est_posterior = self.model.forward()
            mle_loss_posterior = self.model.mle_loss(self.x, x_est_posterior, Sigma_posterior)
            mse_loss_posterior = self.model.mse(self.x, x_est_posterior)
            diff = torch.abs(mle_loss_prior- mle_loss_posterior)
            # if diff < 1e-12:
            #     break
                
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                self.vprint(f"Step {i}: mle = {mle_loss_posterior.item()}")
                self.vprint(f"Step {i}: mse = {mse_loss_posterior.item()}")
                self.vprint(f"Step {i}: Sigma = {Sigma_posterior}")
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True
    
    def fit(self, 
        lambda1: torch.float64 = .02, 
        tau: torch.float64 = .02,
        lambda2: torch.float64 = .005,
        T: torch.int = 4, 
        mu_init: torch.float64 = 0.1, 
        mu_factor: torch.float64 = .5, 
        s: torch.float64 = 1.0,
        warm_iter: torch.int = 5e3, 
        max_iter: torch.int = 8e3, 
        lr: torch.float64 = .005, 
        w_threshold: torch.float64 = 0.3, 
        checkpoint: torch.int = 1000,
    ) -> np.ndarray:

        r"""
        Runs the DAGMA algorithm and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the function penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, by default .005.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : float, optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t < T`, by default 5e3.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t = T`, by default 8e3.
        lr : float, optional
            Learning rate, by default .0002.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        """
        # torch.set_default_dtype(self.dtype)
        # self.x = torch.tensor(data.values, dtype=torch.float64)
        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 

        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                # self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                print(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                # model_copy = self.model.__class__(data=self.x)
                model_copy.load_state_dict(self.model.state_dict())
                lr_decay = False
                while success is False:
                    # print("success: ", success)
                    success = self.minimize(inner_iter, lr, lambda1, tau, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar, t =i)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy()) # restore the model parameters to last iteration
                        # reset lr, lr_decay, s_cur then update the model
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        final_W1 = self.model.fc1_to_adj()
        final_W2, output = self.model.forward()
        print("final_W1: ", final_W1)
        print("final_W2: ", final_W2)
        final_W1 = final_W1.cpu().detach().numpy()
        final_W2 = final_W2.cpu().detach().numpy()
        # final_W1[np.abs(final_W1) < w_threshold] = 0
        # final_W2[np.abs(final_W2) < w_threshold] = 0
        #return get_graph(final_W1, final_W2, data.columns, w_threshold)
        return final_W1, final_W2, output
    
    
if __name__ == "__main__":

    # #  # Sample data generation: let's assume X and X_hat are from some known distributions for the sake of example
    # n_samples = 300  # Number of samples
    # dim = 2  # Dimension of the normal vectors

    # # Random data for X and X_hat
    # True_Sigma = np.array([[0.5, 0.3],    # Variance of X is 1, covariance between X and Y is 0.8
    #               [0.3, 1.5]])   # Variance of Y is 1, covariance between Y and X is 0.8
    # epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples) #[n, d]
    # epsilon = torch.tensor(epsilon, dtype=torch.float64)
    # x1 = torch.randn(n_samples)
    # # x1 = torch.zeros(n_samples)
    # x2 = 10*torch.sin(x1)
    # # Step 4: Combine these results into a new tensor of shape [n, 2]
    # x1_true = x1 + epsilon[:, 0]
    # x2_true = x2 + epsilon[:, 1]
    # X = torch.stack((x1, x2), dim=1)
    # X_true = torch.stack((x1_true, x2_true), dim=1)
    # eq_model2 = Sigma_RKHSDagma(X, gamma = 1)
    # model2 = Sigma_discovery(x=X_true, model=eq_model2, admg_class = "ancestral", verbose=True)
    # x_est, Sigma = model2.fit()
    # y_hat = x_est[:, 1].detach().numpy()
    # empirical_covariance = np.cov(epsilon, rowvar=False)
    # print("Empirical Covariance Matrix:", empirical_covariance)
    # print("estimated Sigma: ", Sigma)

    # plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    # plt.scatter(X.detach().numpy()[:, 0], X_true.detach().numpy()[:, 1], label='y', color='blue', marker='o')  # Plot x vs. y1
    # plt.scatter(X.detach().numpy()[:, 0], x_est.detach().numpy()[:, 1], label='y_est', color='red', marker='s') 
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
    # print("The programm is closed")




    # multiprocessing.set_start_method('spawn')
    # # Sample data generation: let's assume X and X_hat are from some known distributions for the sake of example
    n_samples = 300  # Number of samples
    dim = 2  # Dimension of the normal vectors

    # Random data for X and X_hat
    True_Sigma = np.array([[1, 0.0],    # Variance of X is 1, covariance between X and Y is 0.8
                  [0.0, 2]])   # Variance of Y is 1, covariance between Y and X is 0.8
    epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples) #[n, d]
    epsilon = torch.tensor(epsilon, dtype=torch.float64)
    x1 = epsilon[:, 0]
    # x1 = torch.randn(n_samples)
    x1_true = epsilon[:, 0]
    x2 = 20*torch.sin(x1)
    # Step 4: Combine these results into a new tensor of shape [n, 2]
    X = torch.stack((x1, x2), dim=1)
    x2_true = 20*torch.sin(x1)+ epsilon[:, 1]
    X_true = torch.stack((x1_true, x2_true), dim=1)
    eq_model2 = Sigma_RKHSDagma(X_true, gamma = 1)
    model2 = Sigma_discovery(x=X_true, model=eq_model2, admg_class = "ancestral", verbose=True)
    final_W1, Sigma, x_est = model2.fit()
    # print("X_est: ", x_est)
    y_hat = x_est[:, 1].detach().numpy()
    empirical_covariance = np.cov(epsilon, rowvar=False)
    print("Empirical Covariance Matrix:", empirical_covariance)
    print("estimated Sigma: ", Sigma)
    # estimated_covariance = np.cov(x1.detach().numpy(), (x2_true-x_est[:, 1]).detach().numpy())
    # print("estimated Covariance Matrix:", estimated_covariance[0, 1])

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X.detach().numpy()[:, 0], X.detach().numpy()[:, 1], label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(X.detach().numpy()[:, 0], x_est.detach().numpy()[:, 1], label='y_est', color='red', marker='s') 
    plt.scatter(X.detach().numpy()[:, 0], X_true.detach().numpy()[:, 1], label='y_noise', color='green', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X.detach().numpy()[:, 1], x_est.detach().numpy()[:, 0], label='x_est', color='red', marker='s') 
    plt.scatter(X.detach().numpy()[:, 1], X.detach().numpy()[:, 0], label='x', color='green', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")