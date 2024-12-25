import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import typing
from  torch import optim
import copy
from autograd.extend import primitive, defvjp


torch.set_default_dtype(torch.float64)

@primitive
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
    W = W * W
    M = torch.eye(d) + W * W/d
    E = torch.matrix_power(M, d - 1)
    return torch.sum(E.T * M) - d


def ancestrality_loss(W1: torch.tensor, W2: torch.tensor):
    """
    Compute the loss due to violations of ancestrality in the induced ADMG of W1, W2.

    :param W1: numpy matrix for directed edge coefficients.
    :param W2: numpy matrix for bidirected edge coefficients.
    :return: float corresponding to penalty on violations of ancestrality.
    """
    d = len(W1)
    W1_pos = W1*W1
    W2_pos = W2*W2
    W1k = torch.eye(d)
    M = torch.eye(d)
    for k in range(1, d):
        W1k = W1k@W1_pos
        # M += comb(d, k) * (1 ** k) * W1k (typical binoimial)
        M += 1.0/math.factorial(k) * W1k #(special scaling)

    return torch.sum(M*W2_pos)


def structure_penalty(W1: torch.tensor, W2: torch.tensor, admg_class):

    if admg_class == "ancestral":
        penalty = ancestrality_loss
    # elif admg_class == "arid":
    #     penalty = reachable_loss
    else:
        raise NotImplemented("Invalid ADMG class")
    structure_penalty = cycle_loss(W1) + 0.0001*penalty(W1, W2)
    return structure_penalty


def SPDLogCholesky(M: torch.tensor):
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


# def ancestrality_loss(W1: torch.tensor, W2: torch.tensor):
#     """
#     Compute the loss due to violations of ancestrality in the induced ADMG of W1, W2.

#     :param W1: numpy matrix for directed edge coefficients.
#     :param W2: numpy matrix for bidirected edge coefficients.
#     :return: float corresponding to penalty on violations of ancestrality.
#     """

#     d = len(W1)
#     W1_pos = W1*W1
#     W2_pos = W2*W2
#     W1k = torch.eye(d)
#     M = torch.eye(d)
#     for k in range(1, d):
#         W1k = W1k@W1_pos
#         # M += comb(d, k) * (1 ** k) * W1k (typical binoimial)
#         M += 1.0/math.factorial(k) * W1k #(special scaling)

#     return torch.sum(M*W2_pos)


class RKHSDagma(nn.Module):
    """n: number of samples, d: num variables"""
    def __init__(self, x: torch.tensor, gamma = 1):
        super(RKHSDagma, self).__init__() # inherit  nn.Module
        self.x = x
        self.d = x.shape[1]
        self.n = x.shape[0]
        self.gamma = gamma
        # initialize coefficients alpha
        alpha = torch.zeros(self.d, self.n)
        self.alpha = nn.Parameter(alpha) 
        # initialize coefficients beta
        self.beta = nn.Parameter(torch.zeros(self.d, self.d, self.n))
        
        self.delta = torch.ones(self.d, self.d)
        self.delta.fill_diagonal_(0)
        self.I = torch.eye(self.d)

        Sigma = torch.eye(self.d)

        self.M = reverse_SPDLogCholesky(self.d, Sigma)
        self.M = nn.Parameter(self.M)
    

        # x: [n, d]; K: [n, n]; grad_K1: [n, n, d]: gradient of k(x^i, x^l) wrt x^i_{k}; grad_K2: [n, n, d]: gradient of k(x^i, x^l) wrt x^l_{k}; mixed_grad: [n, n, d, d] gradient of k(x^i, x^l) wrt x^i_{a} and x^l_{b}
        
        # Compute pairwise squared Euclidean distances using broadcasting
        self.diff = self.x.unsqueeze(1) - self.x.unsqueeze(0) # [n, n, d]
        self.sq_dist = torch.einsum('jk, ilk -> jil', self.delta, self.diff**2) # [d, n, n]

        # Compute the Gaussian kernel matrix
        self.K = torch.exp(-self.sq_dist / (self.gamma ** 2)) # [d, n, n] K[j, i, l] = k(x^i, x^l) without jth coordinate
        
        # Compute the gradient of K with respect to x
        self.grad_K1 = -2 / (self.gamma ** 2) * torch.einsum('jil, ila -> jila', self.K, self.diff) # [d, n, n, d] 
        self.identity_mask = torch.eye(self.d, dtype=torch.bool)
        self.broadcastable_mask = self.identity_mask.view(self.d, 1, 1, self.d)
        self.expanded_mask = self.broadcastable_mask.expand(-1, self.n, self.n, -1)
        self.grad_K1[self.expanded_mask] = 0.0
        self.grad_K2 = -self.grad_K1

        self.outer_products_diff = torch.einsum('ila, ilb->ilab', self.diff, self.diff)  # Outer product of the differences [n, n, d, d]
        self.mixed_grad = (-4 / self.gamma**4) * torch.einsum('jil, ilab -> jilab', self.K, self.outer_products_diff) # Apply the formula for a != b [d, n, n, d, d]

        # Diagonal elements (i == j) need an additional term
        self.K_expanded = torch.einsum('jil,ab->jilab', self.K, self.I) #[d, n, n, d, d]
        self.mixed_grad += 2/self.gamma**2 * self.K_expanded

        self.expanded_identity_mask1 = self.identity_mask.view(self.d, 1, 1, 1, self.d).expand(self.d, self.n, self.n, self.d, self.d)
        self.expanded_identity_mask2 = self.identity_mask.view(self.d, 1, 1, self.d, 1).expand(self.d, self.n, self.n, self.d, self.d)

        # # Zero out elements in A where the mask is True
        self.mixed_grad[self.expanded_identity_mask1] = 0
        self.mixed_grad[self.expanded_identity_mask2] = 0
    
    def get_parameters(self): # [d, n]
       alpha = self.alpha
       beta = self.beta
       return alpha, beta
    
    def forward(self): #[n, d] -> [n, d], forward(x)_{i,j} = estimation of x_j at ith observation 
      """
      x: data matrix of shape [n, d] (np.array)
      forward(x)_{l,j} = estimation of x_j at lth observation
      """
      output1 = torch.einsum('jl, jil -> ij', self.alpha, self.K) # [n, d]
      output2 = torch.einsum('jal, jila -> ijl', self.beta, self.grad_K2) # [n, d, n]
      output2 = torch.sum(output2, dim = 2) # [n, d]
      output = output1 + output2 # [n, d]
      Sigma = SPDLogCholesky(self.M)
      return output, Sigma


    def fc1_to_adj(self) -> torch.Tensor: # [d, d]
      """
      return the weighted adjacency matrix
      """
      weight1 = torch.einsum('jl, jilk -> kij', self.alpha, self.grad_K1) # [d, n, d]
      weight2 = torch.einsum('jal, jilka -> kij', self.beta, self.mixed_grad) # [d, n, d]
      weight = weight1 + weight2
      weight = torch.sum(weight ** 2, dim = 1)/self.n # [d, d]
      help = torch.tensor(1e-16) # numerical stability
      W_sqrt = torch.sqrt(weight+help)

      _, Sigma = self.forward()
      Wii = torch.diag(torch.diag(Sigma))
      W2 = Sigma - Wii

      return W_sqrt, W2

    
    # # log determinant h
    # def h_func(self, W1: torch.tensor, W2: torch.tensor, s: torch.tensor):
    # #   weight = weight * weight # not reasonable but performs better
    # #   s = torch.tensor(s, dtype=torch.float64)
    # #   A = s*self.I - weight*weight
    # #   sign, logabsdet = torch.linalg.slogdet(A)
    # #   h = -logabsdet + self.d * torch.log(s)
    # #   return h

    #     d = len(W1)
    #     W1 = W1 * W1
    #     M = torch.eye(d) + W1 * W1/d # not reasonable but performs better, even than h_logdet(weight)
    #     E = torch.matrix_power(M, d - 1)
    #     return torch.sum(E.T * M) - d + 0.0001*ancestrality_loss(W1, W2)
    

    def mse(self, x, x_est: torch.tensor): # [1, 1]
      squared_loss = 0.5 / self.n * torch.sum((x_est - x) ** 2)
      return squared_loss
    

    def mle_loss(self, x, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        # mle = torch.trace((self.x - x_est)@torch.linalg.inv(self.Sigma)@((self.x - x_est).T)) # Check if Sigma invertible !!!!, aslo divide by n
        # Sigma = self.I
        tmp = torch.linalg.solve(Sigma, (x - x_est).T)
        mle = torch.trace((x - x_est)@tmp)/self.n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle
    
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
    

class RKHSDagma_nonlinear:
    """
    Class that implements the DAGMA algorithm
    """

    def __init__(self, x, model: nn.Module, admg_class, verbose: bool = False, dtype: torch.dtype = torch.float64):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.float64``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.admg_class = admg_class
        self.dtype = dtype
        self.x = x

        
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
        ) -> bool:
        r"""
        Solves the optimization problem: 
            .. math::
                \arg\min_{W(\Theta) \in \mathbb{W}^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
        where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
        from the model parameters. 
        This problem is solved via (sub)gradient descent using adam acceleration.

        Parameters
        ----------
        max_iter : float
            Maximum number of (sub)gradient iterations.
        lr : float
            Learning rate.
        lambda1 : float
            function complexity penalty coefficient. 
        tau : float
            sparsity and function complexity penalty coefficient.
        lambda2 : float
            L2 penalty coefficient. Applies to all the model parameters.
        mu : float
            Weights the score function.
        s : float
            Controls the domain of M-matrices.
        lr_decay : float, optional
            If ``True``, an exponential decay scheduling is used. By default ``False``.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        bool
            ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix 
            got outside of the domain of M-matrices.
        """
        self.vprint(f'\nMinimize s={s} -- lr={lr}')

        # optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        # if lr_decay is True:
        #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        # obj_prev = 1e16
        # for i in range(max_iter):
        #     optimizer.zero_grad()
        #     weight = self.model.fc1_to_adj()
        #     h_val = self.model.h_func(weight, s)
        #     if h_val.item() < 0:
        #         self.vprint(f'Found h negative {h_val.item()} at iter {i}')
        #         return False
        #     x_est_prior, Sigma_prior = self.model.forward()
        #     # squared_loss_prior = self.model.mse(x_est_prior)
        #     mle_loss_prior = self.model.mle_loss(x_est_prior, Sigma_prior)
        #     complexity_reg = self.model.complexity_reg(lambda1, tau)
        #     sparsity_reg = self.model.sparsity_reg(weight, tau) 
        #     score = mle_loss_prior + complexity_reg + sparsity_reg
        #     obj = mu * score + h_val
        #     #print("obj: ", obj)
        #     obj.backward()
        #     optimizer.step()


        optimizer_alpha_beta = optim.Adam([self.model.alpha, self.model.beta], lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        optimizer_Sigma = optim.Adam([self.model.M], lr=0.003, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler_alpha_beta = optim.lr_scheduler.ExponentialLR(optimizer_alpha_beta, gamma=0.8)
            scheduler_Sigma = optim.lr_scheduler.ExponentialLR(optimizer_Sigma, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer_alpha_beta.zero_grad()
            weight, W2 = self.model.fc1_to_adj()
            h_val = cycle_loss(weight, s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            x_est_prior, Sigma_prior = self.model.forward()
            # squared_loss_prior = self.model.mse(x_est_prior)
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(weight, tau) 
            cov_regularizer = torch.sqrt(torch.sum((torch.relu(-4 - Sigma_prior) + torch.relu(Sigma_prior - 4))**2))
            score = mle_loss_prior + complexity_reg + sparsity_reg 
            penalty = structure_penalty(weight, W2, self.admg_class)
            obj = mu * score + penalty + cov_regularizer
            #print("obj: ", obj)
            obj.backward()
            optimizer_alpha_beta.step()

            optimizer_Sigma.zero_grad()
            weight, W2 = self.model.fc1_to_adj()
            h_val = cycle_loss(weight, s=1)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            x_est_prior, Sigma_prior = self.model.forward()
            # squared_loss_prior = self.model.mse(x_est_prior)
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(weight, tau) 
            cov_regularizer = torch.sqrt(torch.sum((torch.relu(-4 - Sigma_prior) + torch.relu(Sigma_prior - 4))**2))
            score = mle_loss_prior + complexity_reg + sparsity_reg
            penalty = structure_penalty(weight, W2, self.admg_class)
            obj = mu * score + penalty + cov_regularizer
            obj.backward()
            optimizer_Sigma.step()

            x_est_posterior, Sigma_posterior = self.model.forward()
            # squared_loss_posterior = self.model.mse(x_est_posterior)
            mle_loss_posterior = self.model.mle_loss(self.x, x_est_posterior, Sigma_posterior)
            diff = torch.abs(mle_loss_prior - mle_loss_posterior)
            if diff < 1e-12 and h_val < 1e-9:
                break
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler_alpha_beta.step()
                scheduler_Sigma.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                self.vprint(f'\tmle_loss: {mle_loss_posterior}')
                self.vprint(f'\tweight: {weight}')
                self.vprint(f'\tSigma: {Sigma_posterior}')
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
        torch.set_default_dtype(self.dtype)
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
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, tau, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar)
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
        W_est, _ = self.model.fc1_to_adj()
        W_est = W_est.cpu().detach().numpy()
        W_est[np.abs(W_est) < w_threshold] = 0
        output, Sigma = self.model.forward()
        return W_est, output, Sigma
    
if __name__ == "__main__":

    np.random.seed(0)
    n_samples = 300  # Number of samples
    dim = 2  # Dimension of the normal vectors
    True_Sigma = np.array([[1, 0.6],    # Variance of X is 1, covariance between X and Y is 0.8
                  [0.6, 1.5]]) 
    
    # x = np.random.uniform(low=-3, high=3, size=100)
    # x = np.random.normal(0,1, 100) 
    # epsilon2 = np.random.normal(0,1, 100) 
    epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples) #[n, d]
    # # epsilon = torch.tensor(epsilon, dtype=torch.float64)
    x = epsilon[:, 0]
    epsilon2 = epsilon[:, 1]
    y_without_noise = np.sin(x)*10
    y = np.array([np.sin(x)*10 + epsilon2 for x, epsilon2 in zip(x, epsilon2)])
    X = np.column_stack((x, y))
    X = torch.from_numpy(X)



    eq_model = RKHSDagma(X, gamma = 1)
    model = RKHSDagma_nonlinear(x=X, model=eq_model, admg_class = "ancestral", verbose=True)
    W_est_no_thresh, output, Sigma = model.fit(lambda1=1e-3, tau=1e-4, T = 6, mu_init = 0.1, lr=0.03, w_threshold=0.0)
    print("W_est_no_thresh: ", W_est_no_thresh)

    y_hat = output[:, 1].cpu().detach().numpy()
    covariance = np.cov(y - y_hat, x)
    print("cov: ", covariance[0, 1])
    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(x, y, label='y', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(x, y_hat, label='y_est', color='red', marker='s') 
    plt.scatter(x, y_without_noise, label='y_without_noise', color='green', marker='s') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X.detach().numpy()[:, 1], output.detach().numpy()[:, 0], label='x_est', color='red', marker='s') 
    plt.scatter(X.detach().numpy()[:, 1], X.detach().numpy()[:, 0], label='x', color='green', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")