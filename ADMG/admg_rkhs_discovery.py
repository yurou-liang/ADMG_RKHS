import numpy as np
import math
import autograd.numpy as anp
from autograd import grad
from autograd.extend import primitive, defvjp
import functools
import scipy.optimize as sopt
import pandas as pd
import torch
import torch.nn as nn
from  torch import optim
from ananke.graphs import ADMG
from scipy.special import comb
from tqdm.auto import tqdm
import typing
import copy
import matplotlib.pyplot as plt
from utils.admg2pag import admg_to_pag, pprint_pag
import pandas as pd
import wandb
# from sklearn.preprocessing import StandardScaler

torch.set_default_dtype(torch.float64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
# torch.manual_seed(0)


# h(W) = -logdet(sI-W◦W)+dlogs
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

def bow_loss(W1, W2):
    """
    Compute the loss due to presence of bows in the induced ADMG of W1, W2.

    :param W1: numpy matrix for directed edge coefficients.
    :param W2: numpy matrix for bidirected edge coefficients.
    :return: float corresponding to penalty on bows.
    """
    W1_pos = W1*W1
    W2_pos = W2*W2
    return torch.sum(W1_pos*W2_pos)

def structure_penalty(W1: torch.tensor, W2: torch.tensor, admg_class):

    if admg_class == "ancestral":
        penalty = ancestrality_loss
    # elif admg_class == "arid":
    #     penalty = reachable_loss
    elif admg_class == "bowfree":
        penalty = bow_loss
    elif admg_class == "none":
        penalty = lambda *args, **kwargs: 0
    else:
        raise NotImplemented("Invalid ADMG class")
    structure_penalty = cycle_loss(W1) + 0.0001*penalty(W1, W2)
    return structure_penalty
    
# test with only ancestral loss at first
# consider if bounds can be setted in adma optimizer, only if neccessary
def get_graph(W1: torch.tensor, W2: torch.tensor, vertices, threshold):
    """
    Get the induced ADMG on the matrices W1 and W2.

    :param W1: directed edge coefficients.
    :param W2: bidirected edge coefficients.
    :param vertices: names of vertices in the problem.
    :param threshold: float deciding what is close enough to zero to rule out an edge.
    :return: Ananke ADMG.
    """

    G = ADMG(vertices)
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            if abs(W1[i, j]) > threshold:
                G.add_diedge(vertices[i], vertices[j])
            if i != j and abs(W2[i, j]) > threshold and not G.has_biedge(vertices[i], vertices[j]):
                G.add_biedge(vertices[i], vertices[j])
    return G

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

def reverse_SPDLogCholesky(d):
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

class ADMG_RKHSDagma(nn.Module):
    """
    Class for setting up causal discovery in ADMGs
    """
    def __init__(self, data, gamma = 1):
        super(ADMG_RKHSDagma, self).__init__() # inherit nn.Module
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

        self.M = reverse_SPDLogCholesky(self.d)
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


    def forward(self): #[n, d] -> [n, d]
        """
        forward(x)_{l,j} = estimation of x_j at lth observation
        """
        output1 = torch.einsum('jl, jil -> ij', self.alpha, self.K) # [n, d]
        output2 = torch.einsum('jal, jila -> ijl', self.beta, self.grad_K2) # [n, d, n]
        output2 = torch.sum(output2, dim = 2) # [n, d]
        output = output1 + output2 # [n, d]
        # print("Check M: ", self.M.requires_grad_())
        Sigma = SPDLogCholesky(self.M, d=self.d)
        # print("Sigma: ", Sigma)
        return output, Sigma
    
    def fc1_to_adj(self) -> torch.Tensor: # [d, d]
        """
        return the directed weighted adjacency matrix W1
        """
        weight1 = torch.einsum('jl, jilk -> kij', self.alpha, self.grad_K1) # [d, n, d]
        weight2 = torch.einsum('jal, jilka -> kij', self.beta, self.mixed_grad) # [d, n, d]
        weight = weight1 + weight2
        weight = torch.sum(weight ** 2, dim = 1)/self.n # [d, d]
        help = torch.tensor(1e-8)
        W1 = torch.sqrt(weight+help)

        _, Sigma = self.forward()
        Wii = torch.diag(torch.diag(Sigma))
        W2 = Sigma - Wii
        return W1, W2
    
    def mle_loss(self, x, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        # mle = torch.trace((self.x - x_est)@torch.linalg.inv(self.Sigma)@((self.x - x_est).T)) # Check if Sigma invertible !!!!, aslo divide by n
        # Sigma = self.I
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
    

    
class RKHS_discovery:
    """
    Class that implements the DAGMA algorithm for structure learning in ADMGs.
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
            t = None
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

        # Init Wandb
        # if t == 0:
        #     wandb.init(
        #         project="ADMG",
        #         config={
        #             "learning_rate": lr,
        #             "lambda1": lambda1,
        #             "tau": tau,
        #             "lambda2": lambda2,
        #             "mu": mu,
        #             "s": s
        #         }
        #     )
        self.vprint(f'\nMinimize s={s} -- lr={lr}')

        # # optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        optimizer_alpha_beta = optim.Adam([self.model.alpha, self.model.beta], lr=lr, betas=(.99,.999), weight_decay=lambda2)
        optimizer_Sigma = optim.Adam([self.model.M], lr=0.003, betas=(.99,.999), weight_decay=lambda2)

        if lr_decay is True:
            scheduler_alpha_beta = optim.lr_scheduler.ExponentialLR(optimizer_alpha_beta, gamma=0.8)
            scheduler_Sigma = optim.lr_scheduler.ExponentialLR(optimizer_Sigma, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer_alpha_beta.zero_grad()
            W1, W2 = self.model.fc1_to_adj()
            h_val = cycle_loss(W1, s=1)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            x_est_prior, Sigma_prior = self.model.forward()
            # print("Sigma_prior require_grad: ", Sigma_prior.requires_grad_())
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            # mse_loss_prior = self.model.mse(x_est_prior)
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(W1, tau)
            score = mle_loss_prior #+ sparsity_reg + complexity_reg
            penalty = structure_penalty(W1, W2, self.admg_class)
            cov_regularizer = torch.sqrt(torch.sum((torch.relu(-4 - Sigma_prior) + torch.relu(Sigma_prior - 4))**2))
            # score = mse_loss_prior + complexity_reg + sparsity_reg
            obj = mu * score + penalty #+ mu*cov_regularizer #+ mu*torch.norm(Sigma_prior - torch.eye(Sigma_prior.size(0)) , p='fro') ** 2
            # eax2 = torch.exp((torch.log(torch.tensor(self.model.n, dtype=torch.float64)) * torch.abs(Sigma_prior)))
            # tanh = (eax2 - 1) / (eax2 + 1)
            # obj = mu * score + penalty + torch.sum(tanh) * 0.5
            # print("penalty: ", penalty)
            # print("obj: ", obj)
            obj.backward()
            optimizer_alpha_beta.step()

            optimizer_Sigma.zero_grad()
            W1, W2 = self.model.fc1_to_adj()
            h_val = cycle_loss(W1, s=1)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            x_est_prior, Sigma_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(W1, tau)
            score = mle_loss_prior #+ sparsity_reg + complexity_reg
            penalty = structure_penalty(W1, W2, self.admg_class)
            cov_regularizer = torch.sqrt(torch.sum((torch.relu(-4 - Sigma_prior) + torch.relu(Sigma_prior - 4))**2))
            obj = mu * score + penalty 
            obj.backward()
            optimizer_Sigma.step()

        # optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        # if lr_decay is True:
        #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        # obj_prev = 1e16
        # for i in range(max_iter):
        #     optimizer.zero_grad()
        #     W1, W2 = self.model.fc1_to_adj()
        #     M = self.model.M
        #     # print("M require_grad: ", M.requires_grad_())
        #     h_val = cycle_loss(W1, s=1)
        #     if h_val.item() < 0:
        #         self.vprint(f'Found h negative {h_val.item()} at iter {i}')
        #         return False
        #     penalty = structure_penalty(W1, W2, self.admg_class)
        #     x_est_prior, Sigma_prior = self.model.forward()
        #     # print("Sigma_prior require_grad: ", Sigma_prior.requires_grad_())
        #     mle_loss_prior = self.model.mle_loss(self.x, x_est_prior, Sigma_prior)
        #     # mse_loss_prior = self.model.mse(x_est_prior)
        #     complexity_reg = self.model.complexity_reg(lambda1, tau)
        #     sparsity_reg = self.model.sparsity_reg(W1, tau)
        #     score = mle_loss_prior + complexity_reg + sparsity_reg 
        #     cov_regularizer = torch.sqrt(torch.sum((torch.relu(-4 - Sigma_prior) + torch.relu(Sigma_prior - 4))**2))
        #     # score = mse_loss_prior + complexity_reg + sparsity_reg
        #     obj = mu * score + penalty + mu*cov_regularizer
        #     obj.backward()
        #     optimizer.step()


# Log basic metrics
            # metrics = {
            #     "T": t,
            #     "iteration": i,
            #     f"total_loss_t{t}": obj.item(),
            #     f"mle_loss_t{t}": mle_loss_prior.item(),
            #     f"complexity_reg_t{t}": complexity_reg.item(),
            #     f"sparsity_reg_t{t}": sparsity_reg.item(),
            #     f"score_t{t}": score.item(),
            #     f"structure_loss_t{t}": (penalty-h_val).item(),
            #     f"cycle_loss_t{t}": h_val.item(),
            #     f"penalty_t{t}": penalty.item(),
            #     f"cov_regularizer_t{t}": cov_regularizer.item(),
            #     f"sigma_0_1_t{t}": Sigma_prior[0, 1].item(),
            # }
            # wandb.log(metrics)

            
            # # Create separate Sigma tracking plot for each t
            # wandb.log({
            #     f"sigma_elements_t{t}": wandb.plot.line_series(
            #         xs=[i],
            #         ys=[[Sigma_prior[0, 1].item()]],
            #         keys=[f"Sigma[0,1] (t={t})"],
            #         title=f"Sigma Matrix Elements (t={t}, μ={mu:.4f})",
            #         xname="iteration"
            #     )
            # })




            x_est_posterior, Sigma_posterior = self.model.forward()
            mle_loss_posterior = self.model.mle_loss(self.x, x_est_posterior, Sigma_posterior)
            mse_loss_posterior = self.model.mse(self.x, x_est_posterior)
            diff = torch.abs(mle_loss_prior - mle_loss_posterior)
            # diff = torch.abs(mse_loss_prior - mse_loss_posterior)
            eigenvalues = torch.linalg.eigh(Sigma_prior)[0]
            if diff < 1e-12 and penalty < 1e-9:
                break
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler_alpha_beta.step()
                scheduler_Sigma.step()
                # scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nmu {mu}")
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {penalty.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                self.vprint(f'\t mle: {mle_loss_posterior}')
                self.vprint(f'\t mse: {mse_loss_posterior}')
                self.vprint(f'\tW1: {W1}')
                self.vprint(f'\tcycle loss: {h_val}')
                # self.vprint(f'\tW2: {Sigma_prior}')
                self.vprint(f'\tstructure loss: {penalty-h_val}')
                self.vprint(f'\tSigma: {Sigma_prior}')
                # self.vprint(f'\talpha: {self.model.alpha}')
                # self.vprint(f'\tbeta: {self.model.beta}')
                # self.vprint("Check M: ", self.model.M.grad)
                # self.vprint("Check y: ", x_est_posterior[:, 1])
                self.vprint("Check eigenvalues: ", eigenvalues.min().item())
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
                # model_copy = copy.deepcopy(self.model)
                model_copy = self.model.__class__(data=self.x)
                model_copy.load_state_dict(self.model.state_dict())
                lr_decay = False
                while success is False:
                    print("success: ", success)
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
        final_W1, _ = self.model.fc1_to_adj()
        output, final_W2 = self.model.forward()
        print("final_W1: ", final_W1)
        print("final_W2: ", final_W2)
        final_W1 = final_W1.cpu().detach().numpy()
        final_W2 = final_W2.cpu().detach().numpy()
        # final_W1[np.abs(final_W1) < w_threshold] = 0
        # final_W2[np.abs(final_W2) < w_threshold] = 0
        #return get_graph(final_W1, final_W2, data.columns, w_threshold)
        return final_W1, final_W2, output





        
if __name__ == "__main__":
    # W1 = torch.tensor([[1, 4, 0], [2, 0, 3], [0, 0, 3]], dtype=torch.float64)
    # W2 = torch.tensor([[0, 0, 0.7], [0, 0, 0.5], [0.7, 0.5, 0]], dtype=torch.float64)

    # # test cycle_loss
    # cycle_loss_result = cycle_loss(W1)
    # print("cycle_loss", cycle_loss_result)

    # # test ancestrality_loss
    # ancestrality_loss_result = ancestrality_loss(W1, W2)
    # print("ancestrality_loss", ancestrality_loss_result)

    # # test structure penalty
    # structure_penalty_result = structure_penalty(W1, W2, admg_class ="ancestral")
    # print("structure_penalty", structure_penalty_result)

    # # test get graph
    # vertices = ["X1", "X2", "X3"]
    # threshold = 0
    # G = get_graph(W1, W2, vertices, threshold)
    
    # print("Directed edges", G.di_edges)
    # print("Bidirected edges", G.bi_edges)

    # # ADMG_RKHSDagma class

    # # check forward
    # output, Sigma = eq_model.forward()
    # print("Sigma in the class: ", Sigma)

    # cov_matrix = np.cov(x.numpy().T)
    # print("True covariance matrix: ", cov_matrix)

    # # check fc1_to_adj
    # W1_result, W2_result = eq_model.fc1_to_adj()
    # print("fc1_to_ad W1: ", W1_result)
    # print("fc1_to_ad W2: ", W2_result)

    # check mle 
    # _, Sigma = eq_model.forward()
    # print("mle: ", eq_model.mle_loss(x_est, Sigma))

    # mle2 = 0
    # for i in range(x.shape[0]):
    #     xi = (x[i]-x_est[i]).unsqueeze(1)
    #     mle2 += (xi.T)@torch.linalg.inv(Sigma)@xi/x.shape[0]
    # sign, logdet = torch.linalg.slogdet(Sigma)
    # mle2 += logdet
    # mle2 = 0.5*mle2 + 0.5*x.shape[1]*torch.log(torch.tensor(2 * torch.pi))
    # print("To compared mle2: ", mle2)

    # mean = torch.tensor([0.0, 0.0])
    # mvn = torch.distributions.MultivariateNormal(mean, Sigma)
    # mle3 = 0
    # for i in range(x.shape[0]):
    #     xi = (x[i]-x_est[i])
    #     print("xi ", xi)
    #     mle3 += -mvn.log_prob(xi)
    # mle3 = mle3/x.shape[0]
    # print("To compared mle3: ", mle3)


    
    # # check complexity_reg
    # complexity_reg_result = eq_model.complexity_reg(lambda1=1e-3, tau = 1e-4)
    # print("complexity_reg: ", complexity_reg_result)

    # # check sparsity_reg
    # sparsity_reg_result = eq_model.sparsity_reg(W1_result, tau = 1e-4)
    # print("sparsity_reg: ", sparsity_reg_result)

    # optimization
    # example usage

    # DGP A->B->C->D; B<->D
    # beta = np.array([[0, 1, 0, 0],
    #                  [0, 0, -1.5, 0],
    #                  [0, 0, 0, 1],
    #                  [0, 0, 0, 0]]).T

    # omega = np.array([[1.2, 0, 0, 0],
    #                   [0, 1, 0, 0.6],
    #                   [0, 0, 1, 0],
    #                   [0, 0.6, 0, 1]])

    # # generate data according to the graph
    # true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    # X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    # X = X - np.mean(X, axis=0)  # centre the data
    # data = pd.DataFrame({"A": X[:, 0], "B": X[:, 1], "C": X[:, 2], "D": X[:, 3]})

    # A = np.random.uniform(low=0, high=10, size=100)
    # Z = np.random.uniform(low=0, high=5, size=100)
    # epsilon = np.random.normal(0,1, 100) 
    # B = np.array([A**2 + epsilon + Z for A, epsilon, Z in zip(A, epsilon, Z)])
    # C = np.array([0.05*(B**2) + epsilon for B, epsilon in zip(B, epsilon)])
    # D = np.array([0.1*(C**2) + epsilon + Z for C, epsilon, Z in zip(C, epsilon, Z)])
    # data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    
    # print("data: ", data.head())
    # eq_model2 = ADMG_RKHSDagma(data, gamma = 1).to(device)
    # model2 = RKHS_discovery(eq_model2, admg_class = "bowfree", verbose=True)
    # W1, W2, output = model2.fit(data, lambda1=1e-3, tau=1e-4, T = 6, mu_init = 1.0, lr=0.03, w_threshold=0.0)
    # print("W1: ", W1)
    # print("W2: ", W2)
    print("____________________________________________________________________________________________________________________")
    
    #  # Sample data generation: let's assume X and X_hat are from some known distributions for the sake of example
    n_samples = 300  # Number of samples
    dim = 2  # Dimension of the normal vectors

    # Random data for X and X_hat
    True_Sigma = np.array([[1, 0.3],    # Variance of X is 1, covariance between X and Y is 0.8
                  [0.3, 1.5]])   # Variance of Y is 1, covariance between Y and X is 0.8
    epsilon = np.random.multivariate_normal([0] * dim, True_Sigma, size=n_samples) #[n, d]
    epsilon = torch.tensor(epsilon, dtype=torch.float64)
    # x1 = torch.tensor(np.random.uniform(low=-3, high=3, size=n_samples))
    # x1 = torch.zeros(n_samples)
    x1 = epsilon[:, 0]
    x2 = 10*torch.sin(x1)
    # Step 4: Combine these results into a new tensor of shape [n, 2]
    x1_true = x1 #+ epsilon[:, 0]
    x2_true = x2 + epsilon[:, 1]
    X = torch.stack((x1, x2), dim=1)
    X_true = torch.stack((x1_true, x2_true), dim=1)
    eq_model2 = ADMG_RKHSDagma(X_true, gamma = 1)
    model2 = RKHS_discovery(x=X_true, model=eq_model2, admg_class = "ancestral", verbose=True)
    W1, Sigma, x_est = model2.fit()
    y_hat = x_est[:, 1].detach().numpy()
    empirical_covariance = np.cov(epsilon, rowvar=False)
    print("Empirical Covariance Matrix:", empirical_covariance)
    print("estimated Sigma: ", Sigma)

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X.detach().numpy()[:, 0], X_true.detach().numpy()[:, 1], label='y_noise', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(X.detach().numpy()[:, 0], x_est.detach().numpy()[:, 1], label='y_est', color='red', marker='s') 
    plt.scatter(X.detach().numpy()[:, 0], X.detach().numpy()[:, 1], label='y', color='green', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")

    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.scatter(X.detach().numpy()[:, 1], X_true.detach().numpy()[:, 0], label='x_noise', color='blue', marker='o')  # Plot x vs. y1
    plt.scatter(X.detach().numpy()[:, 1], x_est.detach().numpy()[:, 0], label='x_est', color='red', marker='s') 
    plt.scatter(X.detach().numpy()[:, 1], X.detach().numpy()[:, 0], label='x', color='green', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    print("The programm is closed")
    

    ### To Do:

    # Try the linear simulation from reference and may other simulations from the reference
    # Add other algebraic constraints
    # build other non-linear simulations

    # Try to find the real dataset
    # Check two new papers for further enhancement

    # Set boundary for weight parameters
    # Speed up (Use GPU)
    # check if the java code can be used
