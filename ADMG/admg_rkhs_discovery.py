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
from utils.admg2pag import admg_to_pag, pprint_pag

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


# h(W) = -logdet(sI-W◦W)+dlogs
@primitive
def cycle_loss(W: torch.tensor, s=1):
    """
    Compute the loss, h_acyc, due to directed cycles in the induced graph of W.

    :param W: numpy matrix.
    :return: float corresponding to penalty on directed cycles.
    Use trick when computing the trace of a product
    """
    d = W.size(0)
    s = torch.tensor(s)
    A = s*torch.eye(d) - W*W
    sign, logabsdet = torch.linalg.slogdet(A)
    h = -logabsdet + d * torch.log(s)
    return h


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
    # elif admg_class == "bowfree":
    #     penalty = bow_loss
    else:
        raise NotImplemented("Invalid ADMG class")
    structure_penalty = cycle_loss(W1) + penalty(W1, W2)
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

class ADMG_RKHSDagma(nn.Module):
    """
    Class for setting up causal discovery in ADMGs
    """
    def __init__(self, data, gamma = 1):
        super(ADMG_RKHSDagma, self).__init__() # inherit nn.Module
        if isinstance(data, pd.DataFrame):
            self.x = torch.tensor(data.values, dtype=torch.float64)
        elif isinstance(data, torch.Tensor):
            self.x = data # data matrix [n, d]  
        else:
            raise ValueError("Input data must be a pandas DataFrame or a torch.Tensor")
        self.d = self.x.shape[1]
        self.n = self.x.shape[0]
        self.gamma = gamma
        # initialize coefficients alpha
        alpha = torch.zeros(self.d, self.n)
        #alpha = torch.rand(self.d, self.n)
        self.alpha = nn.Parameter(alpha) 
        # initialize coefficients beta
        self.beta = nn.Parameter(torch.zeros(self.d, self.d, self.n))
        #self.beta = nn.Parameter(torch.rand(self.d, self.d, self.n))
        # initialize the symmetric bidirected adjacency matrix with 0 on the diagonal, entries are uniformly picked from [-0.1, 0.1]
        self.I = torch.eye(self.d)
        self.L = torch.rand(self.d, self.d) * 0.1 - 0.1
        self.L = nn.Parameter(self.L)
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
        Sigma = self.Sigma = self.L @ self.L.T + 1e-6*self.I
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

        Sigma = self.L @ self.L.T + 1e-6*self.I
        Wii = torch.diag(torch.diag(Sigma))
        W2 = Sigma - Wii
        return W1, W2
    
    def mle_loss(self, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        # mle = torch.trace((self.x - x_est)@torch.linalg.inv(self.Sigma)@((self.x - x_est).T)) # Check if Sigma invertible !!!!, aslo divide by n
        tmp = torch.linalg.solve(Sigma, (self.x - x_est).T)
        mle = torch.trace((self.x - x_est)@tmp)/self.n
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
    

    
class RKHS_discovery:
    """
    Class that implements the DAGMA algorithm for structure learning in ADMGs.
    """

    def __init__(self, model: nn.Module, admg_class, verbose: bool = False, dtype: torch.dtype = torch.float64):
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

    def minimize(self, 
            max_iter: float, 
            lr: float, 
            lambda1: float, 
            tau: float,
            lambda2: float, 
            mu: float, 
            s: float,
            lr_decay: float = False, 
            tol: float = 1e-6, 
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

        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            W1, W2 = self.model.fc1_to_adj()
            h_val = cycle_loss(W1, s=1)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            penalty = structure_penalty(W1, W2, self.admg_class)
            x_est_prior, Sigma_prior = self.model.forward()
            mle_loss_prior = self.model.mle_loss(x_est_prior, Sigma_prior)
            complexity_reg = self.model.complexity_reg(lambda1, tau)
            sparsity_reg = self.model.sparsity_reg(W1, tau)
            score = mle_loss_prior + complexity_reg + sparsity_reg 
            obj = mu * score + penalty
            obj.backward()
            optimizer.step()
            x_est_posterior, Sigma_posterior = self.model.forward()
            mle_loss_posterior = self.model.mle_loss(x_est_posterior, Sigma_posterior)
            diff = torch.abs(mle_loss_prior - mle_loss_posterior)
            if diff < 1e-6 and penalty < 1e-9:
                break
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
                pbar.update(1)
        return True


    def fit(self, 
        data: pd.DataFrame,
        lambda1: torch.float64 = .02, 
        tau: torch.float64 = .02,
        lambda2: torch.float64 = .005,
        T: torch.int = 4, 
        mu_init: torch.float64 = 0.1, 
        mu_factor: torch.float64 = .1, 
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
        self.x = torch.tensor(data.values, dtype=torch.float64)
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
                #model_copy = copy.deepcopy(self.model)
                model_copy = self.model.__class__(data=self.x)
                model_copy.load_state_dict(self.model.state_dict())
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
        final_W1, final_W2 = self.model.fc1_to_adj().cpu().detach().numpy()
        final_W1[np.abs(final_W1) < w_threshold] = 0
        final_W2[np.abs(final_W2) < w_threshold] = 0
        return get_graph(final_W1, final_W2, data.columns, w_threshold)





        
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
    # x = torch.tensor([[1, 4], [2, 1], [5, 7]], dtype=torch.float64)
    # x_est = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
    # eq_model = ADMG_RKHSDagma(x)

    # # check fc1_to_adj
    # W1_result = eq_model.fc1_to_adj()
    # print("fc1_to_ad: ", W1_result)

    # # check mle 
    # Sigma = eq_model.Sigma
    # print("mle: ", eq_model.mle_loss(x_est))

    # # check complexity_reg
    # complexity_reg_result = eq_model.complexity_reg(lambda1=1e-3, tau = 1e-4)
    # print("complexity_reg: ", complexity_reg_result)

    # # check sparsity_reg
    # sparsity_reg_result = eq_model.sparsity_reg(W1_result, tau = 1e-4)
    # print("sparsity_reg: ", sparsity_reg_result)

    # mle2 = 0
    # for i in range(x.shape[0]):
    #     xi = (x[i]-x_est[i]).unsqueeze(1)
    #     mle2 += (xi.T)@torch.linalg.inv(Sigma)@xi/x.shape[0]
    # print("To compared mle: ", mle2)

    # optimization
    # example usage
    np.random.seed(42)
    size = 100
    dim = 4

    # DGP A->B->C->D; B<->D
    beta = np.array([[0, 1, 0, 0],
                     [0, 0, -1.5, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]).T

    omega = np.array([[1.2, 0, 0, 0],
                      [0, 1, 0, 0.6],
                      [0, 0, 1, 0],
                      [0, 0.6, 0, 1]])

    # generate data according to the graph
    true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    X = X - np.mean(X, axis=0)  # centre the data

    data = pd.DataFrame({"A": X[:, 0], "B": X[:, 1], "C": X[:, 2], "D": X[:, 3]})
    print("data: ", data.head())
    eq_model2 = ADMG_RKHSDagma(data, gamma = 1)
    model2 = RKHS_discovery(eq_model2, admg_class = "ancestral")
    G = model2.fit(data, lambda1=1e-3, tau=1e-4, T = 6, mu_init = 1.0, lr=0.03, w_threshold=0.0)
    print("directed edges: ", G.di_edges)
    print("bidirected edges: ", G.bi_edges)

    ### To Do:
    # check the logic with example again
    # print intermediate steps during optimization out
    # Try quadratic toy example

    # Advancements:
    # Set boundary for weight parameters
    # Does log cholesky work better?
    # Speed up (Use GPU)
    # build other non-linear simulations
    # check if the java code can be used