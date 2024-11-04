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
from notears.lbfgsb_scipy import LBFGSBScipy
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
    Sigma_init = cov + cov.T + torch.diag(Sigma.diag())
    # Sigma_init = cov + cov.T + torch.eye(d)
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
        Sigma = torch.cov(self.x.T)
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
    
    def mle_loss(self, x_est: torch.tensor, Sigma: torch.tensor) -> torch.tensor: # [n, d] -> [1, 1]
        # mle = torch.trace((self.x - x_est)@torch.linalg.inv(self.Sigma)@((self.x - x_est).T)) # Check if Sigma invertible !!!!, aslo divide by n
        # Sigma = self.I
        tmp = torch.linalg.solve(Sigma, (self.x - x_est).T)
        mle = torch.trace((self.x - x_est)@tmp)/self.n
        sign, logdet = torch.linalg.slogdet(Sigma)
        mle += logdet
        return mle
    
    def mse(self, x_est: torch.tensor): # [1, 1]
      squared_loss = 0.5 / self.n * torch.sum((x_est - self.x) ** 2)
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

    def _create_bounds(self, tiers, unconfounded_vars, var_names):
        """
        Create bounds on parameters given prior knowledge.

        :param tiers: iterable over iterables corresponding to variable tiers.
        :param unconfounded_vars: iterable of names of variables that have no incoming bidirected edges.
        :param var_names: names of all variables in the problem.
        :return: iterable of tuples corresponding to bounds on possible values of the parameters.
        """

        if tiers is None:
            tiers = [var_names]

        unconfounded_vars = set(unconfounded_vars)

        # dictionary containing what tier each variable is in
        tier_dict = {}
        for tier_num in range(len(tiers)):
            for var in tiers[tier_num]:
                tier_dict[var] = tier_num

        # set bounds on possible values by applying background knowledge
        bounds_directed_edges = []
        bounds_bidirected_edges = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):

                # no self loops
                if i == j:
                    bounds_directed_edges.append((0, 0))

                # i -> j is not allowed if i appears later in the causal order???
                elif tier_dict[var_names[i]] > tier_dict[var_names[j]]:
                    bounds_directed_edges.append((0, 0))

                # otherwise i -> j is allowed
                else:
                    bounds_directed_edges.append((-4, 4))

                # no self confounding and enforce symmetry???
                if i <= j:
                    bounds_bidirected_edges.append((0, 0))

                # no confounding between i and j if either are unconfounded
                elif var_names[i] in unconfounded_vars or var_names[j] in unconfounded_vars:
                    bounds_bidirected_edges.append((0, 0))

                # otherwise i <-> j is allowed
                else:
                    bounds_bidirected_edges.append((-4, 4))
        return bounds_directed_edges + bounds_bidirected_edges # combine both to one list
    
    def dual_ascent_step(self, rho, alpha, h, lambda1, tau, tiers=None, unconfounded_vars=[], 
                         rho_max=1e+16, w_threshold=0.05, verbose=False):
        
       h_new = None
       optimizer = LBFGSBScipy(self.model.parameters())
       while rho < rho_max:
           def closure():
                optimizer.zero_grad()
                W1, W2 = self.model.fc1_to_adj()
                h_val = structure_penalty(W1, W2, self.admg_class)
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                x_est_prior, Sigma_prior = self.model.forward()
                mle_loss_prior = self.model.mle_loss(x_est_prior, Sigma_prior)
                complexity_reg = self.model.complexity_reg(lambda1, tau)
                sparsity_reg = self.model.sparsity_reg(W1, tau)
                score = mle_loss_prior + complexity_reg + sparsity_reg 
                obj = score + penalty
                obj.backward()
                return obj
           optimizer.step(closure)  # NOTE: updates model in-place
           with torch.no_grad():
                W1, W2 = self.model.fc1_to_adj()
                h_new = structure_penalty(W1, W2, self.admg_class)
           if h_new > 0.25 * h:
                rho *= 10
           else:
                break
       alpha += rho * h_new
       return rho, alpha, h_new
    
    def notears_nonlinear(self, lambda1, tau, max_iter: int = 100, h_tol: float = 1e-8, rho_max: float = 1e+16,
                            w_threshold: float = 0.3):
        rho, alpha, h = 1.0, 0.0, np.inf
        for _ in range(max_iter):
            rho, alpha, h = self.dual_ascent_step(rho, alpha, h, lambda1, tau, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        final_W1, _ = self.model.fc1_to_adj()
        output, final_W2 = self.model.forward()
        print("final_W1: ", final_W1)
        print("final_W2: ", final_W2)
        final_W1 = final_W1.cpu().detach().numpy()
        final_W2 = final_W2.cpu().detach().numpy()
        final_W1[np.abs(final_W1) < w_threshold] = 0
        final_W2[np.abs(final_W2) < w_threshold] = 0
        #return get_graph(final_W1, final_W2, data.columns, w_threshold)
        return final_W1, final_W2, output
           
