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
from ananke.graphs import ADMG
from scipy.special import comb
import copy
from utils.admg2pag import admg_to_pag, pprint_pag

torch.set_default_dtype(torch.float64)


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
    def __init__(self, x: torch.tensor, gamma = 1):
        super(ADMG_RKHSDagma, self).__init__() # inherit nn.Module
        self.x = x # data matrix [n, d]   
        self.d = x.shape[1]
        self.n = x.shape[0]
        self.gamma = gamma
        # initialize coefficients alpha
        #alpha = torch.zeros(self.d, self.n)
        alpha = torch.rand(self.d, self.n)
        self.alpha = nn.Parameter(alpha) 
        # initialize coefficients beta
        #self.beta = nn.Parameter(torch.zeros(self.d, self.d, self.n))
        self.beta = nn.Parameter(torch.rand(self.d, self.d, self.n))
        
        # x: [n, d]; K: [d, n, n]: K[j, i, l] = k(x^i, x^l) without jth coordinate; grad_K1: [n, n, d]: gradient of k(x^i, x^l) wrt x^i_{k}; 
        # grad_K2: [n, n, d]: gradient of k(x^i, x^l) wrt x^l_{k}; mixed_grad: [n, n, d, d] gradient of k(x^i, x^l) wrt x^i_{a} and x^l_{b}

        self.omega = torch.ones(self.d, self.d)
        self.omega.fill_diagonal_(0)
        self.I = torch.eye(self.d)
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
        return output

        
if __name__ == "__main__":
    # W1 = torch.tensor([[1, 4, 0], [2, 0, 3], [0, 0, 3]], dtype=torch.float64)
    # W2 = torch.tensor([[0, 0, 0.7], [0, 0, 0.5], [0.7, 0.5, 0]], dtype=torch.float64)
    # # result = ancestrality_loss(W1, W2)
    # # print("result", result)
    # vertices = ["X1", "X2", "X3"]
    # threshold = 0
    # G = get_graph(W1, W2, vertices, threshold)
    
    # print("Directed edges", G.di_edges)
    # print("Bidirected edges", G.bi_edges)
    x = torch.tensor([[1, 4], [2, 1], [5, 7]], dtype=torch.float64)
    eq_model = ADMG_RKHSDagma(x)
    print("forward: ", eq_model.forward())
