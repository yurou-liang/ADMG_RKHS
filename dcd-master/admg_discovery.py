import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.extend import primitive, defvjp
import functools
import scipy.optimize as sopt
import pandas as pd
from ananke.graphs import ADMG
from scipy.special import comb
import matplotlib.pyplot as plt
import copy
import math

from ricf import bic
from utils.admg2pag import admg_to_pag, pprint_pag

# h(W) = trace((I + W◦W/d)^d) - d
@primitive
def cycle_loss(W):
    """
    Compute the loss, h_acyc, due to directed cycles in the induced graph of W.

    :param W: numpy matrix.
    :return: float corresponding to penalty on directed cycles.
    Use trick when computing the trace of a product
    """
    d = len(W)
    M = np.eye(d) + W * W/d
    E = np.linalg.matrix_power(M, d - 1)
    return (E.T * M).sum() - d


# (fake) ∇h(W) = [exp(W ◦ W)]^T ◦ 2W
def dcycle_loss(ans, W):
    """
    Analytic derivatives for the cycle loss function.

    :param ans:
    :param W: numpy matrix.
    :return: gradients for the cycle loss.

    ans: This is the value of the function cycle_loss(x) evaluated at x. 
    It is often unused directly in simple derivatives but can be critical in more complex cases 
    where the derivative depends explicitly on the function value as well as its inputs.

    g: represents the upstream gradient passed down during the backpropagation in neural networks or during chain rule application in nested functions. 
    It is a scalar by which the computed derivative must be multiplied.
    """
    W_shape = W.shape
    d = len(W)
    M = anp.eye(d) + W*W/d
    E = anp.linalg.matrix_power(M, d-1)
    return lambda g: anp.full(W_shape, g) * E.T * W * 2


# required for autograd integrate the customized gradient of h to the Autograd's system
defvjp(cycle_loss, dcycle_loss)


def ancestrality_loss(W1, W2):
    """
    Compute the loss due to violations of ancestrality in the induced ADMG of W1, W2.

    :param W1: numpy matrix for directed edge coefficients.
    :param W2: numpy matrix for bidirected edge coefficients.
    :return: float corresponding to penalty on violations of ancestrality.
    """
    d = len(W1)
    W1_pos = anp.multiply(W1, W1)
    W2_pos = anp.multiply(W2, W2)
    W1k = np.eye(d)
    M = np.eye(d)
    for k in range(1, d):
        W1k = anp.dot(W1k, W1_pos)
        # M += comb(d, k) * (1 ** k) * W1k (typical binoimial)
        M += 1.0/math.factorial(k) * W1k #(special scaling)

    return anp.sum(anp.multiply(M, W2_pos))


def reachable_loss(W1, W2, alpha_d=1, alpha_b=2, s=anp.log(5000)):
    """
    Compute the loss due to presence of c-trees in the induced ADMG of W1, W2.

    :param W1: numpy matrix for directed edge coefficients.
    :param W2: numpy matrix for bidirected edge coefficients.
    :return: float corresponding to penalty on c-trees.
    """

    d = len(W1)
    greenery = 0

    # iterate over each vertex in turn
    for var_index in range(d):

        # create a for Vi and an inverse mask
        mask = anp.array([1 if i == var_index else 0 for i in range(d)]) * 1
        W1_fixed = anp.multiply(W1, W1)
        W2_fixed = anp.multiply(W2, W2)

        # try to "primal fix" at most d-1 times
        for i in range(d-1):

            # compute binomial expansion of sum((I + \alpha B)^k \circ D))
            Bk = np.eye(d)
            M = np.eye(d)
            for k in range(1, d):
                Bk = anp.dot(Bk, W2_fixed)
                M += comb(d, k) * (alpha_b ** k) * Bk

            # compute the primal fixability mask
            p_fixability_matrix = anp.multiply(M, W1_fixed)
            e2x = anp.exp(anp.clip(s*(anp.mean(p_fixability_matrix, axis=1) + mask), 0, 4))
            fixability = (e2x - 1)/(e2x + 1)
            fixability_mask = anp.vstack([fixability for _ in range(d)])

            # apply the primal fixing operation
            W1_fixed = anp.multiply(W1_fixed, fixability_mask)
            W2_fixed = anp.multiply(W2_fixed, fixability_mask)
            W2_fixed = anp.multiply(W2_fixed, fixability_mask.T)

        # compute (I + \alpha A)^k for A = W1_fixed and W2_fixed
        Bk, Dk = np.eye(d), np.eye(d)
        eW1_fixed, eW2_fixed = np.eye(d), np.eye(d)

        for k in range(1, d):
            Dk = anp.dot(Dk, W1_fixed)
            Bk = anp.dot(Bk, W2_fixed)
            eW1_fixed += 1/np.math.factorial(k) * Dk
            eW2_fixed += 1/np.math.factorial(k) * Bk
            eW1_fixed += comb(d, k) * (alpha_d ** k) * Dk
            eW2_fixed += comb(d, k) * (alpha_b ** k) * Bk

        # compute penalty on Vi-rooted c-tree
        greenery += anp.sum(anp.multiply(eW1_fixed[:, var_index], eW2_fixed[:, var_index])) - 1

    return greenery


def bow_loss(W1, W2):
    """
    Compute the loss due to presence of bows in the induced ADMG of W1, W2.

    :param W1: numpy matrix for directed edge coefficients.
    :param W2: numpy matrix for bidirected edge coefficients.
    :return: float corresponding to penalty on bows.
    """
    W1_pos = anp.multiply(W1, W1)/len(W1)
    W2_pos = anp.multiply(W2, W2)/len(W1)
    return anp.sum(anp.multiply(W1_pos, W2_pos))


class Discovery:
    """
    Class for structure learning/causal discovery in ADMGs
    """

    def __init__(self, lamda=0.05):
        """
        Constructor.

        :param lamda: float > 0 corresponding to L0-regularization strength.
        """

        self.X_ = None
        self.S_ = None
        self.Z_ = None
        self.W1_ = None
        self.W2_ = None
        self.Wii_ = None
        self.convergence_ = None
        self.lamda = lamda
        self.G_ = None

    def primal_loss(self, params, rho, alpha, Z, structure_penalty_func):
        """
        Calculate the primal loss in RICF.

        :param params: parameter vector theta that can be reshaped into directed/bidirected coefs.
        :param rho: penalty on loss due to violations of given ADMG class.
        :param alpha: dual ascent Lagrangian parameter.
        :param Z: dictionary mapping Vi to pseudovariables computed for Vi.
        :param structure_penalty_func: function computing loss for ancestrality, aridity, bow-freenes.
        :return: float corresponding to the loss.
        """

        n, d = self.X_.shape
        W1 = anp.reshape(params[0:d * d], (d, d))
        W2 = anp.reshape(params[d * d:], (d, d))
        W2 = W2 + W2.T

        loss = 0.0
        for var_index in range(d):
            loss += 0.5 / n * anp.linalg.norm(self.X_[:, var_index] - anp.dot(self.X_, W1[:, var_index]) -
                                                            anp.dot(Z[var_index], W2[:, var_index])) ** 2 #LS(\theta)

        structure_penalty = cycle_loss(W1) + structure_penalty_func(W1, W2)
        structure_penalty = 0.5 * rho * (structure_penalty ** 2) + alpha * structure_penalty
        eax2 = anp.exp((anp.log(n) * anp.abs(params)))
        tanh = (eax2 - 1) / (eax2 + 1)
        return loss + structure_penalty + anp.sum(tanh) * self.lamda

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

    def _compute_pseudo_variables(self, W1, W2):
        """
        Compute pseudo-variables Z for a given set of parameters for directed and bidirected edges.

        :param W1: coefficients for directed edges.
        :param W2: covariance matrix for residual noise terms (bidirected edge coefficients).
        :return: dictionary mapping Vi to its pseudovariables Z.
        """

        # iterate over each vertex and get Zi
        Z = {}
        d = len(W1)
        for var_index in range(d):

            # get omega_{-i, -i}
            indices = list(range(0, var_index)) + list(range(var_index + 1, d))
            omega_minusii = W2[anp.ix_(indices, indices)]
            omega_minusii_inv = anp.linalg.inv(omega_minusii)

            # get epsilon_minusi
            # residual, ignoring the var_index column
            epsilon = self.X_ - anp.matmul(self.X_, W1)
            epsilon_minusi = anp.delete(epsilon, var_index, axis=1)

            # calculate Z_minusi
            Z_minusi = (omega_minusii_inv @ epsilon_minusi.T).T

            # insert a column of zeros to maintain the shape
            Z[var_index] = anp.insert(Z_minusi, var_index, 0, axis=1)

        return Z

    def get_graph(self, W1, W2, vertices, threshold):
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

    def _discover_admg(self, data, admg_class, tiers=None, unconfounded_vars=[], max_iter=100,
                       h_tol=1e-8, rho_max=1e+16, w_threshold=0.00,
                       ricf_increment=1, ricf_tol=1e-4, verbose=False):
        """
        Internal function for running the structure learning procedure once.

        :param data: Pandas dataframe containing data.
        :param admg_class: class of ADMGs to consider. options: ancestral, arid, or bowfree.
        :param tiers: iterable over iterables corresponding to variable tiers.
        :param unconfounded_vars: iterable of names of variables that have no incoming bidirected edges.
        :param max_iter: maximum iterations to run the dual ascent procedure.
        :param h_tol: tolerance for violations of the property defining admg_class.
        :param rho_max: maximum penalty applied to violations of the property defining admg_class.
        :param w_threshold: float deciding what is close enough to zero to rule out an edge.
        :param ricf_increment: positive integer to increase maximum number of RICF iterations at every dual ascent step.
        :param ricf_tol: tolerance for terminating RICF.
        :param verbose: Boolean indicating whether to print intermediate outputs.
        :return: best fitting Ananke ADMG that is found.
        """

        # get shape of the data, make a copy and calculate sample covariance
        self.X_ = anp.copy(data.values)
        n, d = self.X_.shape
        self.S_ = anp.cov(self.X_.T) # The covariance matrix from the given data

        # create bounds by applying background knowledge
        bounds = self._create_bounds(tiers, unconfounded_vars, data.columns)

        # initialize starting point
        W1_hat = anp.random.uniform(-0.5, 0.5, (d, d))
        W2_hat = anp.random.uniform(-0.05, 0.05, (d, d))
        W2_hat[np.tril_indices(d)] = 0 # The lower triangle of W2_hat is zero, includes the diagonal elements
        W2_hat = W2_hat + W2_hat.T # So that W2_hat is symmertic and has zeros on the diagonal elements
        W2_hat = anp.multiply(W2_hat, 1 - np.eye(d)) # Elementwise multiplication, ensure the diagonal elements are 0
        Wii_hat = anp.diag(anp.diag(self.S_))  # zero matrix, with only the diagonal filled by the value of diagonal elements of self.S_

        # initial settings
        rho, alpha, h = 1.0, 0.0, np.inf
        ricf_max_iters = 1
        convergence = False

        # set loss functions according to desired ADMG class
        if admg_class == "ancestral":
            penalty = ancestrality_loss
        elif admg_class == "arid":
            penalty = reachable_loss
        elif admg_class == "bowfree":
            penalty = bow_loss
        elif admg_class == "none":
            penalty = lambda *args, **kwargs: 0
        else:
            raise NotImplemented("Invalid ADMG class")

        # gradient stuff
        objective = functools.partial(self.primal_loss)
        gradient = grad(objective)

        # iterate till convergence or max iterations
        for num_iter in range(max_iter):

            # initialize W1, W2, Wii
            W1_new, W2_new, Wii_new = None, None, None
            h_new = None  # also keep track of loss

            while rho < rho_max:

                # initialize with the last best guess we have of these matrices
                W1_new, W2_new, Wii_new = W1_hat.copy(), W2_hat.copy(), Wii_hat.copy()

                # perform RICF till convergence or max iterations
                ricf_iter = 0
                while ricf_iter < ricf_max_iters:

                    ricf_iter += 1
                    W1_old = W1_new.copy()  # Directed edges = Beta
                    W2_old = W2_new.copy()  # Bidirected edges = Omega
                    Wii_old = Wii_new.copy()

                    # get pseudovariables
                    Z = self._compute_pseudo_variables(W1_new, W2_new + Wii_new)

                    # get values of the current estimates and solve
                    current_estimates = np.concatenate((W1_new.flatten(), W2_new.flatten()))
                    sol = sopt.minimize(self.primal_loss, current_estimates,
                                        args=(rho, alpha, Z, penalty),
                                        method='L-BFGS-B',
                                        options={'disp': False}, bounds=bounds, jac=gradient)

                    W1_new = np.reshape(sol.x[0:d * d], (d, d))
                    W2_new = np.reshape(sol.x[d * d:], (d, d))
                    W2_new = W2_new + W2_new.T # So that W2_new is symmertic, TRY 0.5 time this in my algorithm
                    # Wii stands for Variance of the error, i.e. Wii = \beta_{ii}
                    for var_index in range(d):
                        Wii_new[var_index, var_index] = np.var(
                            self.X_[:, var_index] - np.dot(self.X_, W1_new[:, var_index]))
                    # break criterion in regularized RICF
                    if np.sum(np.abs(W1_old - W1_new)) + np.sum(np.abs((W2_old + Wii_old) - (W2_new + Wii_new))) < ricf_tol:
                        convergence = True
                        break

                h_new = cycle_loss(W1_new) + penalty(W1_new, W2_new)
                if verbose:
                    print(num_iter, h_new)
                    print("W1_est\n", np.round(W1_new, 3), "\n\nW2_est\n", np.round(W2_new, 3))

                if h_new < 0.25 * h:
                    break
                else:
                    rho *= 10

            W1_hat, W2_hat, Wii_hat = W1_new.copy(), W2_new.copy(), Wii_new.copy()
            h = h_new
            alpha += rho * h
            ricf_max_iters += ricf_increment
            if h <= h_tol or rho >= rho_max:
                break

        final_W1, final_W2 = W1_hat.copy(), W2_hat + Wii_hat 
        final_W1[np.abs(final_W1) < w_threshold] = 0
        final_W2[np.abs(final_W2) < w_threshold] = 0
        output = self.X_@final_W1
        return self.get_graph(final_W1, final_W2, data.columns, w_threshold), convergence, output, final_W1, final_W2

    def discover_admg(self, data, admg_class, tiers=None, unconfounded_vars=[], max_iter=100,
                      h_tol=1e-8, rho_max=1e+16, num_restarts=5, w_threshold=0.05,
                      ricf_increment=1, ricf_tol=1e-4, verbose=False, detailed_output=False):
        """
        Function for running the structure learning procedure within a pre-specified ADMG hypothesis class.

        :param data: Pandas dataframe containing data.
        :param admg_class: class of ADMGs to consider. options: ancestral, arid, or bowfree.
        :param tiers: iterable over iterables corresponding to variable tiers.
        :param unconfounded_vars: iterable of names of variables that have no incoming bidirected edges.
        :param max_iter: maximum iterations to run the dual ascent procedure.
        :param h_tol: tolerance for violations of the property defining admg_class.
        :param rho_max: maximum penalty applied to violations of the property defining admg_class.
        :param num_restarts: number of random restarts since W1 and W2 are initialized randomly
        :param w_threshold: float deciding what is close enough to zero to rule out an edge.
        :param ricf_increment: positive integer to increase maximum number of RICF iterations at every dual ascent step.
        :param ricf_tol: tolerance for terminating RICF.
        :param verbose: Boolean indicating whether to print intermediate outputs.
        :param detailed_output: Boolean indicating whether to print detailed intermediate outputs.
        :return: best fitting Ananke ADMG that is found over num_restarts times trials.
        """

        best_bic = np.inf

        for i in range(num_restarts):

            if verbose:
                print("Random restart", i+1)
            G, convergence = self._discover_admg(data, admg_class, tiers, unconfounded_vars, max_iter,
                                                 h_tol, rho_max, w_threshold, ricf_increment,
                                                 ricf_tol, detailed_output)
            curr_bic = bic(data, G)
            if verbose:
                print("Estimated di_edges:", G.di_edges)
                print("Estimated bi_edges", G.bi_edges)
                print("BIC", curr_bic)
            if curr_bic < best_bic:
                self.G_ = copy.deepcopy(G)
                self.convergence_ = convergence
                best_bic = curr_bic

        if verbose:
            print("Final estimated di_edges:", self.G_.di_edges)
            print("Final estimated bi_edges", self.G_.bi_edges)
            print("Final BIC", best_bic)

        return self.G_


if __name__ == "__main__":

    # example usage
    #np.random.seed(42)
    size = 100
    dim = 4

    # DGP A->B->C->D; B<->D, beta is W1, omega is W2
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

    # A = np.random.uniform(low=0, high=10, size=100)
    # Z = np.random.uniform(low=0, high=5, size=100)
    # epsilon = np.random.normal(0,1, 100) 
    # B = np.array([A**2 + epsilon + Z for A, epsilon, Z in zip(A, epsilon, Z)])
    # C = np.array([0.05*(B**2) + epsilon for B, epsilon in zip(B, epsilon)])
    # D = np.array([0.1*(C**2) + epsilon + Z for C, epsilon, Z in zip(C, epsilon, Z)])
    

    # data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    learn = Discovery(lamda=0.05)
    G, convergence, output, final_W1, final_W2= learn._discover_admg(data, admg_class = "none", verbose=True)
    print("final_W1: ", final_W1)
    print("final_W2: ", final_W2)
    print(G.di_edges)
    print(G.bi_edges)

    # # convert ADMG to PAG
    # # pag = admg_to_pag(best_G)
    # # pprint_pag(pag)

    W1 = np.array([[1, 4, 0], [2, 0, 3], [0, 0, 3]])
    W2 = np.array([[0, 0, 0.7], [0, 0, 0.5], [0.7, 0.5, 0]])

    # # test cycle_loss
    # cycle_loss_result = cycle_loss(W1)
    # print("cycle_loss", cycle_loss_result)
        # test ancestrality_loss
    # ancestrality_loss_result = ancestrality_loss(W1, W2)
    # print("ancestrality_loss", ancestrality_loss_result)

    # np.random.seed(0)
    # #z = np.random.uniform(low=0, high=3, size=100)
    # x = np.random.uniform(low=-3, high=3, size=100)
    # epsilon = np.random.normal(0, 1, 100) 
    # y = np.array([np.sin(x)*10 + epsilon for x, epsilon in zip(x, epsilon)])
    # X = np.column_stack((x, y))
    # data = pd.DataFrame(X, columns=['x', 'y'])

    # Step 1: Define the covariance matrix
    # True_Sigma = np.array([[1, 0.8],    # Variance of X is 1, covariance between X and Y is 0.8
    #                 [0.8, 1]])   # Variance of Y is 1, covariance between Y and X is 0.8

    # epsilon = np.random.multivariate_normal([0] * 2, True_Sigma, size=200)
    # np.random.seed(0)
    # epsilon1 = epsilon[:, 0]
    # epsilon2 = epsilon[:, 1]
    # x = np.random.uniform(low=-3, high=3, size=200)
    # true_x = x + epsilon1
    # y = np.array([np.sin(x)*10 + epsilon2 for x, epsilon2 in zip(x, epsilon2)])
    # X = np.column_stack((true_x, y))
    # data = pd.DataFrame(X, columns=['x', 'y'])

    # learn = Discovery(lamda=0.05)
    # G, convergence, output = learn._discover_admg(data, admg_class = "none", verbose=True)
    # y_hat = output[:, 1]
    # print(G.di_edges)
    # print(G.bi_edges)
    # plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    # plt.scatter(x, y, label='y', color='blue', marker='o')  # Plot x vs. y1
    # plt.scatter(x, y_hat, label='y_est', color='red', marker='s') 
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
