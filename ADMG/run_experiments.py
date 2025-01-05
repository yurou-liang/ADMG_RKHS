import utils
# from RKHS import RKHS_DAGMA
import admg_rkhs2
import sys
# from notears import NotearsMLP, notears_nonlinear, NotearsSobolev, notears_linear
import torch
import numpy as np
import json
import argparse
import os

def RKHS_likelihood(lambda1, tau, X, device, B_true, gamma, function_type, d, seed, thresh = 0.00, T=6, lr = 0.03):
    results = {}

    X = X.to(device)
    eq_model = admg_rkhs2.Sigma_RKHSDagma(X, gamma = gamma).to(device)
    model = admg_rkhs2.Sigma_discovery(x=X, model=eq_model, admg_class = "ancestral", verbose=True)
    Sigma_start, x_est_start = eq_model.forward()
    start_mle = eq_model.mle_loss(X, x_est_start, Sigma_start).detach().cpu().numpy()
    W_est_no_thresh, Sigma, x_est = model.fit(lambda1=lambda1, tau=tau, T = 6)
    try: 
        W_est = abs(W_est_no_thresh) * (abs(W_est_no_thresh) > thresh)
        acc = utils.count_accuracy(B_true, W_est != 0)
    except Exception as e:
        print("W_est_no_thresh: ", W_est_no_thresh)
        raise ValueError(f'W_est is not a DAG')

    diff = np.linalg.norm(W_est - abs(B_true))
    Sigma, x_est = eq_model.forward()
    mle = eq_model.mle_loss(X, x_est, Sigma).detach().cpu().numpy()
    W_est = eq_model.fc1_to_adj()
    h_val = eq_model.h_func(W_est, s=1).detach().cpu().numpy()
    filename = f'RKHS_likelihood_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mle.item(),
    'h_val': h_val.item(),
    'start mse': start_mle.tolist(),
    'W_est_no_thresh': W_est_no_thresh.tolist(),
    'B_true': B_true.tolist(),
    'X': X.detach().cpu().numpy().tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

    torch.cuda.empty_cache()


def RKHS_test(lambda1, tau, X, device, B_true, gamma, function_type, d, seed, thresh = 0.05, T=6, lr = 0.03):
    results = {}

    X = X.to(device)
    eq_model = RKHS_DAGMA.RKHSDagma(X, gamma = gamma).to(device)
    model = RKHS_DAGMA.RKHSDagma_nonlinear(eq_model)
    x_est_start = eq_model.forward()
    start_mse = eq_model.mse(x_est_start).detach().cpu().numpy()
    W_est_no_thresh, output = model.fit(X, lambda1=lambda1, tau=tau, T = T, mu_init = 1.0, lr=lr, w_threshold=0.0)
    try: 
        W_est = abs(W_est_no_thresh) * (abs(W_est_no_thresh) > thresh)
        acc = utils.count_accuracy(B_true, W_est != 0)
    except Exception as e:
        print("W_est_no_thresh: ", W_est_no_thresh)
        raise ValueError(f'W_est is not a DAG')

    diff = np.linalg.norm(W_est - abs(B_true))
    x_est = eq_model.forward()
    mse = eq_model.mse(x_est).detach().cpu().numpy()
    W_est = eq_model.fc1_to_adj()
    h_val = eq_model.h_func(W_est, s=1).detach().cpu().numpy()
    filename = f'RKHS_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mse.item(),
    'h_val': h_val.item(),
    'start mse': start_mse.tolist(),
    'W_est_no_thresh': W_est_no_thresh.tolist(),
    'B_true': B_true.tolist(),
    'X': X.detach().cpu().numpy().tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 

    torch.cuda.empty_cache()


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def NOTEARS_MLP(X, B_true, function_type, d, seed):
    results = {}
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est, output = notears_nonlinear(model, X, lambda1=2e-2)
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    X_torch = torch.from_numpy(X)
    x_est = model.forward(X_torch)
    mse = squared_loss(x_est, X_torch)
    h_val = model.h_func().detach().cpu().numpy()
    filename = f'NOTEARS_MLP_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mse.item(),
    'h_val': h_val.item(),
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 


def NOTEARS_SOB(X, B_true, function_type, d, seed):
    results = {}
    model = NotearsSobolev(d = d, k = 10)
    W_est, output = notears_nonlinear(model, X, lambda1=3e-2)
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    X_torch = torch.from_numpy(X)
    x_est = model.forward(X_torch)
    mse = squared_loss(x_est, X_torch)
    h_val = model.h_func().detach().cpu().numpy()
    filename = f'NOTEARS_SOB_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'mse': mse.item(),
    'h_val': h_val.item(),
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 


def LINEAR_NOTEARS(X, B_true, function_type, d, seed):
    results = {}
    W_est = notears_linear(X, lambda1=0.1, loss_type = 'l2')
    acc = utils.count_accuracy(B_true, W_est != 0)
    diff = np.linalg.norm(W_est - abs(B_true))
    filename = f'LINEAR_NOTEARS_function_type_{function_type}_d{d}_seed{seed}'
    results = {
    'SHD': acc['shd'],
    'TPR': acc['tpr'],
    'F1': acc['f1'],
    'diff': diff,
    'W_est': W_est.tolist(),
    'B_true': B_true.tolist(),
    'X': X.tolist()}

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4) 




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Comparison between RKHADagma and NOTEARS',)

    parser.add_argument('-d', '--num_nodes', nargs='+', default=[10, 20, 30, 40], type=int)
    parser.add_argument('-ER_order', default=4, type=int)
    parser.add_argument('-gamma', default=None, type=int)
    parser.add_argument('-l', '--lambda1', default=1e-3, type=float)
    parser.add_argument('-t', '--tau', default=1e-4, type=float)
    parser.add_argument('-T', '--num_iterations', default=6, type=int)
    parser.add_argument('-s', '--random_seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-f', '--function_type',  nargs='+', default=['gp-add', 'gp', 'mlp'], type=str)
    parser.add_argument('-thresh', default=0.05, type = float)
    parser.add_argument('-lr', default=0.03, type = float)
    parser.add_argument('-A', '--algorithm', default='RKHS', type=str)

    args = parser.parse_args()

    torch.set_default_dtype(torch.double)

    torch.backends.cudnn.benchmark = False
    # os.chdir('./experiments')

    for idx_nodes, n_nodes in enumerate(args.num_nodes):
        print('-----------------------------\n' +
              f'| Experiments with {n_nodes} Nodes |\n' +
              '-----------------------------\n')
        if args.gamma is None:
            gamma = 0.4 * n_nodes

        for function_type in args.function_type:
            print(f'>>> Generating Data with function type {function_type} <<<')
            n, d, s0, graph_type, sem_type = 300, n_nodes, n_nodes * args.ER_order, 'ER', function_type 
            utils.set_random_seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            B_true = utils.simulate_dag(d, s0, graph_type)
            print("shape: ", B_true.shape[0])
            noise = np.random.uniform(low=0.5, high=1.5, size=B_true.shape[0])
            print(noise.shape)
            X = utils.simulate_nonlinear_sem(B_true, n, sem_type, noise_scale=noise)
            X_torch = torch.from_numpy(X)
            if args.algorithm == "RKHS":
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                torch.set_default_device(device)
                print('>>> Performing DAGMA-RKHS discovery <<<')
                RKHS_test(lambda1 = args.lambda1, tau = args.tau, X = X_torch, device=device, B_true = B_true, gamma= gamma, function_type = function_type, 
                        d = n_nodes, seed = args.random_seed, thresh = args.thresh, T=args.num_iterations, lr = args.lr)
            elif args.algorithm == "RKHS_likelihood":
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                torch.set_default_device(device)
                print('>>> Performing DAGMA-RKHS discovery <<<')
                RKHS_likelihood(lambda1 = args.lambda1, tau = args.tau, X = X_torch, device=device, B_true = B_true, gamma= gamma, function_type = function_type, 
                        d = n_nodes, seed = args.random_seed, thresh = args.thresh, T=args.num_iterations, lr = args.lr)
            elif args.algorithm == "NOTEARS_MLP":
                print('>>> Performing NOTEARS_MLP discovery <<<')
                NOTEARS_MLP(X = X, B_true = B_true, function_type = function_type, d = n_nodes, seed = args.random_seed)

            elif args.algorithm == "NOTEARS_SOB":
                print('>>> Performing NOTEARS_SOB discovery <<<')
                NOTEARS_SOB(X = X, B_true = B_true, function_type = function_type, d = n_nodes, seed = args.random_seed)

            elif args.algorithm == "LINEAR_NOTEARS":
                print('>>> Performing LINEAR_NOTEARS discovery <<<')
                LINEAR_NOTEARS(X = X, B_true = B_true, function_type = function_type, d = n_nodes, seed = args.random_seed)  
            else:
                print("Given algorithm is not valid.")
            
