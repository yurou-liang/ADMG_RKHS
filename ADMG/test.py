import torch
import torch.nn as nn
import torch.optim as optim

class SigmaModule(nn.Module):
    def __init__(self, dim):
        super(SigmaModule, self).__init__()
        # Initialize a lower triangular matrix with positive values on the diagonal.
        # self.L = nn.Parameter(torch.randn(dim, dim))
        # # Make L lower triangular
        # self.register_buffer('tril_indices', torch.tril_indices(dim, dim, offset=0))
        self.d = dim
        self.n = 4
        alpha = torch.zeros(self.d, self.n)
        #alpha = torch.rand(self.d, self.n)
        self.alpha = nn.Parameter(alpha) 
        # initialize coefficients beta
        self.I = torch.eye(self.d)
        self.L = torch.rand(self.d, self.d) * 0.1 - 0.1
        self.L = nn.Parameter(self.L)

    def forward(self):
        # Ensure L is lower triangular with positive diagonal elements
        # L = torch.zeros_like(self.L)
        # L[self.tril_indices[0], self.tril_indices[1]] = self.L[self.tril_indices[0], self.tril_indices[1]]
        x = self.alpha**2
        Sigma = self.L @ self.L.T + 1e-6*self.I
        # Compute Sigma = L L^T
        return torch.sum(x), Sigma

    def log_det_sigma(self):
        # Compute log(det(Sigma)) using L
        score1, Sigma = self.forward()
        diag_elements = torch.diag(Sigma)
        return score1 + 2 * torch.sum(torch.log(torch.diag(diag_elements)))

# Example usage
dim = 3
model = SigmaModule(dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy objective function to minimize
for epoch in range(100):
    optimizer.zero_grad()
    
    Sigma = model()
    # Example loss: log determinant of Sigma
    loss = model.log_det_sigma()
    
    # Perform backpropagation
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(f'Epoch {epoch+1}, Loss: {Sigma}')
