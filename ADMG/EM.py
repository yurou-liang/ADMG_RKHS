# a simple example of how you can optimize two variables alternatively using PyTorch. In this example, we'll minimize a function 
# ( f(x, y) = 2x^2 + 2xy + 2y^2 - 6y ) by alternating the optimization between ( x ) and ( y ).
import torch
import torch.optim as optim

# Define the function to optimize
def func(x, y):
    return 2 * x**2 + 2 * x * y + 2 * y**2 - 6 * y

# Initialize variables
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)

# Define optimizers for x and y
# optimizer_x = optim.SGD([x], lr=0.1)
# optimizer_y = optim.SGD([y], lr=0.1)
optimizer = optim.SGD([x, y], lr=0.1)

# Number of iterations
num_iterations = 100

for i in range(num_iterations):
    # # Optimize x
    # optimizer_x.zero_grad()
    # loss = func(x, y)
    # loss.backward()
    # optimizer_x.step()
    
    # # Optimize y
    # optimizer_y.zero_grad()
    # loss = func(x, y)
    # loss.backward()
    # optimizer_y.step()

    optimizer.zero_grad()
    loss = func(x, y)
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Iteration {i}: x = {x.item()}, y = {y.item()}, f(x, y) = {loss.item()}")

print(f"Final result: x = {x.item()}, y = {y.item()}, f(x, y) = {func(x, y).item()}")
# In this code:

# We define the function ( f(x, y) ) to be minimized.
# We initialize the variables ( x ) and ( y ) with requires_grad=True to enable gradient computation.
# We create separate optimizers for ( x ) and ( y ) using stochastic gradient descent (SGD).
# We alternate the optimization steps for ( x ) and ( y ) within a loop.