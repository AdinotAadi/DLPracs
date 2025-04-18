# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# Sample loss function: f(x) = x^2 + 3x + 2
def loss_function(x):
    return x**2 + 3*x + 2

# Derivative of the loss function: f'(x) = 2x + 3
def gradient(x):
    return 2*x + 3

# Optimization algorithms
class OptimizerDemo:
    def __init__(self, learning_rate=0.1, max_iters=50):
        self.lr = learning_rate
        self.max_iters = max_iters

    def gradient_descent(self, x_init):
        x = x_init
        history = [x]
        for _ in range(self.max_iters):
            x -= self.lr * gradient(x)
            history.append(x)
        return history

    def momentum(self, x_init, beta=0.9):
        x = x_init
        v = 0  # Momentum term
        history = [x]
        for _ in range(self.max_iters):
            v = beta * v + (1 - beta) * gradient(x)
            x -= self.lr * v
            history.append(x)
        return history

    def adagrad(self, x_init, epsilon=1e-8):
        x = x_init
        g_square = 0
        history = [x]
        for _ in range(self.max_iters):
            g = gradient(x)
            g_square += g**2
            x -= (self.lr / (np.sqrt(g_square) + epsilon)) * g
            history.append(x)
        return history

    def rmsprop(self, x_init, beta=0.9, epsilon=1e-8):
        x = x_init
        s = 0
        history = [x]
        for _ in range(self.max_iters):
            g = gradient(x)
            s = beta * s + (1 - beta) * g**2
            x -= (self.lr / (np.sqrt(s) + epsilon)) * g
            history.append(x)
        return history

    def adam(self, x_init, beta1=0.9, beta2=0.999, epsilon=1e-8):
        x = x_init
        m, v = 0, 0
        history = [x]
        for t in range(1, self.max_iters + 1):
            g = gradient(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            x -= (self.lr / (np.sqrt(v_hat) + epsilon)) * m_hat
            history.append(x)
        return history

# Initialize optimizer class
optim = OptimizerDemo(learning_rate=0.1, max_iters=50)

# Initial value of x
x_init = 10

# Apply different optimization algorithms
gd_path = optim.gradient_descent(x_init)
momentum_path = optim.momentum(x_init)
adagrad_path = optim.adagrad(x_init)
rmsprop_path = optim.rmsprop(x_init)
adam_path = optim.adam(x_init)

# Plot optimization progress
plt.figure(figsize=(10, 6))
plt.plot(gd_path, label="Gradient Descent", linestyle="--")
plt.plot(momentum_path, label="Momentum", linestyle="--")
plt.plot(adagrad_path, label="AdaGrad", linestyle="--")
plt.plot(rmsprop_path, label="RMSprop", linestyle="--")
plt.plot(adam_path, label="Adam", linestyle="--")
plt.axhline(y=-1.5, color='black', linestyle='dotted', label="Global Minimum")
plt.xlabel("Iterations")
plt.ylabel("x Value")
plt.title("Optimization Function Demonstration")
plt.legend()
plt.show()