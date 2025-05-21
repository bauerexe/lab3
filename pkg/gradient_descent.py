import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import time

class GradientDescent(ABC):
    def __init__(self, learning_rate=0.1, name="Gradient Descent", max_iter=20_000):
        self.learning_rate = learning_rate
        self.name = name
        self.max_iter = max_iter

    @abstractmethod
    def schedule(self, k):
        pass

    def optimize(self, func, grad_func, x0, eps=1e-8, **kwargs):
        x = np.array(x0, dtype=float)
        trajectory = [x.copy()]

        start_time = time.time()
        for k in range(self.max_iter):
            grad = grad_func(x)

            if np.isnan(grad).any() or np.isinf(grad).any():
                print(f"NaN or Inf encountered at iteration {k}. Stopping optimization.")
                break

            if np.linalg.norm(grad) < eps:
                break

            lr = self.schedule(k)
            x_new = x - lr * grad

            if np.isnan(x_new).any() or np.isinf(x_new).any():
                print(f"NaN or Inf encountered in new point at iteration {k}. Stopping optimization.")
                break

            x = x_new
            trajectory.append(x.copy())

        end_time = time.time()
        time_taken = end_time - start_time
        num_iterations = len(trajectory) - 1

        return x, np.array(trajectory), num_iterations, time_taken

    def plot_3d_trajectory(self, trajectory, func, xrange=(-10, 10), yrange=(-10, 10), resolution=100):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np.linspace(*xrange, resolution), np.linspace(*yrange, resolution))
        Z = func([X, Y])

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
        ax.plot(trajectory[:, 0], trajectory[:, 1], func([trajectory[:, 0], trajectory[:, 1]]),
                marker='o', markersize=7, linestyle='-', linewidth=2, color='r', label=self.name)

        ax.set_title(f"Trajectory of {self.name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.legend()
        plt.show()

    def plot_2d_trajectory(self, trajectory, func, xrange=(-10, 10), yrange=(-10, 10), resolution=100):
        X, Y = np.meshgrid(np.linspace(*xrange, resolution), np.linspace(*yrange, resolution))
        Z = func([X, Y])

        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=50, cmap='viridis')
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=4, label=self.name)

        plt.title(f"2D Contour and Trajectory of {self.name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.show()

class Function:
    def __init__(self, func):
        self.func = func

    def value(self, x):
        return self.func(x)

    def gradient(self, x, eps=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1, x2 = x.copy(), x.copy()
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (self.func(x1) - self.func(x2)) / (2 * eps)
        return grad

    def plot_all_methods(self, methods, x0, xrange=(-6, 6), yrange=(-6, 6), resolution=100, gradient=None):
        if gradient is None:
            gradient = self.gradient
        for method in methods:
            x_min, trajectory, num_iterations, time_taken = method.optimize(self.value, grad_func= gradient, x0=x0, eps=1e-8)
            print(f"{method.name}, minimum found at: {x_min} f(minX) is {self.func(x_min)}")
            print(f"Iterations: {num_iterations}, Time taken: {time_taken:.4f} seconds")
            method.plot_3d_trajectory(trajectory, self.func, xrange, yrange, resolution)
            method.plot_2d_trajectory(trajectory, self.func, xrange, yrange, resolution)

class ExponentialDecay(GradientDescent):
    def __init__(self, learning_rate=0.1, lambd=0.01, max_iterations=20_000):
        super().__init__(learning_rate, name="Exponential Decay", max_iter= max_iterations)
        self.lambd = lambd

    def schedule(self, k):
        return self.learning_rate * np.exp(-self.lambd * k)

class PolynomialDecay(GradientDescent):
    def __init__(self, learning_rate=0.1, alpha=0.5, beta=1, max_iterations=20_000):
        super().__init__(learning_rate, name="Polynomial Decay", max_iter= max_iterations)
        self.alpha = alpha
        self.beta = beta

    def schedule(self, k):
        return self.learning_rate / ((self.beta * k + 1) ** self.alpha)

class ConstantDecay(GradientDescent):
    def __init__(self, learning_rate=0.1,  max_iterations=20_000):
        super().__init__(learning_rate, name="Constant Decay", max_iter= max_iterations)


    def schedule(self, k):
        return self.learning_rate

class StepDecay(GradientDescent):
    def __init__(self, learning_rate=0.1, step_size=100, decay_factor=0.5, max_iterations=20_000):
        super().__init__(learning_rate, name="Step Decay", max_iter= max_iterations)
        self.step_size = step_size
        self.decay_factor = decay_factor

    def schedule(self, k):
        return self.learning_rate * (self.decay_factor ** (k // self.step_size))

def quadratic_function(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == "__main__":
    func = Function(quadratic_function)

    methods = [
        ExponentialDecay(learning_rate=0.1, lambd=0.05),
        PolynomialDecay(learning_rate=0.1, alpha=0.5, beta=1),
        ConstantDecay(learning_rate=0.1),
        StepDecay(learning_rate=0.1, step_size=50, decay_factor=0.5)
    ]

    x0 = [5.0, 5.0]

    func.plot_all_methods(methods, x0)
