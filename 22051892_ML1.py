import numpy as np
import matplotlib.pyplot as plt

# Given dataset
x = np.array(
    [9.1, 8, 9.1, 8.4, 6.9, 7.7, 15.6, 7.3, 7, 7.2, 10.1, 11.5, 7.1, 10, 8.9, 7.9, 5.6, 6.3, 6.7, 10.4, 8.5, 7.4, 6.3,
     5.4, 8.9, 9.4, 7.5, 11.9, 7.8, 7.4, 10.8, 10.2, 6.2, 7.7, 13.7, 8, 6.7, 6.7, 7, 8.3, 7.4, 9.9, 6.1, 7, 5.4, 10.7,
     7.6, 8.9, 9.2, 6.6, 7.2, 8, 7.8, 7.9, 7, 7, 7.6, 9.1, 9, 7.9, 6.6, 11.9, 6.5, 7.1, 8.8, 7.5, 7.7, 6, 10.6, 6.6,
     8.2, 7.9, 7.1, 5.6, 6.4, 7.5, 9.8, 7, 10.5, 7.1, 6.2, 6.5, 7.7, 7.2, 9.3, 8.5, 7.7, 6.8, 7.8, 8.7, 9.6, 7.2, 9.3,
     8.1, 6.6, 7.8, 10.2, 6.1, 7.3, 7.3])
y = np.array(
    [0.99523, 0.99007, 0.99769, 0.99386, 0.99508, 0.9963, 1.0032, 0.99768, 0.99584, 0.99609, 0.99774, 1.0003, 0.99694,
     0.99965, 0.99549, 0.99364, 0.99378, 0.99379, 0.99524, 0.9988, 0.99733, 0.9966, 0.9955, 0.99471, 0.99354, 0.99786,
     0.9965, 0.9988, 0.9964, 0.99713, 0.9985, 0.99565, 0.99578, 0.9976, 1.0014, 0.99685, 0.99648, 0.99472, 0.99914,
     0.99408, 0.9974, 1.0002, 0.99402, 0.9966, 0.99402, 1.0029, 0.99718, 0.9986, 0.9952, 0.9952, 0.9972, 0.9976, 0.9968,
     0.9978, 0.9951, 0.99629, 0.99656, 0.999, 0.99836, 0.99396, 0.99387, 1.0004, 0.9972, 0.9972, 0.99546, 0.9978,
     0.99596, 0.99572, 0.9992, 0.99544, 0.99747, 0.99668, 0.9962, 0.99346, 0.99514, 0.99476, 1.001, 0.9961, 0.99598,
     0.99608, 0.9966, 0.99732, 0.9962, 0.99546, 0.99738, 0.99456, 0.9966, 0.99553, 0.9984, 0.9952, 0.997, 0.99586,
     0.9984, 0.99542, 0.99655, 0.9962, 0.9976, 0.99464, 0.9983, 0.9967])

# Normalize the data
x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))


# Hypothesis function
def hypothesis(x, theta_0, theta_1):
    return theta_0 + theta_1 * x


# Cost function
def compute_cost(x, y, theta_0, theta_1):
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((hypothesis(x, theta_0, theta_1) - y) ** 2)
    return cost


# Gradient descent function
def gradient_descent(x, y, theta_0, theta_1, alpha, iterations, epsilon=1e-6):
    m = len(y)
    cost_history = []

    for it in range(iterations):
        h = hypothesis(x, theta_0, theta_1)
        theta_0 -= alpha * (1 / m) * np.sum(h - y)
        theta_1 -= alpha * (1 / m) * np.sum((h - y) * x)

        cost = compute_cost(x, y, theta_0, theta_1)
        cost_history.append(cost)

        # Convergence check
        if it > 0 and abs(cost_history[-2] - cost_history[-1]) < epsilon:
            break

    return theta_0, theta_1, cost_history


# Gradient descent for lr = 0.5
alpha = 0.5
iterations = 1000
theta_0, theta_1 = 0, 0
theta_0, theta_1, cost_history = gradient_descent(x_norm, y, theta_0, theta_1, alpha, iterations)

# Print final theta values and cost
print(f"Final theta_0: {theta_0}, theta_1: {theta_1}")
print(f"Final Cost: {cost_history[-1]}")

# Plot cost function vs iterations (for first 50 iterations)
plt.plot(range(50), cost_history[:50], label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (First 50 Iterations)")
plt.legend()
plt.show()

# Plot dataset and regression line
plt.scatter(x_norm, y, color='blue', label='Data points')
plt.plot(x_norm, hypothesis(x_norm, theta_0, theta_1), color='red', label='Linear regression')
plt.xlabel('Normalized x')
plt.ylabel('y')
plt.legend()
plt.show()

# Test different learning rates
learning_rates = [0.005, 0.5, 5]

for lr in learning_rates:
    theta_0, theta_1, cost_history = gradient_descent(x_norm, y, 0, 0, lr, iterations)

    # Plot cost for the first 50 iterations
    plt.plot(range(50), cost_history[:50], label=f"lr = {lr}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations for Different Learning Rates")
plt.legend()
plt.show()


# Stochastic gradient descent (SGD)
def stochastic_gradient_descent(x, y, theta_0, theta_1, alpha, iterations):
    m = len(y)
    cost_history = []

    for it in range(iterations):
        for i in range(m):
            h = hypothesis(x[i], theta_0, theta_1)
            theta_0 -= alpha * (h - y[i])
            theta_1 -= alpha * (h - y[i]) * x[i]

        cost = compute_cost(x, y, theta_0, theta_1)
        cost_history.append(cost)

    return theta_0, theta_1, cost_history


# Mini-batch gradient descent
def mini_batch_gradient_descent(x, y, theta_0, theta_1, alpha, iterations, batch_size=10):
    m = len(y)
    cost_history = []

    for it in range(iterations):
        idx = np.random.permutation(m)
        x_shuffled = x[idx]
        y_shuffled = y[idx]

        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            h = hypothesis(x_batch, theta_0, theta_1)

            theta_0 -= alpha * (1 / batch_size) * np.sum(h - y_batch)
            theta_1 -= alpha * (1 / batch_size) * np.sum((h - y_batch) * x_batch)

        cost = compute_cost(x, y, theta_0, theta_1)
        cost_history.append(cost)

    return theta_0, theta_1, cost_history


# Choose suitable learning rate for SGD and Mini-batch
alpha = 0.05

# Stochastic Gradient Descent
theta_0_sgd, theta_1_sgd, cost_history_sgd = stochastic_gradient_descent(x_norm, y, 0, 0, alpha, iterations)

# Mini-Batch Gradient Descent
theta_0_mini, theta_1_mini, cost_history_mini = mini_batch_gradient_descent(x_norm, y, 0, 0, alpha, iterations)

# Plot cost function for different gradient descent methods
plt.plot(range(50), cost_history[:50], label="Batch Gradient Descent")
plt.plot(range(50), cost_history_sgd[:50], label="Stochastic Gradient Descent")
plt.plot(range(50), cost_history_mini[:50], label="Mini-batch Gradient Descent")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations for Different Gradient Descent Methods")
plt.legend()
plt.show()
