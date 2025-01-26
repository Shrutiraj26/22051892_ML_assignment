# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load datasets
X = pd.read_csv('logisticX.csv', header=None).values
y = pd.read_csv('logisticY.csv', header=None).values.ravel()

# Logistic Regression Model
model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e10)
model.fit(X, y)

# Extract weights and intercept
weights = model.coef_[0]
intercept = model.intercept_[0]

# Define cost function
def compute_cost(X, y, weights, intercept):
    z = np.dot(X, weights) + intercept
    h = 1 / (1 + np.exp(-z))
    cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost

# Cost after training
cost_value = compute_cost(X, y, weights, intercept)
print(f"Weights: {weights}, Intercept: {intercept}, Cost: {cost_value}")

# Plot cost vs iterations
def simulate_gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    weights = np.zeros(n)
    intercept = 0
    costs = []
    
    for _ in range(iterations):
        z = np.dot(X, weights) + intercept
        h = 1 / (1 + np.exp(-z))
        
        dw = (1 / m) * np.dot(X.T, (h - y))
        db = (1 / m) * np.sum(h - y)
        
        weights -= learning_rate * dw
        intercept -= learning_rate * db
        cost = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        costs.append(cost)
    
    return costs

# Plot cost
costs = simulate_gradient_descent(X, y, learning_rate=0.1, iterations=50)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(costs) + 1), costs, marker='o')
plt.title("Cost Function vs Iterations (First 50)")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.grid()
plt.show()

# Plot decision boundary
def plot_decision_boundary(X, y, weights, intercept):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = weights[0] * xx + weights[1] * yy + intercept
    Z = 1 / (1 + np.exp(-Z))
    Z = Z >= 0.5

    plt.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="bwr", marker='o')
    plt.title("Dataset with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()

plot_decision_boundary(X, y, weights, intercept)

# Add new features (squared terms)
X_enhanced = np.hstack((X, X**2))
model_enhanced = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e10)
model_enhanced.fit(X_enhanced, y)

weights_enhanced = model_enhanced.coef_[0]
intercept_enhanced = model_enhanced.intercept_[0]

# Enhanced decision boundary
def plot_enhanced_decision_boundary(X, y, weights, intercept):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    x1, x2 = xx.ravel(), yy.ravel()
    x3, x4 = x1**2, x2**2
    Z = weights[0] * x1 + weights[1] * x2 + weights[2] * x3 + weights[3] * x4 + intercept
    Z = 1 / (1 + np.exp(-Z))
    Z = Z >= 0.5

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="bwr", marker='o')
    plt.title("Enhanced Dataset with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()

plot_enhanced_decision_boundary(X, y, weights_enhanced, intercept_enhanced)

# Confusion matrix and metrics
y_pred = model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
