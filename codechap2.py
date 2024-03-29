import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate synthetic data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize and train the linear regression model
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)

# Make predictions on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
accu = r2_score(y_test, predictions)

print("MSE:", mse)
print("R-squared (Accuracy):", accu)

# Visualize the results
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(8, 6))

m1 = ax.scatter(X_train, y_train, color=cmap(0.9), s=10, label='Training Data')
m2 = ax.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Test Data')
ax.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")

ax.set_title('Linear Regression - Predicted vs Actual')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
plt.show()
