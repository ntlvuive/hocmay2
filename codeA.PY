def fit(self, X, y):
    n_samples, n_features = X.shape
    
    # Thêm cột bias vào X
    X = np.hstack((np.ones((n_samples, 1)), X))
    
    # Khởi tạo tham số
    self.weights = np.zeros(n_features + 1)  # +1 cho cột bias
    self.bias = 0

    # Gradient descent
    for _ in range(self.n_iters):
        y_predicted = np.dot(X, self.weights)
        # Tính gradient
        dw = (1.0 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1.0 / n_samples) * np.sum(y_predicted - y)

        # Cập nhật tham số
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
