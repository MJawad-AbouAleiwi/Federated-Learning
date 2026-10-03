# A softmax-regression model
import numpy as np

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)

    # Parameter Exchange
    def get_weights(self):
        return [self.W.copy(), self.b.copy()]

    def set_weights(self, weights):
        self.W, self.b = weights[0].copy(), weights[1].copy()

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        p = self.predict_proba(X)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        acc = (p.argmax(axis=1) == y).mean()
        return loss, acc

    def fit(self, X, y, epochs=1, lr=0.1, batch_size=32, seed=0):
        # Mini-batch SGD
        rng = np.random.default_rng(seed)
        n_classes = self.b.shape[0]
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = idx[start:start + batch_size]
                Xb, yb = X[batch], y[batch]
                p = self.predict_proba(Xb)
                p[np.arange(len(yb)), yb] -= 1
                p /= len(yb)
                self.W -= lr * (Xb.T @ p)
                self.b -= lr * p.sum(axis=0)