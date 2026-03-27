import numpy as np
import scipy as sp


class LogisticRegression:
    def __init__(self):
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y, lamb=0.0):
        z = X @ self.weights
        p = self.sigmoid(z)
        eps = 1e-12
        data_loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        regularization_loss = 0.5 * lamb * np.sum(self.weights[1:] ** 2)
        return data_loss + regularization_loss

    def gradient(self, X, y, lamb=0.0):
        z = X @ self.weights
        p = self.sigmoid(z)
        grad = X.T @ (p - y) / len(y)
        grad[1:] += lamb * self.weights[1:]
        return grad

    def hessian(self, X, lamb=0.0):
        z = X @ self.weights
        p = self.sigmoid(z)
        R = p * (1 - p)
        H = (X.T * R) @ X / len(p)
        H[1:, 1:] += lamb * np.eye(H.shape[0] - 1)
        return H

    def hessian_vector_product(self, X, v, lamb=0.0):
        z = X @ self.weights
        p = self.sigmoid(z)
        R = p * (1 - p)
        Xv = X @ v
        RXv = R * Xv
        v_reg = v.copy()
        v_reg[0] = 0
        return (X.T @ RXv) / len(p) + lamb * v_reg

    def fit(self, X, y, X_val, y_val, method='gd', epochs=100, lr=0.1, batch_size=None, lamb=0.0,
            early_stopping=True, patience=5, tolerance=1e-3):
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
        n, d = X.shape
        self.weights = np.zeros(d)

        use_batch = batch_size is not None

        best_val_loss = float('inf')
        epochs_wo_imp = 0
        best_weights = self.weights.copy()
        for i in range(epochs):
            if method == 'gd':
                if use_batch:
                    indices = np.random.permutation(n)
                    X_shuffled = X[indices]
                    y_shuffled = y[indices]

                    for start in range(0, n, batch_size):
                        end = min(start + batch_size, n)
                        X_batch = X_shuffled[start:end]
                        y_batch = y_shuffled[start:end]
                        grad = self.gradient(X_batch, y_batch, lamb)
                        self.weights -= lr * grad
                else:
                    grad = self.gradient(X, y, lamb)
                    self.weights -= lr * grad

            elif method == 'newton':
                grad = self.gradient(X, y, lamb)
                H = self.hessian(X, lamb)
                H += 1e-6 * np.eye(H.shape[0])
                step = np.linalg.solve(H, grad)
                self.weights -= step

            elif method == 'cholesky':
                grad = self.gradient(X, y, lamb)
                H = self.hessian(X, lamb)
                H += 1e-6 * np.eye(H.shape[0])
                L = sp.linalg.cholesky(H, lower=True)
                step = sp.linalg.cho_solve((L, True), grad)
                self.weights -= step

            elif method == 'cg':
                grad = self.gradient(X, y, lamb)

                def Hv(v):
                    return self.hessian_vector_product(X, v, lamb) + 1e-6 * v

                H_op = sp.sparse.linalg.LinearOperator((d, d), matvec=Hv)
                step, info = sp.sparse.linalg.cg(H_op, grad)
                if info > 0: print(info)
                self.weights -= step

            # training stuff
            loss = self.loss(X, y, lamb)
            preds = self.predict(X[:, 1:])
            acc = np.mean(preds == y)
            train_accs.append(acc)
            train_losses.append(loss)

            # validation stuff
            loss = self.loss(X_val, y_val, lamb=lamb)

            if early_stopping:
                if loss < best_val_loss - tolerance:
                    epochs_wo_imp = 0
                    best_val_loss = loss
                    best_weights = self.weights.copy()

                else:
                    epochs_wo_imp += 1

                if epochs_wo_imp >= patience:
                    break

            preds = self.predict(X_val[:, 1:])
            y_labels = np.argmax(y_val, axis=1)
            acc = np.mean(preds == y_labels)
            val_accs.append(acc)
            val_losses.append(loss)

        if early_stopping:
            self.weights = best_weights

        return {
            'train_accs': train_accs,
            'train_losses': train_losses,
            'val_accs': val_accs,
            'val_losses': val_losses
        }

    def predict_prob(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.sigmoid(X @ self.weights)

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    def score(self, X, y, type='accuracy'):
        return np.mean(self.predict(X) == y)
