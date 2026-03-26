import numpy as np
import scipy as sp


class SoftmaxRegression:
    def __init__(self):
        self.weights = None

    def softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def loss(self, X, y, lamb=0.0):
        Z = X @ self.weights
        P = self.softmax(Z)
        eps = 1e-12
        data_loss = -np.mean(np.sum(y * np.log(P + eps), axis=1))
        regularization_loss = 0.5 *  lamb * np.sum(self.weights[1:] ** 2)
        return data_loss + regularization_loss

    def gradient(self, X, y, lamb=0.0):
        Z = X @ self.weights
        P = self.softmax(Z)
        grad = X.T @ (P - y) / y.shape[0]
        grad[1:] += lamb * self.weights[1:]
        return grad

    def hessian_vector_product(self, X, v, lamb=0.0):
        Z = X @ self.weights
        P = self.softmax(Z)

        Xv = X @ v

        S = P * Xv
        S -= P * np.sum(S, axis=1, keepdims=True)

        v_reg = v.copy()
        v_reg[0] = 0

        return X.T @ S / X.shape[0] + lamb * v_reg

    def fit(self, X, y, X_val, y_val, method='gd', epochs=100, lr=0.1, batch_size=None, lamb=0.0):
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
        n, d = X.shape
        K = y.shape[1]
        self.weights = np.zeros((d, K))

        use_batch = batch_size is not None

        if use_batch and method != 'gd':
            raise ValueError('Can\'t use batch for anything other than GD')

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
                        grad = self.gradient(X_batch, y_batch, lamb=lamb)
                        self.weights -= lr * grad
                else:
                    grad = self.gradient(X, y, lamb=lamb)
                    self.weights -= lr * grad

            elif method == 'cg':
                grad = self.gradient(X, y, lamb=lamb)
                def Hv(v_flat):
                    v = v_flat.reshape(d, K)
                    Hv_mat = self.hessian_vector_product(X, v, lamb=lamb)
                    return Hv_mat.ravel() + 1e-6 * v_flat

                H_op = sp.sparse.linalg.LinearOperator((d * K, d * K), matvec=Hv)
                grad_flat = grad.ravel()
                step, info = sp.sparse.linalg.cg(H_op, grad_flat, maxiter=50)
                step = step.reshape(d, K)
                self.weights -= step
                if info > 0:
                    print("CG did not fully converge:", info)

            # training stuff
            loss = self.loss(X, y, lamb=lamb)
            preds = self.predict(X[:, 1:])
            y_labels = np.argmax(y, axis=1)
            acc = np.mean(preds == y_labels)
            train_accs.append(acc)
            train_losses.append(loss)

            # validation stuff
            loss = self.loss(X_val, y_val, lamb=lamb)
            preds = self.predict(X_val[:, 1:])
            y_labels = np.argmax(y_val, axis=1)
            acc = np.mean(preds == y_labels)
            val_accs.append(acc)
            val_losses.append(loss)

        return {
            'train_accs': train_accs,
            'train_losses': train_losses,
            'val_accs': val_accs,
            'val_losses': val_losses
        }

    def predict_prob(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.softmax(X @ self.weights)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Z = X @ self.weights
        return np.argmax(Z, axis=1)

    def score(self, X, y, type='accuracy'):
        y_labels = np.argmax(y, axis=1)
        return np.mean(self.predict(X) == y_labels)
