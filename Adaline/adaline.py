import numpy as np

class Adaline:

    def _init_(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter): #wagi, koszt
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost_ = (errors**2).sum() / 2
            self.cost_.append(cost_)
            return self

    def activation(self, X)
        return self.net_input(X)

    def predict(self,X)
        return np.where(self.activation(X) >= 0.0, 1, -1)
        
