import numpy as np

class LinearReg:

    def __init__(self,learn_rate = 0.01,n_iters = 1000):
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.bias = None
        self.weight = None


    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X,self.weight) + self.bias

            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))

            db = (1/n_samples) * np.sum(y_pred - y)

            self.weight = self.weight - self.learn_rate * dw

            self.bias = self.bias - self.learn_rate * db

    def predict(self,X):
        y_pred = np.dot(X,self.weight) + self.bias

        return y_pred