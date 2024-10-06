import numpy as np
from scipy.optimize import minimize


class MySVM:
    def __init__(self, X_len, C=1, rbf_gamma=2):
        self.C = C
        self.rbf_gamma = rbf_gamma
        self.alpha_vec = np.random.rand(X_len)
        self.X_len = X_len
        self.alpha_bounds = [(0, C) for _ in range(X_len)]
        self.alpha_constraints = {'type': 'eq', 'fun': self._zeroequal_constr}


    def _zeroequal_constr(self, alpha):
        return np.dot(alpha, self.t)
    

    def fit(self, X, y):
        X, y = X.copy(), y.copy()
        y = y.flatten()
        self.X = X
        self.t = np.where(y == 0, -1, 1)
        self.K = self._k_rbf_matrix(X)

        self.alpha_vec = minimize(fun = self._dual_problem,
                                 x0 = self.alpha_vec,
                                 bounds = self.alpha_bounds,
                                 constraints = self.alpha_constraints)
        self.alpha_vec = self.alpha_vec['x']
        
        ns = np.count_nonzero(self.alpha_vec)
        self.sv = np.nonzero(self.alpha_vec)[0]
        temp_sum = np.zeros(ns)
        for j in self.sv:
            temp_sum = temp_sum + self.alpha_vec[j] * self.t[j] * self._k_rbf_val(X[self.sv], X[j])
        self.b = np.sum(self.t[self.sv] - temp_sum) / ns
        
        return self
    

    def predict(self, X_new):
        y = np.zeros(len(X_new))
        for j in self.sv:
            y = y + self.alpha_vec[j] * self.t[j] * self._k_rbf_val(X_new, self.X[j])
        y = y + self.b
        return np.where(y < 0, 0, 1)


    def _dual_problem(self, alpha):
        alpha_matrix = alpha.reshape(-1, 1) @ alpha.reshape(1, -1)
        t_matrix = self.t.reshape(-1, 1) @ self.t.reshape(1, -1)
        return 0.5 * np.sum(alpha_matrix * t_matrix * self.K) - np.sum(alpha)
    

    def _k_rbf_matrix(self, X):
        a = X.reshape(X.shape[0], -1, X.shape[1])
        b = X.reshape(-1, X.shape[0], X.shape[1])
        K = self._k_rbf_val(a, b)
        return K


    def _k_rbf_val(self, a, b):
        return np.exp(-self.rbf_gamma * np.linalg.norm(a - b, axis=-1)**2)
