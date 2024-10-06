from scipy.optimize import minimize
import numpy as np


class MyNode:
    def __init__(self, bounds, X, y, max_depth, depth=0):
        self.depth = depth
        self.max_depth = max_depth
        self.X = X
        self.y = y
        self.bounds = bounds
        self.left_node = None
        self.right_node = None
        self.gini = MyNode._gini_ind(y)
        self.res = round(len(self.y[self.y == 1]) / len(self.y))
        self.p = max(len(self.y[self.y == 1]) / len(self.y), len(self.y[self.y == 0]) / len(self.y))
        self.node_abort = False
        if self.depth <= self.max_depth and self.gini > 0:
            self.node_abort = self.split()


    
    def split(self):
        options_of_t = np.empty(len(self.bounds))
        ginis = np.empty(len(self.bounds))
        flag = False
        i = 0
        while not flag:
            for k in range(len(self.bounds)):
                self.k = k
                t0 = [np.random.uniform(self.bounds[k, 0], self.bounds[k, 1])]
                t = minimize(fun=self.J, x0=t0, 
                            bounds=[(self.bounds[k, 0], self.bounds[k, 1])],
                            method='Powell')['x']
                options_of_t[k] = t[0]
                ginis[k] = self.J(t)
            t_best = options_of_t[np.argmin(ginis)]
            self.t = t_best
            self.k = np.argmin(ginis)
            if (len(self.y[self.X[:, self.k] < t_best]) == 0 or
                len(self.y[self.X[:, self.k] >= t_best]) == 0):
                flag = False
            else:
                flag = True
            i = i + 1
            if i == 10:
                return True
        self.split_data(t_best)
        return False
    

    def J(self, t):
        t = t[0]
        y_left = self.y[self.X[:, self.k] < t]
        y_right = self.y[self.X[:, self.k] >= t]
        G_left = MyNode._gini_ind(y_left)
        G_right = MyNode._gini_ind(y_right)
        m, m_left, m_right = len(self.y), len(y_left), len(y_right)
        return m_left * G_left / m + m_right * G_right / m
    

    def split_data(self, t):
        bounds_right, bounds_left = self.bounds.copy(), self.bounds.copy()
        bounds_right[self.k, 0] = t
        bounds_left[self.k, 1] = t
        X_right = self.X[self.X[:, self.k] >= t]
        X_left = self.X[self.X[:, self.k] < t]
        y_right = self.y[self.X[:, self.k] >= t]
        y_left = self.y[self.X[:, self.k] < t]
        self.left_node = MyNode(bounds_left, X_left, y_left, self.max_depth, self.depth + 1)
        self.right_node = MyNode(bounds_right, X_right, y_right, self.max_depth, self.depth + 1)



    @staticmethod
    def _gini_ind(y):
        if len(y) == 0:
            return 0
        len_y1 = len(y[y == 1])
        return 1 - (len_y1**2 + (len(y) - len_y1)**2) / len(y)**2
    

class MyTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.redo = False


    def fit(self, X, y):
        X, y = X.copy(), y.flatten()
        bounds = MyTree._X_bound(X)
        self._node_0 = MyNode(bounds, X, y, self.max_depth)
        self.redo = self.check_abort()
        return self
    

    def predict(self, X_new):
        y_new = np.empty(len(X_new))
        for i in range(len(X_new)):
            x_i = X_new[i]
            node_c = self.find_node(x_i)
            y_new[i] = node_c.res
        return y_new


    def soft_predict(self, X_new):
        y_new = np.empty(len(X_new))
        for i in range(len(X_new)):
            x_i = X_new[i]
            node_c = self.find_node(x_i)
            y_new[i] = node_c.p
        return self.predict(X_new), y_new


    def find_node(self, x_i):
        depth = 0
        node_c = self._node_0
        while depth < self.max_depth and node_c.gini != 0:
            k, t = node_c.k, node_c.t
            if x_i[k] < t:
                node_c = node_c.left_node
            else:
                node_c = node_c.right_node
            depth = depth + 1
        return node_c


    def check_abort(self):
        nodes = self._traverse_nodes(self._node_0)
        for node in nodes:
            if node.node_abort == True:
                return True
        return False

    def _traverse_nodes(self, node):
        nodes = [node]
        if node.left_node is not None:
            nodes += self._traverse_nodes(node.left_node)
        if node.right_node is not None:
            nodes += self._traverse_nodes(node.right_node)
        return nodes


    @staticmethod
    def _X_bound(X):
        bounds = np.empty((X.shape[1], 2))
        for k in range(len(bounds)):
            bounds[k] = np.array([np.min(X[:, k]), np.max(X[:, k])])
        return bounds
    

class MyRandomForest:
    def __init__(self, max_depth, trees_number):
        self.max_depth = max_depth
        self.trees_number = trees_number

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.flatten()
        self.forest = []
        for i in range(self.trees_number):
            flag = True
            while flag:
                X_temp, y_temp = self._rand_batch()
                tree = MyTree(max_depth=self.max_depth)
                flag = tree.fit(X_temp, y_temp).redo
            self.forest.append(tree)
        return self
    

    def predict(self, X_new):
        result = np.empty((len(X_new), self.trees_number))
        for i in range(self.trees_number):
            tree = self.forest[i]
            y_new, p = tree.soft_predict(X_new)
            result[:, i] = np.abs(np.ones(len(X_new)) - y_new - p)
        result = np.round(np.mean(result, axis=1))
        return result.flatten()


    def _rand_batch(self):
        indexes = np.random.choice(np.array(range(len(self.X))), size=np.floor(np.sqrt(len(self.X))).astype(int), replace=False)
        return self.X[indexes], self.y[indexes]
