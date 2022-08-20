import numpy as np
import cupy as cp

from handmade_ml_algorithms.basic_class import BaseClass


class LDA(BaseClass):
    def __init__(self, model_type='cpu'):
        self._model_type = model_type

    def initialise(self, dims=None):
        self.w = None

    def cal_cov(self, x, y=None):
        m = x.shape[0]
        if self._model_type == 'cpu':
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = x if y == None else (y - np.mean(y, axis=0)) / np.std(y, axis=0)
            result = 1 / m * np.matmul(x.T, y)
        elif self._model_type == 'gpu':
            x = (x - cp.mean(x, axis=0)) / cp.std(x, axis=0)
            y = x if y == None else (y - cp.mean(y, axis=0)) / cp.std(y, axis=0)
            result = 1 / m * cp.matmul(x.T, y)
        else:
            raise NotImplemented

        return result

    def train(self, x, y):
        x0 = x[y == 0]
        x1 = x[y == 1]
        sigma0 = self.cal_cov(x0)
        sigma1 = self.cal_cov(x1)
        sw = sigma0 + sigma1
        if self._model_type == 'cpu':
            u0, u1 = np.mean(x0, axis=0), np.mean(x1, axis=0)
            mean_diff = np.atleast_1d(u0 - u1)
            u, s, v = np.linalg.svd(sw)
            sw_ = np.dot(np.dot(v.T, np.linalg.pinv(np.diag(s))), u.T)
            self.w = sw_.dot(mean_diff)
        elif self._model_type == 'gpu':
            u0, u1 = cp.mean(x0, axis=0), cp.mean(x1, axis=0)
            mean_diff = cp.atleast_1d(u0 - u1)
            u, s, v = cp.linalg.svd(sw)
            sw_ = cp.dot(cp.dot(v.T, cp.linalg.pinv(cp.diag(s))), u.T)
            self.w = sw_.dot(mean_diff)
        else:
            raise NotImplemented

    def predict(self, x, params=None):
        y_pred = []
        if self._model_type=='cpu':
            for i in x:
                h = i.dot(self.w)
                y = 1 * (h < 0)
                y_pred.append(y)
        elif self._model_type=='gpu':
            for i in x:
                h = i.dot(self.w)
                y = 1 * (h < 0)
                y_pred.append(y.get().item())
        else:
            raise NotImplemented
        return y_pred


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = datasets.load_iris()
    x, y = data.data, data.target
    x = x[y != 2]
    y = y[y != 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)
    # cpu version
    lda = LDA()
    lda.train(x_train, y_train)
    y_pred = lda.predict(x_test, None)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    # gpu version
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)
    x_train, x_test, y_train, y_test = cp.array(x_train), cp.array(x_test), cp.array(y_train), cp.array(y_test)
    lda = LDA('gpu')
    lda.train(x_train, y_train)
    y_pred = lda.predict(x_test, None)
    acc = accuracy_score(y_test.get(), y_pred)
    print(acc)
