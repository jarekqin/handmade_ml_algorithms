from handmade_ml_algorithms.basic_class import BaseClass

from collections import Counter

import numpy as np
import cupy as cp


class KNearest(BaseClass):
    def __init__(self, k, model_type='cpu'):
        self._model_type = model_type
        self._k = k

    def train(self, x, x_train):
        num_test = x.shape[0]
        num_train = x_train.shape[0]
        if self._model_type == 'cpu':
            dists = np.zeros((num_test, num_train))
            m = np.dot(x, x_train.T)
            te = np.square(x).sum(axis=1)
            tr = np.square(x_train).sum(axis=1)
            dists = np.sqrt(-2 * m + tr + np.matrix(te).T)
        elif self._model_type == 'gpu':
            dists = cp.zeros((num_test, num_train))
            m = cp.dot(x, x_train.T)
            te = cp.square(x).sum(axis=1)
            tr = cp.square(x_train).sum(axis=1)
            dists = np.sqrt(cp.array(-2 * m.get() + tr.get() + np.matrix(te.get()).T))
        else:
            raise NotImplemented
        return dists

    def predict(self, x, params):
        """
        predict new elements class
        :param x: this should be trained labels
        :param params: this should not be dictionary, but cupy array or numpy array
        :return: predicted labels
        """
        num_test = params.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            if self._model_type == 'cpu':
                labels = x[np.argsort(params[i, :])].flatten()
            elif self._model_type == 'gpu':
                labels = x[cp.argsort(params[i, :])].flatten()
            else:
                raise NotImplemented
            closest_y = labels[0:self._k]
            if self._model_type=='cpu':
                c = Counter(closest_y)
            elif self._model_type=='gpu':
                c=Counter(closest_y.tolist())
            else:
                raise NotImplemented
            y_pred[i] = c.most_common(1)[0][0]
        if self._model_type=='cpu':
            return y_pred
        elif self._model_type=='gpu':
            return cp.array(y_pred)
        else:
            raise NotImplemented


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.utils import shuffle

    iris = datasets.load_iris()
    x, y = shuffle(iris.data, iris.target, random_state=13)
    x = x.astype(np.float32)
    offset = int(x.shape[0] * 0.7)
    x_train, y_train = x[:offset], y[:offset]
    x_test, y_test = x[offset:], y[offset:]
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # cpu version
    k_ = KNearest(10, 'cpu')
    dists = k_.train(x_test, x_train)
    y_test_pred = k_.predict(y_train, dists)
    print(y_test_pred)
    # accuracy
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = np.sum(y_test_pred == y_test)
    print(float(num_correct) / x_test.shape[0])
    print('*' * 100)

    # cuda version
    from cuml.neighbors import KNeighborsClassifier
    neigh=KNeighborsClassifier(num_neighbors=10)
    neigh.fit(x_train,y_train)
    y_pred=neigh.predict(x_test)
    y_pred=y_pred.reshape((-1,1))
    print(float(np.sum(y_pred==y_test))/x_test.shape[0])
    print('*' * 100)

    # gpu version
    x_train, x_test, y_train, y_test = cp.array(x_train), cp.array(x_test), cp.array(y_train), cp.array(y_test)
    k_ = KNearest(10, 'gpu')
    dists = k_.train(x_test, x_train)
    y_test_pred = k_.predict(y_train, dists)
    print(y_test_pred)
    # accuracy
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = cp.sum(y_test_pred == y_test)
    print(float(num_correct) / x_test.shape[0])



