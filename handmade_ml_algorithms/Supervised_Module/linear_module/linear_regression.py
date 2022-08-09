import numpy
import numpy as np
import cupy as cp

from handmade_ml_algorithms.basic_class import BaseClass
from handmade_ml_algorithms.loss_function.loss import linear_loss, sigmoid, L1_loss, L2_loss


class SimpleLinearRegression(BaseClass):
    def __init__(self, learning_rate, epochs, model_type='cpu'):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._model_type = model_type

    def initialise(self, dims):
        if self._model_type == 'cpu':
            w = np.zeros(((dims, 1)))
        elif self._model_type == 'gpu':
            w = cp.zeros((dims, 1))
        else:
            raise NotImplemented

        b = 0
        return w, b

    def train(self, x, y):
        loss_his = []
        w, b = self.initialise(x.shape[1])
        for i in range(1, self._epochs):
            y_hat, loss, dw, db = linear_loss(x, y, w, b, self._model_type)
            w += -self._learning_rate * dw
            b += -self._learning_rate * db
            loss_his.append(loss)

            if i % 1e4 == 0:
                print('epoch %d loss %0.4f' % (i, loss))
            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss_his, params, grads

    def predict(self, x, params):
        w = params['w']
        b = params['b']
        if self._model_type == 'cpu':
            y_pred = np.dot(x, w) + b
        else:
            y_pred = cp.dot(x, w) + b
        return y_pred


class LogisticRegression(BaseClass):
    def __init__(self, learning_rate, epochs, model_type='cpu'):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._model_type = model_type

    def initialise(self, dims):
        if self._model_type == 'cpu':
            w = np.zeros((dims, 1))
        elif self._model_type == 'gpu':
            w = cp.zeros((dims, 1))
        else:
            raise NotImplemented
        b = 0
        return w, b

    def logistic_main(self, x, y, w, b):
        num_train = x.shape[0]

        if self._model_type == 'cpu':
            a = sigmoid(np.dot(x, w) + b)
            cost = -1 / num_train * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
            dw = np.dot(x.T, (a - y)) / num_train
            db = np.sum(a - y) / num_train
            cost = np.squeeze(cost)
        elif self._model_type == 'gpu':
            a = sigmoid(cp.dot(x, w) + b)
            cost = -1 / num_train * cp.sum(y * cp.log(a) + (1 - y) * cp.log(1 - a))
            dw = cp.dot(x.T, (a - y)) / num_train
            db = cp.sum(a - y) / num_train
            cost = cp.squeeze(cost)
        return a, cost, dw, db

    def train(self, x, y):
        w, b = self.initialise(x.shape[1])
        cost_list = []
        for i in range(self._epochs):
            a, cost, dw, db = self.logistic_main(x, y, w, b)
            w = w - self._learning_rate * dw
            b = b - self._learning_rate * db

            if i % 100 == 0:
                cost_list.append(cost)
                print('epoch %d cost is 0.4%f' % (i, cost))

        params = {
            'w': w,
            'b': b
        }

        grads = {
            'dw': dw,
            'db': db
        }
        return cost_list, params, grads

    def predict(self, x, params):
        if self._model_type == 'cpu':
            y_pred = sigmoid(np.dot(x, params['w']) + params['b'])
        elif self._model_type == 'gpu':
            y_pred = sigmoid(cp.dot(x, params['w']) + params['b'])
        else:
            raise NotImplemented

        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred


class LassoRidgeRegression(BaseClass):
    def __init__(self, learning_rate, epochs, alpha=0.1,model_name='lasso', model_type='cpu'):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._model_type = model_type
        self._alpha = alpha
        self._model_name=model_name

    def initialise(self, dims):
        if self._model_type == 'cpu':
            w = np.zeros((dims, 1))
        elif self._model_type == 'gpu':
            w = cp.zeros((dims, 1))
        else:
            raise NotImplemented
        b = 0
        return w, b

    def train(self, x, y):
        loss_his = []
        w, b = self.initialise(x.shape[1])
        for i in range(1, self._epochs):
            if self._model_name.lower()=='lasso':
                y_hat, loss, dw, db = L1_loss(x, y, w, b, self._alpha, self._model_type)
            elif self._model_name.lower()=='ridge':
                y_hat, loss, dw, db = L2_loss(x, y, w, b, self._alpha, self._model_type)
            w += -self._learning_rate * dw
            b += -self._learning_rate * db
            loss_his.append(loss)
            if i % 100 == 0:
                print('epoch %d loss %0.4f' % (i, loss))
            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss_his, params, grads

    def predict(self, x, params):
        if self._model_type == 'cpu':
            y_pred = sigmoid(np.dot(x, params['w']) + params['b'])
        elif self._model_type == 'gpu':
            y_pred = sigmoid(cp.dot(x, params['w']) + params['b'])
        else:
            raise NotImplemented

        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred


if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    from sklearn.utils import shuffle
    from handmade_ml_algorithms.modify_tools import r2_score

    # diabetes = load_diabetes()
    # data, target = diabetes.data, diabetes.target
    #
    # x, y = shuffle(data, target, random_state=13)
    # offset = int(x.shape[0] * 0.8)
    # x_train, y_train = x[:offset], y[:offset]
    # x_test, y_test = x[offset:], y[offset:]
    # y_train = y_train.reshape((-1, 1))
    # y_test = y_test.reshape((-1, 1))
    #
    # # cpu test case
    # model_1 = SimpleLinearRegression(0.01, 200000, 'cpu')
    # loss_his, params, grads = model_1.train(x_train, y_train)
    # print(params)
    # y_pred = model_1.predict(x_test, params)
    # print(r2_score(y_test, y_pred, 'cpu'))
    #
    # print('*' * 100)
    # # gpu test case
    # x_train, y_train, x_test, y_test = cp.array(x_train), cp.array(y_train), cp.array(x_test), cp.array(y_test)
    #
    # model_1 = SimpleLinearRegression(0.01, 200000, 'gpu')
    # loss_his, params, grads = model_1.train(x_train, y_train)
    # print(params)
    # y_pred = model_1.predict(x_test, params)
    # print(r2_score(y_test, y_pred, 'gpu'))
    #
    # print('*' * 100)
    # # cuml test case
    # from cuml.linear_model import LinearRegression
    # from cuml.metrics import mean_absolute_error, r2_score as r2
    #
    # regr = LinearRegression()
    # regr.fit(x_train, y_train)
    # y_pred = regr.predict(x_test)
    #
    # print(mean_absolute_error(y_test, y_pred))
    # print(r2(y_test, y_pred))
    #
    # print('*' * 100)
    # # logistic regression case
    # # cpu case
    # from sklearn.datasets._samples_generator import make_classification
    # from sklearn.metrics import classification_report
    #
    # x, labels = make_classification(
    #     n_samples=100,
    #     n_features=2,
    #     n_redundant=0,
    #     n_informative=2,
    #     random_state=1,
    #     n_clusters_per_class=2
    # )
    # rng = np.random.RandomState(2)
    #
    # offset = int(x.shape[0] * 0.9)
    # x_train, y_train = x[:offset], labels[:offset]
    # x_test, y_test = x[offset:], labels[offset:]
    # y_train = y_train.reshape((-1, 1))
    # y_test = y_test.reshape((-1, 1))
    #
    # model_ = LogisticRegression(0.01, 1000, 'cpu')
    # cost_list, params, grads = model_.train(x_train, y_train)
    # y_pred = model_.predict(x_test, params)
    # print(y_pred)
    # print(classification_report(y_test, y_pred))
    # print('*' * 100)
    # # gpu case
    # x_train, y_train, x_test, y_test = cp.array(x_train), cp.array(y_train), cp.array(x_test), cp.array(y_test)
    # model_ = LogisticRegression(0.01, 1000, 'gpu')
    # cost_list, params, grads = model_.train(x_train, y_train)
    # y_pred = model_.predict(x_test, params)
    # print(y_pred)
    # print(classification_report(y_test.get(), y_pred.get()))
    # print('*' * 100)
    #
    # # cuml case
    # from cuml.linear_model import LogisticRegression
    #
    # clf = LogisticRegression().fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print(y_pred)

    # lasso regression cpu version
    print('*' * 100)
    data = np.random.random([101, 101])
    label = np.random.choice([0, 1], size=101)

    x_train, y_train = data[:70], label[:70].reshape(-1,1)
    x_test, y_test = data[70:], label[70:].reshape(-1,1)

    lasso_ = LassoRidgeRegression(0.01, 300)
    loss_list, params, grads = lasso_.train(x_train, y_train)
    print(params)

    print('*' * 100)

    # lasso regression gpu version
    data = cp.random.random([101, 101])
    label = cp.random.choice([0, 1], size=101).reshape(-1, 1)

    x_train, y_train = data[:70], label[:70].reshape(-1,1)
    x_test, y_test = data[70:], label[70:].reshape(-1,1)

    lasso_ = LassoRidgeRegression(0.01, 300, model_type='gpu')
    loss_list, params, grads = lasso_.train(x_train, y_train)
    print(params)
    print('*' * 100)

    # ridge regression cpu version
    print('*' * 100)
    data = np.random.random([101, 101])
    label = np.random.choice([0, 1], size=101)

    x_train, y_train = data[:70], label[:70].reshape(-1,1)
    x_test, y_test = data[70:], label[70:].reshape(-1,1)

    lasso_ = LassoRidgeRegression(0.01, 300,model_name='ridge')
    loss_list, params, grads = lasso_.train(x_train, y_train)
    print(params)

    print('*' * 100)

    # ridge regression gpu version
    data = cp.random.random([101, 101])
    label = cp.random.choice([0, 1], size=101).reshape(-1, 1)

    x_train, y_train = data[:70], label[:70].reshape(-1,1)
    x_test, y_test = data[70:], label[70:].reshape(-1,1)

    lasso_ = LassoRidgeRegression(0.01, 300, model_name='ridge',model_type='gpu')
    loss_list, params, grads = lasso_.train(x_train, y_train)
    print(params)
    print('*' * 100)

    # gpu sklearn version
    from cuml.linear_model import Ridge,Lasso
    clf=Ridge(alpha=1.0)
    clf.fit(x_train,y_train)
    print(clf.coef_)

    lasso=Lasso(alpha=0.1)
    lasso.fit(x_train,y_train)
    print(lasso.coef_)
    print(lasso.intercept_)
