import cupy as cp
import numpy as np


def linear_loss(x, y, w, b, use_type='cpu'):
    """
    linear loss function gpu version
    :param x: params matrix
    :param y: labels
    :param w: params weights
    :param b: params biases
    :param use_type: host or device used for calculation
    :return:
        y_hat: predicted value
        loss: loss value
        dw: 1st derivative weights
        db: 1st derivative bias
    """
    num_train = x.shape[0]
    if use_type == 'gpu':
        y_hat = cp.dot(x, w) + b
        loss = cp.sum((y_hat - y) ** 2) / num_train
        dw = cp.dot(x.T, (y_hat - y)) / num_train

        db = cp.sum((y_hat - y)) / num_train
    elif use_type == 'cpu':
        y_hat = np.dot(x, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train
        dw = np.dot(x.T, (y_hat - y)) / num_train

        db = np.sum((y_hat - y)) / num_train
    else:
        raise NotImplemented

    return y_hat, loss, dw, db


def sigmoid(x, use_type='cpu'):
    """
    sigmoid loss function
    :param x: numpy array or cupy array
    :param use_type: cpu or gpu
    :return: sigmoid loss value
    """
    if use_type == 'cpu':
        z = 1 / (1 + np.exp(-x))
    elif use_type == 'gpu':
        z = 1 / (1 + cp.exp(-x))
    else:
        raise NotImplemented
    return z


def sign(x):
    return 1 if x > 0 else 0


def L1_loss(x, y, w, b, alpha, use_type='cpu'):
    """
    L1 loss functin
    :param x: observed data
    :param y: labels
    :param w: weights
    :param b: bias for data
    :param alpha: punishment parameters
    :return: loss value
    """
    num_train = x.shape[0]
    if use_type == 'cpu':
        nec_sign=np.vectorize(sign)
        y_hat = np.dot(x, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(alpha * abs(w))
        dw = np.dot(x.T, (y_hat - y)) / num_train + alpha * nec_sign(w)
        db = np.sum((y_hat - y)) / num_train
    elif use_type == 'gpu':
        nec_sign = cp.vectorize(sign)
        y_hat = cp.dot(x, w) + b
        loss = cp.sum((y_hat - y) ** 2) / num_train + cp.sum(alpha * abs(w))
        dw = cp.dot(x.T, (y_hat - y)) / num_train + alpha * nec_sign(w)
        db = cp.sum((y_hat - y)) / num_train
    else:
        raise NotImplemented

    return y_hat,loss,dw,db


def L2_loss(x, y, w, b, alpha, use_type='cpu'):
    """
    L2 loss functin
    :param x: observed data
    :param y: labels
    :param w: weights
    :param b: bias for data
    :param alpha: punishment parameters
    :return: loss value
    """
    num_train = x.shape[0]
    if use_type == 'cpu':
        nec_sign = np.vectorize(sign)
        y_hat = np.dot(x, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + alpha*(np.sum(np.square(w)))
        dw = np.dot(x.T, (y_hat - y)) / num_train + alpha * 2 * w
        db = np.sum((y_hat - y)) / num_train
    elif use_type == 'gpu':
        y_hat = cp.dot(x, w) + b
        loss = cp.sum((y_hat - y) ** 2) / num_train + alpha*(cp.sum(cp.square(w)))
        dw = cp.dot(x.T, (y_hat - y)) / num_train + alpha * 2 * w
        db = cp.sum((y_hat - y)) / num_train
    else:
        raise NotImplemented

    return y_hat, loss, dw, db


def entropy(x,use_type='cpu'):
    if use_type=='cpu':
        probs = [list(x).count(i) / len(x) for i in set(x)]
        return sum([prob*np.log2(prob) for prob in probs])
    elif use_type=='gpu':
        probs = [list(x).count(i) / len(x.get()) for i in set(x.get())]
        return sum([prob*cp.log2(prob) for prob in probs]).tolist()
    else:
        raise NotImplemented


def calc_gini(x,use_type='cpu'):
    if use_type=='cpu':
        probs = [list(x).count(i) / len(x) for i in np.unique(x)]
        return sum([p*(1-p) for p in probs])
    elif use_type=='gpu':
        probs = [list(x).count(i) / len(x.get()) for i in cp.unique(x)]
        return sum([p*(1-p) for p in probs])
    else:
        raise NotImplemented


if __name__=='__main__':
    test_=np.array([0.1,0.2,0.3])
    print(entropy(test_))

    test_2=cp.array([0.1,0.2,0.3])
    print(entropy(test_2,'gpu'))


    print(gini(test_))
    print(gini(test_2,'gpu'))

