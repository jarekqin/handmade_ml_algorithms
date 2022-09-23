import numpy as np

from handmade_ml_algorithms.basic_class import BaseClass


class EM(BaseClass):
    def __init__(self, max_iter, eps):
        self.max_iter = max_iter
        self.eps = eps

    def train(self, x=None, y=None):
        ll_old = 0
        for i in range(self.max_iter):
            log_like = np.array([np.sum(x * np.log(yy), axis=1) for yy in y])
            like = np.exp(log_like)
            ws = like / like.sum(0)
            vs = np.array([w[:, None] * x for w in ws])
            y = np.array([v.sum(0) / v.sum() for v in vs])
            ll_new = np.sum([w * l for w, l in zip(ws, log_like)])
            print('Iteration: %d' % (i + 1))
            print('theta_b=%.2f, theta_c=%.2f, ll=%.2f' % (y[0, 0], y[1, 0], ll_new))
            if np.abs(ll_new - ll_old) < self.eps:
                break
            ll_old = ll_new
        return y


if __name__ == '__main__':
    data_ = np.array([[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]])
    thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
    model=EM(30,1e-3)
    print(model.train(data_,thetas))