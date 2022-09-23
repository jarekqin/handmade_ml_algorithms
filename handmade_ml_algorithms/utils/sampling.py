import random
from scipy.stats import norm, multivariate_normal

import math


def smooth_dist(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y


def MH_sample(t, sigma):
    pi = [0 for i in range(t)]
    t1 = 0
    while t1 < t - 1:
        t1 += 1
        pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
        alpha = min(1, (smooth_dist(pi_star[0]) / smooth_dist(pi[t1 - 1])))
        u = random.uniform(0, 1)
        if u < alpha:
            pi[t1] = pi_star[0]
        else:
            pi[t1] = pi[t1 - 1]
    return pi


class GibbsSample(object):
    def __init__(self):
        self.target_distribution = multivariate_normal(mean=[5, -1], cov=[[1., 0.5], [0.5, 2.]])

    def p_yx(self, x,mu1, mu2, sigma1, sigma2, rho):
        return random.normalvariate(mu2 + rho * sigma2 / sigma1 * (x - mu1), math.sqrt(1 - rho ** 2) * sigma2)

    def p_xy(self, y, mu1, mu2, sigma1, sigma2, rho):
        return random.normalvariate(mu1 + rho * sigma1 / sigma2 * (y - mu2), math.sqrt(1 - rho ** 2) * sigma1)

    def sampling(self, n, k):
        x_res = []
        y_res = []
        z_res = []
        for i in range(n):
            for j in range(k):
                x = self.p_xy(-1, 5, -1, 1, 2, 0.5)
                y = self.p_yx(x, 5, -1, 1, 2, 0.5)
                z=self.target_distribution.pdf([x,y])
                x_res.append(x)
                y_res.append(y)
                z_res.append(z)
        return x_res,y_res,z_res


if __name__=='__main__':
    import matplotlib.pyplot as plt

    model=GibbsSample()
    x_res,y_res,z_res=model.sampling(10000,50)
    num_bins=50
    plt.hist(x_res,num_bins,facecolor='red',alpha=0.5,label='x')
    plt.hist(y_res, num_bins, facecolor='dodgerblue', alpha=0.5, label='y')
    plt.title('sample histogram of x and y')
    plt.legend()
    plt.show()