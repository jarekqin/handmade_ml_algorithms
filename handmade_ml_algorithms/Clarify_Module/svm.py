import numpy as np

from handmade_ml_algorithms.basic_class import BaseClass
from handmade_ml_algorithms.utils import linear_kernel,gaussian_kernel

from cvxopt import matrix, solvers


class SoftSVM(BaseClass):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def train(self, x, y):
        m, n = x.shape

        # 基于先行核计算的gram矩阵
        k = self._gram_matrix(x)

        # 二次规划相关初始化

        p = matrix(np.outer(y, y) * k)
        q = matrix(np.ones(m) * -1)
        a = matrix(y, (1, m))
        b = matrix(0.0)

        if self.C is None:
            g = matrix(np.diag(np.ones(m) * -1))
            h = matrix(np.zeros(m))
        else:
            tmp1 = np.diag(np.ones(m) * -1)
            tmp2 = np.identity(m)
            g = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(m)
            tmp2 = np.ones(m) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        sol = solvers.qp(p, q, g, h, a, b)
        a2 = np.ravel(sol['x'])

        # 寻找支持响亮
        spv = a2 > 1e-5
        ix = np.arange(len(a2))[spv]
        self.a = a2[spv]
        self.spv = x[spv]
        self.spv_y = y[spv]
        print('%d support vectors out of %d points' % (len(self.a), m))

        self.b = 0
        for i in range(len(self.a)):
            self.b += self.spv_y[i]
            self.b -= np.sum(self.a * self.spv_y * k[ix[i], spv])
        self.b /= len(self.a)

        self.w = np.zeros(n, )
        for i in range(len(self.a)):
            self.w += self.a[i] * self.spv_y[i] * self.spv[i]

    def _gram_matrix(self, x):
        m, n = x.shape
        k = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                k[i, j] = self.kernel(x[i], x[j])
        return k

    def project(self, x):
        if self.w is not None:
            return np.dot(x, self.w) + self.b

    def predict(self, x, params=None):
        return np.sign(np.dot(self.w, x.T)) + self.b


class Non_Linear_SVM(BaseClass):
    def __init__(self,kernel=gaussian_kernel):
        self.kernel=kernel

    def train(self,x,y):
        m,n=x.shape

        k=self._gram_matrix(x)

        p = matrix(np.outer(y, y) * k)
        q = matrix(np.ones(m) * -1)
        a = matrix(y, (1, m))
        b = matrix(0.0)
        g=matrix(np.diag(np.ones(m)*-1))
        h=matrix(np.zeros(m))

        sol=solvers.qp(p,q,g,h,a,b)
        a2=np.ravel(sol['x'])

        spv=a2>1e-5
        ix=np.arange(len(a2))[spv]
        self.a2=a2[spv]
        self.spv=x[spv]
        self.spv_y=y[spv]
        print('%d support vectors out of %d' % (len(self.a2),m))

        self.b=0
        for i in range(len(self.a2)):
            self.b+=self.spv_y[i]
            self.b-=np.sum(self.a2*self.spv_y*k[ix[i],spv])

        self.b/=len(self.a2)

        self.w=None

    def _gram_matrix(self,x):
        m,n=x.shape
        k=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                k[i,j]=self.kernel(x[i],x[j])
        return k

    def project(self,x):
        y_pred=np.zeros(len(x))
        for i in range(x.shape[0]):
            s=0
            for a, spv_y,spv in zip(self.a2,self.spv_y,self.spv):
                s+=a*spv_y*self.kernel(x[i],spv)
            y_pred[i]=s
        return y_pred+self.b

    def predict(self,x,params=None):
        return np.sign(self.project(x))



if __name__ == '__main__':
    from sklearn.metrics import accuracy_score

    mean1, mean2 = np.array([0, 2]), np.array([2, 0])
    covar = np.array([[1.5, 1.0], [1.0, 1.5]])
    x1 = np.random.multivariate_normal(mean1, covar, 100)
    y1 = np.ones(x1.shape[0])
    x2 = np.random.multivariate_normal(mean2, covar, 100)
    y2 = -1 * np.ones(x2.shape[0])
    x_train = np.vstack((x1[:80], x2[:80]))
    y_train = np.hstack((y1[:80], y2[:80]))
    x_test = np.vstack((x1[80:], x2[80:]))
    y_test = np.hstack((y1[80:], y2[80:]))

    soft_margin_svm = SoftSVM(C=0.1)
    soft_margin_svm.train(x_train, y_train)
    y_pred = soft_margin_svm.predict(x_test)
    y_pred=[1 if x>0 else -1 for x in y_pred]
    print(accuracy_score(y_test,y_pred))

    # gpu
    from cuml.svm import LinearSVC
    from cuml.metrics import accuracy_score as as2
    clf=LinearSVC()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print(as2(y_test,y_pred))

    # 核函数部分

    mean1, mean2 = np.array([-1, 2]), np.array([1, -1])
    mean3, mean4 = np.array([4, -4]), np.array([-4, 4])
    covar = np.array([[1.0, 0.8], [0.8, 1.0]])
    x1 = np.random.multivariate_normal(mean1, covar, 50)
    x1=np.vstack((x1,np.random.multivariate_normal(mean3,covar,50)))
    y1 = np.ones(x1.shape[0])
    x2 = np.random.multivariate_normal(mean2, covar, 50)
    x2 = np.vstack((x2, np.random.multivariate_normal(mean4, covar, 50)))
    y2 = -1 * np.ones(x2.shape[0])
    x_train = np.vstack((x1[:80], x2[:80]))
    y_train = np.hstack((y1[:80], y2[:80]))
    x_test = np.vstack((x1[80:], x2[80:]))
    y_test = np.hstack((y1[80:], y2[80:]))

    # Non_Linear_SVM
    model=Non_Linear_SVM()
    model.train(x_train,y_train)
    y_pred=model.predict(x_test)
    print(accuracy_score(y_test,y_pred))

    # gpu
    from sklearn import svm

    clf=svm.SVC(kernel='rbf')
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print(as2(y_test,y_pred))
