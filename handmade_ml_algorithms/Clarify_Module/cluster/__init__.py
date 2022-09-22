import numpy as np

from collections import Counter

from handmade_ml_algorithms.utils import euclidean_distance
from handmade_ml_algorithms.basic_class import BaseClass


class KMean(BaseClass):
    def __init__(self, x, k, max_iterations):
        self.x = x
        self.k = k
        self.max_iterations = max_iterations

    def centroids_init(self):
        m, n = self.x.shape
        self.centroids = np.zeros((self.k, n))
        for i in range(self.k):
            centroid = self.x[np.random.choice(range(m))]
            self.centroids[i] = centroid

    def closest_centroid(self,x):
        if len(self.centroids) == 0:
            raise ValueError('centroids has not been calculated!')
        closest_i, closest_dist = 0, float('inf')
        for i, centroid in enumerate(self.centroids):
            distance = euclidean_distance(x, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def build_clusters(self):
        self.clusters=[[] for _ in range(self.k)]
        for x_i,x in enumerate(self.x):
            centroid_i=self.closest_centroid(x)
            self.clusters[centroid_i].append(x_i)

    def calculate_centroids(self):
        n=self.x.shape[1]
        self.centroids=np.zeros((self.k,n))
        for i,cluster in enumerate(self.clusters):
            centroid=np.mean(self.x[cluster],axis=0)
            self.centroids[i]=centroid

    def train(self,x=None,y=None):
        if x is None:
            x=self.x
        if y is None:
            y=self.clusters

        y_pred=np.zeros(x.shape[0])
        for cluster_i,cluster in enumerate(y):
            for sample_i in cluster:
                y_pred[sample_i]=cluster_i
        return y_pred


    def get_trained_labels(self):
        self.centroids_init()
        for _ in range(self.max_iterations):
            self.build_clusters()
            self.calculate_centroids()
            cur_centroids=self.centroids
            self.calculate_centroids()
            diff=self.centroids-cur_centroids
            if not diff.any():
                break
        return self.train(None,None)


class KNearest(BaseClass):
    def __init__(self, k):
        self._k = k

    def train(self, x, x_train):
        num_test = x.shape[0]
        num_train = x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        m = np.dot(x, x_train.T)
        te = np.square(x).sum(axis=1)
        tr = np.square(x_train).sum(axis=1)
        dists = np.sqrt(-2 * m + tr + np.matrix(te).T)
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
            labels = x[np.argsort(params[i, :])].flatten()
            closest_y = labels[0:self._k]
            c = Counter(closest_y)

            y_pred[i] = c.most_common(1)[0][0]
        return y_pred

class PCA(BaseClass):
    def __init__(self,x):
        self.x=x

    def cal_cov(self):
        m=self.x.shape[0]
        self.x=(self.x-np.mean(self.x,axis=0)/np.var(self.x,axis=0))
        return 1/m*np.matmul(self.x.T,self.x)

    def train(self,x,y):
        cov_matrix=self.cal_cov()
        eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)
        idx=eigenvalues.argsort()[::-1]
        eigenvectors=eigenvectors[:,idx]
        eigenvectors=eigenvectors[:,:y]
        return np.matmul(self.x,eigenvectors)



if __name__=='__main__':
    x=np.array([[0,2],[0,0],[1,0],[5,0],[5,2]])

    model=KMean(x,2,10)
    print(model.get_trained_labels())

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
    k_ = KNearest(10)
    dists = k_.train(x_test, x_train)
    y_test_pred = k_.predict(y_train, dists)
    print(y_test_pred)
    # accuracy
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = np.sum(y_test_pred == y_test)
    print(float(num_correct) / x_test.shape[0])
    print('*' * 100)

    iris=datasets.load_iris()
    x,y=iris.data,iris.target
    model=PCA(x)
    print(model.train(x,3))
    print('*' * 100)

    from cuml.decomposition import PCA
    pca_=PCA(n_components=3)
    pca_.fit(x)
    print(pca_.transform(x))


