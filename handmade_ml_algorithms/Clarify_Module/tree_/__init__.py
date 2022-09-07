from handmade_ml_algorithms.basic_class.tree_basic import BinaryDecisionTree, DecisionStump
from handmade_ml_algorithms.loss_function.loss import calc_gini

from handmade_ml_algorithms.basic_class import BaseClass

import numpy as np


class CARTClassificationTree(BinaryDecisionTree):

    def _calculate_gini_impurity(self, y, y1, y2):
        p = len(y1) / len(y)
        gini = calc_gini(y)
        gini_impurity = p * calc_gini(y1) + (1 - p) * calc_gini(y2)
        return gini_impurity

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def train(self, x, y):
        self.gini_impurity_calc = self._calculate_gini_impurity
        self.leaf_value_calc = self._majority_vote
        super(CARTClassificationTree, self).train(x, y)


class CARTRegressionTree(BinaryDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = np.var(y, axis=0)
        var_y1 = np.var(y1, axis=0)
        var_y2 = np.var(y2, axis=0)

        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        variance_reduction = var_tot - (frac_1 * var_y1 + frac_2 * var_y2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def train(self, x, y):
        self.gini_impurity_calc = self._calculate_variance_reduction
        self.leaf_value_calc = self._mean_of_y
        super(CARTRegressionTree, self).train(x, y)


class DecisionStump():
    def __init__(self):
        self.label = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None


class Adaboost(BaseClass):
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators

    def train(self, x, y):
        m, n = x.shape
        w = np.full(m, (1 / m))
        self.estimators=[]
        for _ in range(self.n_estimators):
            estimator = DecisionStump()
            min_error = float('inf')
            for i in range(n):
                values = np.expand_dims(x[:, i], axis=1)
                unique_values = np.unique(values)
                for threshold in unique_values:
                    p = 1
                    pred = np.ones(np.shape(y))
                    pred[x[:, i] < threshold] = -1
                    error = sum(w[y != pred])
                    if error>0.5:
                        error=1-error
                        p=-1
                    if error<min_error:
                        estimator.label=p
                        estimator.threshold=threshold
                        estimator.feature_index=i
                        min_error=error
            estimator.alpha=0.5*np.log((1.0-min_error)/(min_error+1e-9))
            preds=np.ones(np.shape(y))
            negative_idx = (estimator.label * x[:, estimator.feature_index] < estimator.label * estimator.threshold)
            preds[negative_idx]=-1
            w*=np.exp(-estimator.alpha*y*preds)
            w/=np.sum(w)
            self.estimators.append(estimator)

    def predict(self,x,params=None):
        m=len(x)
        y_pred=np.zeros((m,1))
        for estimator in self.estimators:
            predictions=np.ones(np.shape(y_pred))
            negative_idx = (estimator.label * x[:, estimator.feature_index] < estimator.label * estimator.threshold)
            predictions[negative_idx]=-1
            y_pred+=estimator.alpha*predictions
        y_pred=np.sign(y_pred).flatten()
        return y_pred

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score, mean_squared_error

    data = datasets.load_iris()
    x, y = data.data, data.target
    y = y.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = CARTClassificationTree()
    clf.train(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('accuracy:', accuracy_score(y_test, y_pred))

    print('*' * 100)
    x, y = datasets.load_boston(return_X_y=True)
    y = y.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    reg = CARTRegressionTree()
    reg.train(x_train, y_train)
    y_pred = reg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('accuracy: ', mse)

    from sklearn.model_selection import train_test_split
    from sklearn.datasets._samples_generator import make_blobs
    x,y=make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.2,random_state=40)
    y_=y.copy()
    y_[y_==0]=-1
    y_=y_.astype(float)
    x_train,x_test,y_train,y_test=train_test_split(x,y_,test_size=0.3,random_state=43)
    adboost=Adaboost()
    adboost.train(x_train,y_train)
    y_pred=adboost.predict(x_test)
    print('accuracy: ',accuracy_score(y_test,y_pred))