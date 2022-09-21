from handmade_ml_algorithms.basic_class.tree_basic import BinaryDecisionTree, BaseClass
from handmade_ml_algorithms.loss_function.loss import calc_gini
from handmade_ml_algorithms.utils import boostrap_sample

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


class RandomForest(BaseClass):
    def __init__(self, n_estimators=100, min_samples_split=2, min_gini_impurity=0,
                 max_depth=float('inf'), max_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gini_impurity = min_gini_impurity
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        for _ in range(self.n_estimators):
            tree = CARTClassificationTree(min_samples_split=self.min_samples_split,
                                          min_gini_impurity=self.min_gini_impurity,
                                          max_depth=self.max_depth)
            self.trees.append(tree)

    def train(self, x, y):
        sub_sets = boostrap_sample(x, y, self.n_estimators)
        n_features = x.shape[1]

        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            sub_x, sub_y = sub_sets[i]
            idx2 = np.random.choice(n_features, self.max_features, replace=True)
            sub_x = sub_x[:, idx2]
            self.trees[i].train(sub_x, sub_y.reshape([sub_y.shape[0], -1]))
            self.trees[i].feature_indices = idx2
            print('The %sth tree is trained done...' % str(i + 1))

    def predict(self, x):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_x = x[:, idx]
            y_pred = self.trees[i].predict(sub_x)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds).T
        res = []
        for j in y_preds:
            res.append(np.bincount(j.astype('int')).argmax())
        return res


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score, mean_squared_error
    #
    # data = datasets.load_iris()
    # x, y = data.data, data.target
    # y = y.reshape((-1, 1))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # clf = CARTClassificationTree()
    # clf.train(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print('accuracy:', accuracy_score(y_test, y_pred))
    #
    # print('*' * 100)
    # x, y = datasets.load_boston(return_X_y=True)
    # y = y.reshape((-1, 1))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # reg = CARTRegressionTree()
    # reg.train(x_train, y_train)
    # y_pred = reg.predict(x_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print('accuracy: ', mse)

    print('*' * 100)
    x, y = datasets.make_classification(1000, 20, n_repeated=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    x += 2 * rng.uniform(size=x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # rf = RandomForest(n_estimators=10, max_features=15)
    # rf.train(x_train, y_train)
    # y_pred = rf.predict(x_test)
    # print(accuracy_score(y_test, y_pred))

    from cuml.ensemble import RandomForestClassifier
    from cudf import cuda
    import cupy as cp
    cuda.select_device(1)
    print(cuda.get_current_device().id,cuda.get_current_device().name)
    # 此处的n_streams必须为1，否则可能由于随即森林算法在GPU内的线程异步执行原因，造成堵塞出错
    clf=RandomForestClassifier(max_depth=3,random_state=0, n_streams=1,verbose=True)
    # 使用cuml模型进行训练，所有数据最好转换为cp.asarray或者cudf.dataframe放到gpu上去执行
    clf.fit(cp.asarray(x_train).astype(cp.float32),cp.asarray(y_train).astype(cp.float32))
    y_pred=clf.predict(cp.asarray(x_test).astype(cp.float32))
    print(accuracy_score(y_test,y_pred.get()))
