from handmade_ml_algorithms.basic_class.tree_basic import BinaryDecisionTree
from handmade_ml_algorithms.loss_function.loss import calc_gini

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

    def _mean_of_y(self,y):
        value=np.mean(y,axis=0)
        return value if len(value)>1 else value[0]

    def train(self, x, y):
        self.gini_impurity_calc=self._calculate_variance_reduction
        self.leaf_value_calc=self._mean_of_y
        super(CARTRegressionTree,self).train(x,y)



if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score,mean_squared_error

    data = datasets.load_iris()
    x, y = data.data, data.target
    y = y.reshape((-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = CARTClassificationTree()
    clf.train(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('accuracy:', accuracy_score(y_test, y_pred))

    print('*' * 100)
    x,y=datasets.load_boston(return_X_y=True)
    y=y.reshape((-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    reg=CARTRegressionTree()
    reg.train(x_train,y_train)
    y_pred=reg.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    print('accuracy: ',mse)

