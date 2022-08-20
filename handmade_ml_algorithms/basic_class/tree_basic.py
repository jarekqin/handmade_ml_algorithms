from handmade_ml_algorithms.utils import feature_split
from handmade_ml_algorithms.basic_class import BaseClass

import numpy as np


class TreeNode(object):
    def __init__(self, feature_x=None, threshold=None, leaf_value=None, left_branch=None, right_branch=None):
        self.feature_x = feature_x
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.left_branch = left_branch
        self.right_branch = right_branch





class BinaryDecisionTree(BaseClass):
    def __init__(self, min_samples_split=3, min_gini_impurity=999, max_depth=float('inf'), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_gini_impurity = min_gini_impurity
        self.max_depth = max_depth
        self.loss = loss
        self.leaf_value_calc = None
        self.gini_impurity_calc = None

    def train(self, x, y):
        self.root = self.initialise(x, y)
        self.loss = None

    def initialise(self, dim):
        pass

    def initialise(self, x, y, current_depth=0):
        init_gini_impurity = 999
        best_criteria = None
        best_sets = None

        xy = np.concatenate((x, y), axis=1)
        m, n = x.shape
        if m >= self.min_samples_split and current_depth <= self.max_depth:
            # circling every features for cal gini_impurity
            for f_i in range(n):
                f_values = np.expand_dims(x[:, f_i], axis=1)
                unique_values = np.unique(f_values)

                # circling every best criteria
                for threshold in unique_values:
                    xy1, xy2 = feature_split(xy, f_i, threshold)
                    # if sub_set <0
                    if len(xy1) != 0 and len(xy2) != 0:
                        y1 = xy1[:, n:]
                        y2 = xy2[:, n:]

                        impurity = self.gini_impurity_calc(y, y1, y2)
                        # get smallest impurity
                        # index of best features and thresholds
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {'f_i': f_i, "threshold": threshold}
                            best_sets = {
                                'leftx': xy1[:, :n],
                                'lefty': xy1[:, n:],
                                'rightx': xy2[:, :n],
                                'righty': xy2[:, n:]
                            }
        if init_gini_impurity < self.min_gini_impurity:
            left_branch = self.initialise(best_sets['leftx'], best_sets['lefty'], current_depth + 1)
            right_branch = self.initialise(best_sets['rightx'], best_sets['righty'], current_depth + 1)
            return TreeNode(feature_x=best_criteria['f_i'], threshold=best_criteria['threshold'],
                            left_branch=left_branch, right_branch=right_branch)

        leaf_value = self.leaf_value_calc(y)
        return TreeNode(leaf_value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.leaf_value is not None:
            return tree.leaf_value
        feature_value = x[tree.feature_x]

        branch = tree.right_branch
        if feature_value >= tree.threshold:
            branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch
        return self.predict_value(x, branch)

    def predict(self, x, params=None):
        y_pred = [self.predict_value(sample) for sample in x]
        return y_pred


if __name__ == '__main__':
    tree = TreeNode()
