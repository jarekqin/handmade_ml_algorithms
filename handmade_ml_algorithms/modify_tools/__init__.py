import numpy as np
import cupy as cp

def r2_score(y_test,y_pred,use_type='cpu'):
    """
    R square score testing
    :param y_test: y testing data
    :param y_pred: y predicted data
    :param use_type: cpu or gpu
    :return: score
    """
    if use_type=='gpu':
        y_avg=cp.mean(y_test)
        ss_tot=cp.sum((y_test-y_avg)**2)
        ss_res=cp.sum((y_test-y_pred)**2)

    elif use_type=='cpu':
        y_avg=np.mean(y_test)
        ss_tot=np.sum((y_test-y_avg)**2)
        ss_res=np.sum((y_test-y_pred)**2)
    return 1 - (ss_res / ss_tot)