import numpy as np

### 定义二叉特征分裂函数
def feature_split(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_left, X_right])

def linear_kernel(x1,x2):
    return np.dot(x1,x2)

def gaussian_kernel(x1,x2,sigma=5.0):
    return np.exp(-1*np.linalg.norm(x1-x2)**2/(2*(sigma**2)))


def boostrap_sample(x,y,n_estimators):
    x_y=np.concatenate([x,y.reshape(-1,1)],axis=1)
    np.random.shuffle(x_y)
    n_samples=x.shape[0]
    sampling_subsets=[]
    for _ in range(n_estimators):
        idx1=np.random.choice(n_samples,n_samples,replace=True)
        boostrap_xy=x_y[idx1,:]
        boostrap_x=boostrap_xy[:,:-1]
        boostrap_y=boostrap_xy[:,-1]
        sampling_subsets.append([boostrap_x,boostrap_y])
    return sampling_subsets


def euclidean_distance(x,y):
    distance=0
    for i in range(len(x)):
        distance+=pow(x[i]-y[i],2)
    return np.sqrt(distance)



