import numpy as np
from sklearn.preprocessing import FunctionTransformer
def dummy(x):
    return x


def preprocess_tokenized(lowercase, x):
    if lowercase:
        x = [w.lower() for w in x]
    return x

def removeMinus(X, y):
    indices = np.where(y >= 0)
    x = X[indices]
    y = y[indices]
    print("After removing zeros: ", x.shape, y.shape)

    return x, y

def reshape(x):
    return x.to_numpy().reshape(-1, 1)

pandas_reshaper = FunctionTransformer(reshape, validate=False)


def to_pandas(x):
    return x.to_pandas()

def to_nparray(x):
    return np.array(x)
