from mxnet import nd


def synthetic_data(w, b, num_examples):
    # y = Xw + b + noise
    X = nd.random.normal(scale=1, shape=(num_examples, len(w)))
    y = nd.dot(X, w) + b
    y += nd.random.normal(scale=0.1, shape=y.shape)
    return X, y
