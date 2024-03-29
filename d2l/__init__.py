from IPython import display
from matplotlib import pyplot as plt

import sys

from mxnet import gluon, autograd, nd
from d2l import Accumulator
from d2l import Animator


def use_svg_display():
    # Use the svg format to display plot in jupyter
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    # Change the default figure size
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype("float32")).sum().asscalar()


def evaluate_accuracy(net, data_iter):
    metric = Accumulator.Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator.Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(l.sum().asscalar(), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                 ylim=[0.3, 0.9],
                                 legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

def predict_ch3(net, test_iter, n = 6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y.asnumpy())
    preds = get_fashion_mnist_labels(net(X).argmax(axis = 1).asnumpy())
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape(n,28,28), 1, n, titles = titles[0:n])


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(3.5, 2.5), axes=None):
    """Plot multiple lines"""
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    if isinstance(X, nd.NDArray): X = X.asnumpy()
    if isinstance(Y, nd.NDArray): Y = Y.asnumpy()
    if not hasattr(X[0], "__len__"): X = [X]
    if Y is None: X, Y = [[]]*len(X), X
    if not hasattr(Y[0], "__len__"): Y = [Y]
    if len(X) != len(Y): X = X * len(Y)
    if not fmts: fmts = ['-']*len(X)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if isinstance(x, nd.NDArray): x = x.asnumpy()
        if isinstance(y, nd.NDArray): y = y.asnumpy()
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
