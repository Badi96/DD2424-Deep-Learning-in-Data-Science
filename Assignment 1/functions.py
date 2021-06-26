import numpy as np
import matplotlib.pyplot as plt
import pickle

#def softmax(x):
#    #""" Standard definition of the softmax function """
#    #return np.exp(x) / np.sum(np.exp(x), axis=0)
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=0)


# Added to instead of one_hot inside LoadBatch, for testing.
def one_hot(labels, number_distinct_labels=10):
    labels_one_hot = np.zeros(shape=(number_distinct_labels, len(labels)))
    for i, label in enumerate(labels):
        labels_one_hot[label, i] = 1

    return labels_one_hot


def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('Datasets/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        #X = dict[b"data"].T
        X = dict[b"data"].T
        y = dict[b"labels"]
        Y = (np.eye(10)[y]).T
    return X, Y, y


#return dict


def ComputeGradsNum(X, Y, P, W, b, lamda, h=0.000001):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


def montage(W):
    """ Display the image for each label in W """

    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()


#def save_as_mat(data, name="model"):
#	""" Used to transfer a python model to matlab """
#	import scipy.io as sio
#	sio.savemat(name'.mat',{name:b})


# LoadBatch already in the function.py file!
def unpickle(file):
    import pickle
    with open('Datasets/' + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


batches = unpickle('cifar-10-batches-py/batches.meta')
label_names = [
    label_name.decode('utf-8') for label_name in batches[b'label_names']
]
