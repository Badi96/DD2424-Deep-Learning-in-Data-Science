import numpy as np
import pickle
#from functions import *
import sys
import matplotlib.pyplot as plt
import random


def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('Datasets/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b"data"].T
        y = dict[b"labels"]
        Y = (np.eye(10)[y]).T
    return X, Y, y


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def normalize_data(data, mean, std):
    if mean is None and std is None:
        mean = np.mean(data, axis=1,
                       keepdims=True)  # mean of each row (for all images)
        std = np.std(data, axis=1, keepdims=True)  # Std of each row
    data = (data - mean) / std
    return data, mean, std


X_train, Y_train, y_train = LoadBatch('cifar-10-batches-py/data_batch_1')

X_train, mean_train, std_train = normalize_data(X_train, None, None)

X_val, Y_val, y_val = LoadBatch('cifar-10-batches-py/data_batch_2')

X_val = normalize_data(X_val, mean_train, std_train)[0]

X_test, Y_test, y_test = LoadBatch('cifar-10-batches-py/data_batch_3')

X_test = normalize_data(X_test, mean_train, std_train)[0]

batches = unpickle('Datasets/cifar-10-batches-py/batches.meta')

label_names = [
    label_name.decode('utf-8') for label_name in batches[b'label_names']
]


def init_weights(input_dimension,
                 hidden_dimension,
                 output_dimension,
                 seed=0,
                 std=0.01):
    #randomize weights with seed:
    np.random.seed(seed)
    std1 = 1 / np.sqrt(input_dimension)
    std2 = 1 / np.sqrt(hidden_dimension)

    W1 = np.random.normal(size=(hidden_dimension, input_dimension),
                          loc=0,
                          scale=std1)
    W2 = np.random.normal(size=(output_dimension, hidden_dimension),
                          loc=0,
                          scale=std2)
    np.random.seed(seed)
    b1 = np.zeros(shape=(hidden_dimension, 1))
    b2 = np.zeros(shape=(output_dimension, 1))
    #W = np.array([W1, W2], dtype=object)  # perhaps remove "object"
    #b = np.array([b1, b2], dtype=object)
    #return W, b
    return W1, b1, W2, b2


"""
input_dimension = X_train.shape[0]
hidden_dimension = 50
output_dimension = Y_train.shape[0]
W, b = init_weights(input_dimension, hidden_dimension, output_dimension)
print("W1 shape: ", W[0].shape)
print("W2 shape: ", W[1].shape)

print("b1 shape: ", b[0].shape)
print("b2 shape: ", b[1].shape)

exit()
"""


def softmax(x):
    """ Standard definition of the softmax function """
    e_x = np.exp(x)
    p = e_x / e_x.sum(axis=0)
    #s = np.exp(x - np.max(x, axis=0)) / np.exp(x -
    #                                           np.max(x, axis=0)).sum(axis=0)
    return p
    #return s


def relu(x):
    """ReLU activation function """
    return np.maximum(0, x)


def EvaluateClassifier(X, W_1, b_1, W_2, b_2):
    #Two layer NN. Relu --> softmax Softmax acitvation function for two layer NN
    s_1 = W_1 @ X + b_1
    h = relu(s_1)
    s = W_2 @ h + b_2
    p = softmax(s)
    return p, h


def compute_loss(X_c, Y, W1, b1, W2, b2, lmbda):
    N = X_c.shape[1]
    P, _ = EvaluateClassifier(X_c, W1, b1, W2, b2)
    loss = 1 / N * (-np.sum(Y * np.log(P)))
    return loss


def ComputeCost(X_c, Y, W1, b1, W2, b2, lmbda):

    #predictions, cross entropy loss
    #P, H = EvaluateClassifier(X_c, W, b)
    P, H = EvaluateClassifier(X_c, W1, b1, W2, b2)
    N = X_c.shape[1]
    # loss function term
    loss_cross = -np.sum(
        Y * np.log(P)) / N  # seems to work best, aslo recommended by teacher

    # regularzation term
    loss_regularization = lmbda * (np.sum((W1**2)) + np.sum((W2**2)))

    return loss_cross + loss_regularization


def ComputeAccuracy(X_c, y_c, W1, b1, W2, b2):
    #calculates mean accuracy of predictions
    probabilities, hidden_activation = EvaluateClassifier(X_c, W1, b1, W2, b2)
    acc = np.mean(y_c == np.argmax(probabilities, 0))
    return acc


def ComputeGradients(X_b, Y_b, p, h, W_1, W_2, lamb):
    n = X_b.shape[1]  # batch size
    O = Y_b.shape[0]  # size of output data

    G = -(Y_b - p)  # batch

    grad_b2 = (G @ np.ones(shape=(n, 1)) / n).reshape(O, 1)
    grad_W2 = (G @ h.T) / n + 2 * lamb * W_2
    G = W_2.T @ G
    G = G * (h > 0)

    grad_b1 = (G @ np.ones(shape=(n, 1)) / n).reshape(h.shape[0], 1)

    grad_W1 = G @ X_b.T / n + 2 * lamb * W_1

    return grad_W2, grad_b2, grad_W1, grad_b1


def ComputeGradsNum(X, Y, W1, b1, W2, b2, lambda_, h=0.00001):
    """ Python version of provided Matlab code. """
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b2 = np.zeros(shape=b2.shape)
    grad_W1 = np.zeros(shape=W1.shape)
    grad_b1 = np.zeros(shape=b1.shape)
    c = ComputeCost(X, Y, W1, b1, W2, b2, lambda_)

    for i in range(b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i, 0] = b1_try[i, 0] + h
        c2 = ComputeCost(X, Y, W1, b1_try, W2, b2, lambda_)
        grad_b1[i, 0] = (c2 - c) / h

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i, j] = W1_try[i, j] + h
            c2 = ComputeCost(X, Y, W1_try, b1, W2, b2, lambda_)
            grad_W1[i, j] = (c2 - c) / h

    for i in range(b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i, 0] = b2_try[i, 0] + h
        c2 = ComputeCost(X, Y, W1, b1, W2, b2_try, lambda_)
        grad_b2[i, 0] = (c2 - c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i, j] = W2_try[i, j] + h
            c2 = ComputeCost(X, Y, W1, b1, W2_try, b2, lambda_)
            grad_W2[i, j] = (c2 - c) / h

    return grad_W2, grad_b2, grad_W1, grad_b1


def test_gradients_num(X, Y, W1, b1, W2, b2, lambda_):
    # Finction for checking the analytical gradietn with the numerical ones.
    # Compute the gradients analytically
    P, h = EvaluateClassifier(X, W1, b1, W2, b2)

    grad_W2_analytical, grad_b2_analytical, grad_W1_analytical, grad_b1_analytical = ComputeGradients(
        X, Y, P, h, W1, W2, lambda_)

    grad_W2_numerical, grad_b2_numerical, grad_W1_numerical, grad_b1_numerical = ComputeGradsNum(
        X, Y, W1, b1, W2, b2, lambda_)

    #grad_W_numerical_slow, grad_b_numerical_slow = ComputeGradsNumSlow(
    #    X, Y, W, b, lambda_)
    # Absolute error between numerically and analytically computed gradient
    #grad_W_analytical = np.array(grad_W_analytical)
    grad_W1_analytical = np.array(grad_W1_analytical)
    grad_W2_analytical = np.array(grad_W2_analytical)
    grad_b1_analytical = np.array(grad_b1_analytical)
    grad_b2_analytical = np.array(grad_b2_analytical)

    #grad_W_abs_diff = np.abs(grad_W_numerical - grad_W_analytical)
    grad_W1_abs_diff = np.abs(grad_W1_numerical - grad_W1_analytical)
    grad_W2_abs_diff = np.abs(grad_W2_numerical - grad_W2_analytical)
    #grad_b_abs_diff = np.abs(
    #    np.array(grad_b_numerical) - np.array(grad_b_analytical))
    grad_b1_abs_diff = np.abs(
        np.array(grad_b1_numerical) - np.array(grad_b1_analytical))
    grad_b2_abs_diff = np.abs(
        np.array(grad_b2_numerical) - np.array(grad_b2_analytical))
    """
    grad_W_abs_diff_slow = np.abs(grad_W_numerical_slow - grad_W_analytical)
    grad_b_abs_diff_slow = np.abs(grad_b_numerical_slow - grad_b_analytical)
    
    # Absolute errors........

    print('For weights the maximum absolute error is ' +
          str((grad_W_abs_diff).max()))
    print(
        'For bias the maximum absolute error is ' + str(
            (grad_b_abs_diff).max()), "\n")

    print('For weights (slow grads) the maximum absolute error is ' +
          str((grad_W_abs_diff_slow).max()))
    print(
        'For bias (slow grads) the maximum absolute error is ' + str(
            (grad_b_abs_diff_slow).max()), "\n")
    """

    grad_W1_abs_sum = np.maximum(
        np.array([np.abs(grad_W1_numerical) + np.abs(grad_W1_analytical)]),
        0.00000001)

    grad_b1_abs_sum = np.maximum(
        np.array(([np.abs(grad_b1_numerical) + np.abs(grad_b1_analytical)])),
        0.00000001)
    grad_W2_abs_sum = np.maximum(
        np.array([np.abs(grad_W2_numerical) + np.abs(grad_W2_analytical)]),
        0.00000001)

    print('For weights W1 the maximum relative error is ' +
          str((grad_W1_abs_diff / grad_W1_abs_sum).max()))
    print('For weights W2 the maximum relative error is ' +
          str((grad_W2_abs_diff / grad_W2_abs_sum).max()))
    print('For bias b1 the maximum relative error is ' +
          str((grad_b1_abs_diff / grad_b1_abs_sum).max()))
    print('For bias the maximum relative error is ' +
          str((grad_b1_abs_diff / grad_b1_abs_sum).max()))


def update_cyclical_rates(t, n_s, eta_min, eta_max):
    # n_s: step size

    cycl = t // (2 * n_s)  # number of complete cycles elapsed
    if (2 * cycle + 1) * n_s <= t <= 2 * (cycle + 1) * n_s:
        return eta_max - (t -
                          (2 * cycle + 1) * n_s) / n_s * (eta_max - eta_min)
    elif 2 * cycle * n_s <= t <= (2 * cycle + 1) * n_s:
        return eta_min + (t - 2 * cycle * n_s) / n_s * (eta_max - eta_min)


def miniBatchGD(X_trainn, Y_trainn, y_trainn, W1, b1, W2, b2, X_validation,
                Y_validation, y_validation, dict, lmbda):
    #need x_train- X_test, X_validation and Y_validation as inputs...
    # take out params from dictionary
    #eta = GDparams['eta']
    n_batch = dict['n_batch']  # batch size
    n_epochs = dict['n_epochs']
    eta_max = dict['eta_max']
    eta_min = dict['eta_min']  # arrcording to instructions
    t = dict['t']
    n_s = dict['n_s']
    N = X_trainn.shape[1]  # Number of data images
    d = X_trainn.shape[0]
    #n_s = 2 * int(X_trainn.shape[1] / n_batch)

    #Dictionary for the results of the trained network
    nn_dict = {
        'epochs': [],
        'train_loss': [],
        'train_cost': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_cost': [],
        'validation_acuracy': [],
        'train_loss_mean': [],
        'train_cost_mean': [],
        'train_accuracy_mean': [],
        'validation_loss_mean': [],
        'validation_cost_mean': [],
        'validation_acuracy_mean': [],
    }

    t = 0  # time that we update the learning rates over
    eta = eta_min
    n_batch = int(np.floor(X_trainn.shape[1] / n_batch))
    for epoch in range(n_epochs):
        #np.random.seed(epoch)
        # permutation of indicies
        #shuffled_indexes = np.random.permutation(N)
        for j in range(N // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            # batch of shuffled indicies
            #X_batch = X_trainn[:, shuffled_indexes[j_start:j_end]]
            #Y_batch = Y_trainn[:, shuffled_indexes[j_start:j_end]]

            #batching without schuffling
            X_batch = X_trainn[:, j_start:j_end]
            Y_batch = Y_trainn[:, j_start:j_end]
            #grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W_star, b_star,
            #                                  lmbda)

            #print("Y (in miniBatchGD) shape: ", Y_batch.shape)
            P, H = EvaluateClassifier(X_batch, W1, b1, W2, b2)

            grad_W2, grad_b2, grad_W1, grad_b1 = ComputeGradients(
                X_batch, Y_batch, P, H, W1, W2, lmbda)
            W2 -= eta * grad_W2
            b2 -= eta * grad_b2
            W1 -= eta * grad_W1
            b1 -= eta * grad_b1
            if t <= n_s:
                eta = eta_min + t / n_s * (eta_max - eta_min)
            elif t <= 2 * n_s:
                eta = eta_max - (t - n_s) / n_s * (eta_max - eta_min)
            t = (t + 1) % (2 * n_s)

        nn_dict['epochs'].append(epoch + 1)
        #print("Accuracy shape: ", ComputeAccuracy(X, y, W_star, b_star))
        #nn_dict['train_accuracy'].append(
        #    ComputeAccuracy(X_trainn, y_trainn, W_star, b_star))
        nn_dict['train_accuracy'].append(
            ComputeAccuracy(X_trainn, y_trainn, W1, b1, W2, b2))

        #nn_dict['train_loss'].append(
        #    ComputeCost(X_trainn, Y_trainn, W_star, b_star, lmbda))
        nn_dict['train_cost'].append(
            ComputeCost(X_trainn, Y_trainn, W1, b1, W2, b2, lmbda))

        nn_dict['train_loss'].append(
            compute_loss(X_trainn, Y_trainn, W1, b1, W2, b2, lmbda))

        #nn_dict['validation_acuracy'].append(
        #    ComputeAccuracy(X_validation, y_validation, W_star, b_star))
        nn_dict['validation_acuracy'].append(
            ComputeAccuracy(X_validation, y_validation, W1, b1, W2, b2))
        nn_dict['validation_cost'].append(
            ComputeCost(X_validation, Y_validation, W1, b1, W2, b2, lmbda))
        #nn_dict['validation_loss'].append(
        #    ComputeCost(X_validation, Y_validation, W_star, b_star, lmbda))
        nn_dict['validation_loss'].append(
            compute_loss(X_validation, Y_validation, W1, b1, W2, b2, lmbda))

    nn_dict['train_loss_mean'] = ((np.mean(nn_dict['train_loss'])))
    #nn_dict['train_cost_mean'] = np.mean(nn_dict['train_cost'])
    nn_dict['train_accuracy_mean'] = (np.mean(nn_dict['train_accuracy']))

    nn_dict['validation_loss_mean'] = ((np.mean(nn_dict['validation_loss'])))
    #nn_dict['validation_cost_mean'] = np.mean(nn_dict['validation_cost'])
    nn_dict['validation_acuracy_mean'] = ((np.mean(
        nn_dict['validation_acuracy'])))

    #print("Epoch " + str(epoch + 1) + ", train loss: " +
    #      str(nn_dict['train_loss'][-1]) + ", train accuracy=" +
    #      str(nn_dict['train_accuracy'][-1]) + "\r")

    #return W_star, b_star, nn_dict
    return W1, b1, W2, b2, nn_dict


def montage(W, label_names, GDparams):
    """ Display the image for each label in W """
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']

    fig, ax = plt.subplots(1, 10, figsize=(15, 2.5))
    plt.suptitle("Weights for n_batch= " + str(GDparams['n_batch']) +
                 ", eta = " + str(GDparams['eta']) + "and labda = " +
                 str(lambda_))
    for i in range(10):
        im = W[i, :].reshape(32, 32, 3, order='F')

        sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
        sim = sim.transpose(1, 0, 2)

        ax[i].imshow(sim, interpolation='nearest')
        ax[i].set_title(label_names[i])
        ax[i].axis('off')
    plt.show()


def plot_learning_curve(nn_dictionary):
    """ Function for plotting the accuracy and loss for the training/validation data over epoches"""
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 5))

    epochs = nn_dictionary['epochs']

    train_loss = nn_dictionary['train_loss']
    train_accuracy = nn_dictionary['train_accuracy']
    train_cost = nn_dictionary['train_cost']

    validation_loss = nn_dictionary['validation_loss']
    validation_acuracy = nn_dictionary['validation_acuracy']
    validation_cost = nn_dictionary['validation_cost']

    ax[0].plot(epochs, train_loss, label="Train loss")
    ax[0].plot(epochs, validation_loss, label="Validation loss")

    ax[0].legend()
    ax[0].set(xlabel='Epoch', ylabel='Loss (cross entropy)')
    ax[0].grid()

    ax[1].plot(epochs, validation_acuracy, label="validation_acuracy")
    ax[1].plot(epochs, train_accuracy, label="train_accuracy")
    ax[1].legend()
    ax[1].set(xlabel='Epoch', ylabel='Accuracy')
    ax[1].grid()

    ax[2].plot(epochs, validation_cost, label="validation_cost")
    ax[2].plot(epochs, train_cost, label="train_cost")
    ax[2].legend()
    ax[2].set(xlabel='Epoch', ylabel='Cost')
    ax[2].grid()

    plt.show()


def plot_lambda_search(GDparams_search, lambdas):
    """ Function for plotting the accuracy and loss for the training/validation data over the lambda search"""
    #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    validation_acuracy_mean = GDparams_search['validation_acuracy_mean']
    #print("lambdas shape:", np.shape(lambdas))
    ax.plot(lambdas,
            validation_acuracy_mean,
            'o',
            label="Validation acuracy loss")
    ax.legend()
    ax.set(xlabel='Log Lambas', ylabel='Accuracy')
    ax.grid()
    plt.show()


lambda_ = 0
input_dimension = X_train.shape[0]
#input_dimension = 20
hidden_dimension = 50
output_dimension = Y_train.shape[0]
#W, b = init_weights(input_dimension, hidden_dimension, output_dimension)
W1, b1, W2, b2 = init_weights(input_dimension, hidden_dimension,
                              output_dimension)

# --------------- testing gradients ---------------
#X = X_train[0:20, [0]]
#W1 = W1[:, 0:20]
#Y = Y_train[:, 0:50]
#X = X_train[:, 0:50]
#y = y_train[0:50]

#Y = Y_train[:, 0:2]
#W_1, W_2 = W
#b_1, b_2 = b

#W_2 = W[1]

#W1 = W1[:, 0:20]

#W = np.array([[W_1, W_2]])
#b = b[:, 0:50]
lambda_ = 0.001
#test_gradients_num(X, Y, W, b, lambda_)
# Add a dictionary to pass around the weights and bias?
#test_gradients_num(X, Y, W1, b1, W2, b2, lambda_)
#exit()

#For numerical gradients checks
# test on the entire image nr 100,
#X = X_train[:, [100]]
#Y = Y_train[:, [100]]

# test on the 50 first inputs on one image (nr 100)
#X = X_train[0:50, [100]]
#Y = Y_train[0:50, [100]]
#W = W[:, 0:50]
#b = b[:, 0:50]

#X = X_train
#Y = Y_train
#y = y_train

X_v = X_val
Y_v = Y_val
y_v = y_val
lambda_ = 0.01

#test_gradients_num(X, Y, W, b, lambda_)
#lambda_ = 0
# ------------------ after tested gradients -------------------
GDparams = {
    'n_batch': 100,  # size of each batch
    'eta': 0.01,
    'n_epochs': 48,
    'eta_max': 1e-1,
    'eta_min': 1e-5,  # arrcording to instructions
    't': 0,
    'n_s': 800,
    'etas_varying': [],
}

### Excersise 3: batch_size = 100
"""
# for plotting the cyclical learning rates (without optimizing the lambdas)
W1, b1, W2, b2, nn_dict = miniBatchGD(X_train, Y_train, y_train, W1, b1, W2,
                                      b2, X_v, Y_v, y_v, GDparams, lambda_)
plot_learning_curve(nn_dict)


#print("post weights, W1: " + str(W1.shape) + "b1: " + str(b1.shape) + " W2: " +
#      str(W2.shape) + "b2 :" + str(b2.shape))
#print("returned W1 shape: ", W1.shape)
#c = ComputeCost(X_test, Y_test, W_new, b_new, lambda_)
c = ComputeCost(X_test, Y_test, W1, b1, W2, b2, lambda_)

#ComputeCost(X, Y, W, b, lmbda):

#accc = ComputeAccuracy(X_test, y_test, W_new, b_new)
accc_test = ComputeAccuracy(X_test, y_test, W1, b1, W2, b2)
accc_train = ComputeAccuracy(X_train, y_train, W1, b1, W2, b2)
print("Accuracy on test set is :", accc_test)
print("Accuracy on train set is :", accc_train)
print("Cost on test set is :", c)
"""

#montage(W_new, label_names, GDparams)  #vizualize the trained weights
#montage(W2, label_names, GDparams)  #For now, vizualize W2 weights
#plot_learning_curve(nn_dict)

#------------------------- lambda search---------------------
#Use all the data for training/validation :
images_1, labels_one_hot_1, labels_1 = LoadBatch(
    'cifar-10-batches-py/data_batch_1')
images_2, labels_one_hot_2, labels_2 = LoadBatch(
    'cifar-10-batches-py/data_batch_2')
images_3, labels_one_hot_3, labels_3 = LoadBatch(
    'cifar-10-batches-py/data_batch_3')
images_4, labels_one_hot_4, labels_4 = LoadBatch(
    'cifar-10-batches-py/data_batch_4')
images_5, labels_one_hot_5, labels_5 = LoadBatch(
    'cifar-10-batches-py/data_batch_5')
images_test, labels_one_hot_test, labels_test = LoadBatch(
    'cifar-10-batches-py/test_batch')

images = np.hstack((images_1, images_2, images_3, images_4, images_5))
labels_one_hot = np.hstack(
    (labels_one_hot_1, labels_one_hot_2, labels_one_hot_3, labels_one_hot_4,
     labels_one_hot_5))
labels = labels_1 + labels_2 + labels_3 + labels_4 + labels_5

# Subset the validation set
np.random.seed(0)
indexes_validation = np.random.choice(range(images.shape[1]),
                                      5000,
                                      replace=False)
X_val_search = images[:, indexes_validation]
Y_val_search = labels_one_hot[:, indexes_validation]
y_val_search = [labels[i] for i in indexes_validation]

# Subset the training set
images_train = np.delete(images, indexes_validation, 1)

#labels_one_hot_train
Y_train_search = np.delete(labels_one_hot, indexes_validation, 1)

#labels_train
y_train_search = [
    labels[i] for i in range(images.shape[1]) if i not in indexes_validation
]

# Normalize the new data
X_train_search, mean_train_search, std_train_search = normalize_data(
    images_train, None, None)
X_val_search = normalize_data(X_val_search, mean_train_search,
                              std_train_search)[0]

# New dictionary for lambda search

GDparams_search = {
    'n_batch': 100,
    'eta_min': 1e-5,
    'eta_max': 1e-1,
    'eta': 0.01,
    'cycles': 2,
    'n_epochs': 10,
    't': 0,
    #'n_s': 1000,
    'etas_varying': [],
    'loss_train_mean': [],
    'acc_train_mean': [],
    'loss_val_mean': [],
    'acc_val_mean': [],
    'validation_loss_mean': [],
    'validation_acuracy_mean': []
}

GDparams_search['n_s'] = int(2 * images_train.shape[1] /
                             GDparams_search['n_batch'])
input_dimension = X_train_search.shape[0]
#input_dimension = 20
hidden_dimension = 50
output_dimension = Y_train_search.shape[0]


def lambda_search():
    #lambdas = []
    n_samples = 20
    lamb_max, lamb_min = -2.5, -5
    lambdas = []
    for j in range(n_samples):
        np.random.seed(j)

        #uniform distribution
        l = lamb_min + (lamb_max - lamb_min) * np.random.rand()

        #lamb = list(10**l)  # log scale
        lamb = 10**l  # log scale
        lambdas.append(l)
        #list_lambdas_coarse.sort()
        #list_lambdas_coarse

        print("going for the grid search! ")
        #print("lambdas: ", list_lambdas_coarse)
        #grid_search()
        W1, b1, W2, b2 = init_weights(input_dimension, hidden_dimension,
                                      output_dimension)

        W1, b1, W2, b2, nn_dict = miniBatchGD(X_train_search, Y_train_search,
                                              y_train_search, W1, b1, W2, b2,
                                              X_val_search, Y_val_search,
                                              y_val_search, GDparams_search,
                                              lamb)
        #print("lambda =  %s, Train accuracy mean : %s " %
        #      (lamb, nn_dict['train_accuracy_mean']))
        #print("lambda =  %s, Train loss mean : %s " %
        #      (lamb, nn_dict['train_loss_mean']))

        #GDparams_search['validation_loss_mean'].append(
        #    nn_dict['validation_loss_mean'])
        #GDparams_search['validation_acuracy_mean'].append(
        #    nn_dict['validation_acuracy_mean'])
        #'validation_loss_mean': [],
        #'validation_acuracy_mean': []

        #print("l =  %s, lambda = 10**l = %s,  Validation accuracy mean : %s " %
        #      (l, lamb, nn_dict['validation_acuracy_mean']))
        #print("l=  %s, lambda = 10**l = %s, validation loss mean : %s " %
        #      (l, lamb, nn_dict['validation_loss_mean']))

        acc = ComputeAccuracy(X_val_search, y_val_search, W1, b1, W2, b2)
        print("l =  %s, lambda = 10**l = %s,  Validation accuracy mean : %s " %
              (l, lamb, acc))
        #print("final accuracy: ", acc)

        GDparams_search['validation_acuracy_mean'].append(acc)
        #print("Train accuracy mean: ", nn_dict['train_accuracy_mean'])
        #print("Train accuracy mean: ", nn_dict['train_accuracy_mean'])

    print("lambdas shape: ", np.shape(lambdas))
    #validation_loss_mean = GDparams_search['validation_loss_mean']
    validation_acuracy_mean = GDparams_search['validation_acuracy_mean']
    #print("validation_loss_mean shape: ", np.shape(validation_loss_mean))
    print("validation_acuracy_mean shape: ", np.shape(validation_acuracy_mean))
    plot_lambda_search(GDparams_search, lambdas)


#------------------ Train with best Lambda found ---------------------------
#lambda_search()

best_lambda = 10**(-2.816)
W1, b1, W2, b2 = init_weights(input_dimension, hidden_dimension,
                              output_dimension)

W1, b1, W2, b2, nn_dict = miniBatchGD(X_train_search, Y_train_search,
                                      y_train_search, W1, b1, W2, b2,
                                      X_val_search, Y_val_search, y_val_search,
                                      GDparams, best_lambda)
final_performace = ComputeAccuracy(X_val_search, Y_val_search, W1, b1, W2, b2)
print("Final accuracy on the test dataset with best lambda is: ",
      final_performace)
plot_learning_curve(nn_dict)

#miniBatchGD()
#nn_dict['validation_acuracy']
