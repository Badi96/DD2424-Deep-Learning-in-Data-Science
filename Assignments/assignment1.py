import numpy as np
import pickle
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


def init_weights(input_dimension, output_dimension, seed=0, std=0.01):
    #randomize weights with seed:
    np.random.seed(seed)
    W = np.random.normal(size=(output_dimension, input_dimension),
                         loc=0,
                         scale=std)
    np.random.seed(seed)
    b = np.random.normal(size=(output_dimension, 1), loc=0, scale=std)
    return W, b


def softmax(x):
    """ Standard definition of the softmax function """
    e_x = np.exp(x)
    p = e_x / e_x.sum(axis=0)
    return p


def EvaluateClassifier(X, W, b):
    """ Softmax acitvation function """
    a = softmax(W @ X + b)
    return a


def ComputeCost(X_c, Y, W, b, lmbda):

    #predictions, cross entropy loss
    P = EvaluateClassifier(X_c, W, b)
    # loss function term
    loss_cross = -np.sum(Y * np.log(P))
    # regularzation term
    loss_regularization = lmbda * (W**2).sum()
    return loss_cross / X_c.shape[1] + loss_regularization


def ComputeAccuracy(X_c, y_c, W_star, b_star):
    #calculates mean accuracy of predictions
    probabilities = EvaluateClassifier(X_c, W_star, b_star)
    return np.mean(y_c == np.argmax(probabilities, 0))


def ComputeGradients(X_b, Y_b, W_star, b_star, lamb):
    n = X_b.shape[1]
    O = Y_b.shape[0]
    predicted_labels = EvaluateClassifier(X_b, W_star, b)
    G = -(Y_b - predicted_labels)
    grad_W = (G @ X_b.T) / n + 2 * lamb * W_star
    grad_b = (G @ np.ones(shape=(n, 1)) / n).reshape(O, 1)
    return grad_W, grad_b


def ComputeGradsNum(X, Y, P, W, b, lamda, h=0.000001):
    #Converted from matlab code
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


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h=0.000001):
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


def test_gradients_num(X, Y, W, b, lambda_):
    # Finction for checking the analytical gradietn with the numerical ones.
    # Compute the gradients analytically
    P = EvaluateClassifier(X, W, b)
    grad_W_analytical, grad_b_analytical = ComputeGradients(
        X, Y, W, b, lambda_)

    # Compute the gradients numerically
    grad_W_numerical, grad_b_numerical = ComputeGradsNum(
        X, Y, P, W, b, lambda_)
    grad_W_numerical_slow, grad_b_numerical_slow = ComputeGradsNumSlow(
        X, Y, P, W, b, lambda_)
    # Absolute error between numerically and analytically computed gradient
    grad_W_abs_diff = np.abs(grad_W_numerical - grad_W_analytical)
    grad_b_abs_diff = np.abs(grad_b_numerical - grad_b_analytical)

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

    # Relative error between numerically and analytically computed gradient
    grad_W_abs_sum = np.maximum(
        np.abs(grad_W_numerical) + np.abs(grad_W_analytical), 0.00000001)
    grad_b_abs_sum = np.maximum(
        np.abs(grad_b_numerical) + np.abs(grad_b_analytical), 0.00000001)

    print('For weights the maximum relative error is ' +
          str((grad_W_abs_diff / grad_W_abs_sum).max()))
    print('For bias the maximum relative error is ' +
          str((grad_b_abs_diff / grad_b_abs_sum).max()))

    grad_W_abs_sum_slow = np.maximum(
        np.abs(grad_W_numerical_slow) + np.abs(grad_W_analytical), 0.00000001)
    grad_b_abs_sum_slow = np.maximum(
        np.abs(grad_b_numerical_slow) + np.abs(grad_b_analytical), 0.00000001)

    print('For weights (slow grad) the maximum relative error is ' +
          str((grad_W_abs_diff_slow / grad_W_abs_sum_slow).max()))
    print('For bias (slow grad) the maximum relative error is ' +
          str((grad_b_abs_diff_slow / grad_b_abs_sum_slow).max()))


def miniBatchGD(X_trainn, Y_trainn, y_trainn, W_star, b_star, X_validation,
                Y_validation, y_validation, GDparams, lmbda):
    #need x_train- X_test, X_validation and Y_validation as inputs...
    # take out params from dictionary
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    N = X_trainn.shape[1]  # Number of data images
    d = X_trainn.shape[0]

    #Dictionary for the results of the trained network
    nn_dict = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_acuracy': []
    }
    for epoch in range(n_epochs):
        np.random.seed(epoch)
        # permutation of indicies
        shuffled_indexes = np.random.permutation(N)
        for j in range(N // n_batch):
            #differently than in Matlab...
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            # batch of shuffled indicies
            X_batch = X_trainn[:, shuffled_indexes[j_start:j_end]]
            Y_batch = Y_trainn[:, shuffled_indexes[j_start:j_end]]
            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W_star, b_star,
                                              lmbda)
            W_star -= eta * grad_W
            b_star -= eta * grad_b
        nn_dict['epochs'].append(epoch + 1)
        #print("Accuracy shape: ", ComputeAccuracy(X, y, W_star, b_star))
        nn_dict['train_accuracy'].append(
            ComputeAccuracy(X_trainn, y_trainn, W_star, b_star))

        nn_dict['train_loss'].append(
            ComputeCost(X_trainn, Y_trainn, W_star, b_star, lmbda))

        nn_dict['validation_acuracy'].append(
            ComputeAccuracy(X_validation, y_validation, W_star, b_star))

        nn_dict['validation_loss'].append(
            ComputeCost(X_validation, Y_validation, W_star, b_star, lmbda))

        print("Epoch " + str(epoch + 1) + ", train loss: " +
              str(nn_dict['train_loss'][-1]) + ", train accuracy=" +
              str(nn_dict['train_accuracy'][-1]) + "\r")

    return W_star, b_star, nn_dict


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
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    epochs = nn_dictionary['epochs']
    train_loss = nn_dictionary['train_loss']
    train_accuracy = nn_dictionary['train_accuracy']

    validation_acuracy = nn_dictionary['validation_acuracy']
    validation_loss = nn_dictionary['validation_loss']

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
    plt.show()


input_dimension = X_train.shape[0]
output_dimension = Y_train.shape[0]
W, b = init_weights(input_dimension, output_dimension)

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
lambda_ = 1

#test_gradients_num(X, Y, W, b, lambda_)
#exit()
#lambda_ = 0
GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 40}
W_new, b_new, nn_dict = miniBatchGD(X_train, Y_train, y_train, W, b, X_v, Y_v,
                                    y_v, GDparams, lambda_)

c = ComputeCost(X_test, Y_test, W_new, b_new, lambda_)
#ComputeCost(X, Y, W, b, lmbda):
accc = ComputeAccuracy(X_test, y_test, W_new, b_new)

print("Accuracy on test set is :", accc)
print("Cost on test set is :", c)

montage(W_new, label_names, GDparams)  #vizualize the trained weights
plot_learning_curve(nn_dict)