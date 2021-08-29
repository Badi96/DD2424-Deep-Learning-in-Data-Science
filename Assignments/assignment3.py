from typing import NewType
import numpy as np
import pickle
from numpy.core.fromnumeric import var

from numpy.lib.function_base import kaiser
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

batches = unpickle('Datasets/cifar-10-batches-py/batches.meta')

label_names = [
    label_name.decode('utf-8') for label_name in batches[b'label_names']
]


def init_weights(input_dimension,
                 hidden_dimensions,
                 output_dimension,
                 he_initialization=True,
                 seed=0,
                 std=0.01):
    k = len(hidden_dimensions) + 1
    W, b, gamma, beta = [None] * k, [None] * k, [None] * (k - 1), [None
                                                                   ] * (k - 1)
    n_nodes = [input_dimension] + hidden_dimensions + [output_dimension]

    # Define random seed and iterate layers
    np.random.seed(seed)
    for l in range(k):

        # Define dimensions of the layer
        inputs = n_nodes[l]
        outputs = n_nodes[l + 1]

        # Define standard deviation for random weights and bias normal distribution
        scale = np.sqrt(2 / inputs) if he_initialization else std

        # Initialize weights, bias, gammas and betas
        W[l] = np.random.normal(size=(outputs, inputs), loc=0, scale=scale)
        b[l] = np.zeros((outputs, 1))
        if l < (k - 1):
            gamma[l] = np.ones((outputs, 1))
            beta[l] = np.zeros((outputs, 1))

    return W, b, gamma, beta


def softmax(x):
    """ Standard definition of the softmax function """
    e_x = np.exp(x)
    p = e_x / e_x.sum(axis=0)
    return p


def relu(x):
    """ReLU activation function """
    return np.maximum(0, x)


def EvaluateClassifier(
    X,
    W,
    b,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    batch_normalization=False,
):
    #Two layer NN. Relu --> softmax Softmax acitvation function for two layer NN

    k = len(W)  # number of layers

    X_layers, S, S_BN = [X.copy()] + [None] * (k - 1), [None] * (
        k - 1), [None] * (k - 1)
    if batch_normalization == True:
        if mean is None and variance is None:
            return_mean_var = True
            mean, variance = [None] * (k - 1), [None] * (k - 1)
        else:
            return_mean_var = False
    # Iterate through hidden layers
    for l in range(k - 1):
        S[l] = W[l] @ X_layers[l] + b[l]
        if batch_normalization == True:
            if return_mean_var == True:
                mean[l] = S[l].mean(axis=1).reshape(-1, 1)
                variance[l] = S[l].var(axis=1).reshape(-1, 1)
            S_BN[l] = (S[l] - mean[l]) / (np.sqrt(variance[l] + 1e-15))
            S_BatchNorm_Scaled = S_BN[l] * gamma[l] + beta[l]
            X_layers[l + 1] = relu(S_BatchNorm_Scaled)
        else:
            X_layers[l + 1] = relu(S[l])

    # Output layer
    P = softmax(W[k - 1] @ X_layers[k - 1] + b[k - 1])

    if batch_normalization:
        if return_mean_var:
            return P, S_BN, S, X_layers[1:], mean, variance
        else:
            return P, S_BN, S, X_layers[1:]
    else:
        return P, X_layers[1:]


def ComputeCost(
    X_c,
    Y,
    W,
    b,
    lmbda,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    batch_norm=False,
):
    """ Calculated the total loss of the NN"""
    loss_regularization = 0
    if batch_norm == True:
        if mean is None and variance is None:
            P, _, _, _, _, _ = EvaluateClassifier(X_c,
                                                  W,
                                                  b,
                                                  gamma,
                                                  beta,
                                                  batch_normalization=True)

        else:
            P, _, _, _ = EvaluateClassifier(X_c,
                                            W,
                                            b,
                                            gamma,
                                            beta,
                                            mean,
                                            variance,
                                            batch_normalization=True)
    else:
        P, _ = EvaluateClassifier(X_c, W, b)

    N = X_c.shape[1]
    # loss function term
    loss_cross = -np.sum(
        Y * np.log(P)) / N  # seems to work best, aslo recommended by teacher

    # regularzation term
    for W_layer in W:
        loss_regularization += lmbda * (np.sum((W_layer**2)))
        #loss_regularization += lmbda * (((W_layer**2).sum()))
    return loss_cross + loss_regularization


def ComputeAccuracy(X_c,
                    y_c,
                    W,
                    b,
                    gamma=None,
                    beta=None,
                    mean=None,
                    variance=None,
                    batch_norm=False):
    """calculates mean accuracy of predictions"""
    if batch_norm == True:
        if mean is None and variance is None:
            probabilities, _, _, _, mean, variance = EvaluateClassifier(
                X_c, W, b, gamma=gamma, beta=beta, batch_normalization=True)
        else:
            probabilities, _, _, _ = EvaluateClassifier(
                X_c,
                W,
                b,
                gamma,
                beta,
                mean,
                variance,
                batch_normalization=True)
    else:
        probabilities, _ = EvaluateClassifier(X_c, W, b)
    acc = np.mean(y_c == np.argmax(probabilities, 0))
    return acc


def BatchNormBackPass(G_batch, S_batch, mean, variance):
    """ propagate through the batch normalization """
    N = S_batch.shape[1]
    G1 = G_batch * (((variance + 1e-15)**(-0.5)) @ np.ones((1, N)))
    G2 = G_batch * (((variance + 1e-15)**(-1.5)) @ np.ones((1, N)))
    D = S_batch - mean @ np.ones((1, N))
    c = (G2 * D) @ np.ones((N, 1))

    return G1 - (1 / N) * (G1 @ np.ones((N, 1))) - (1 / N) * D * (c @ np.ones(
        (1, N)))


def ComputeGradients(X_b,
                     Y_b,
                     P_batch,
                     S_BN,
                     S,
                     X_layers,
                     W,
                     b,
                     lamb,
                     gamma=None,
                     beta=None,
                     mean=None,
                     variance=None,
                     batch_norm=False):
    """Calcultes the gradients of all  the weights, bias, gammas and betas on the Neural netowrk according to the lecture notes."""
    N = X_b.shape[1]  # batch size
    O = Y_b.shape[0]  # size of output data
    k = len(W)

    X_layers = [X_b.copy()] + X_layers
    if batch_norm == True:
        #create empty gradient vectorts
        grad_W = [None] * k
        grad_b = [None] * k
        grad_gamma = [None] * (k - 1)
        grad_beta = [None] * (k - 1)
        # weights of gradients and bias for each layers k

        #Backwards pass on batch
        G_batch = -(Y_b - P_batch)

        grad_W[k -
               1] = 1 / N * (G_batch @ X_layers[k - 1].T) + 2 * lamb * W[k - 1]
        grad_b[k - 1] = 1 / N * (G_batch @ np.ones((N, 1)))
        G_batch = W[k - 1].T @ G_batch
        G_batch = G_batch * (X_layers[k - 1] > 0)

        #iterate through all layers:
        for layer in range(k - 2, -1, -1):
            #compute gradients for the scale and offset parameters
            grad_gamma[layer] = (1 / N) * ((G_batch * S_BN[layer]) @ np.ones(
                (N, 1)))
            grad_beta[layer] = (1 / N) * (G_batch @ np.ones((N, 1)))

            # Propagate the gradients through the scale and shift
            G_batch = G_batch * (gamma[layer] @ np.ones((1, N)))

            # Propagate G through the batch normalization
            G_batch = BatchNormBackPass(G_batch, S[layer], mean[layer],
                                        variance[layer])

            # Gradient of weights and bias for layer l+1 (Python indexes starts on 0)
            grad_W[layer] = (1 / N) * (
                G_batch @ X_layers[layer].T) + 2 * lamb * W[layer]
            grad_b[layer] = (1 / N) * (G_batch @ np.ones((N, 1)))

            # If layer>1 propagate G to the previous layer
            if layer > 0:
                G_batch = W[layer].T @ G_batch
                G_batch = G_batch * (X_layers[layer] > 0)

        return grad_W, grad_b, grad_gamma, grad_beta
    else:
        grad_b, grad_W = [], []
        for w_i, b_i in zip(W, b):
            grad_W.append(np.zeros_like(w_i))
            grad_b.append(np.zeros_like(b_i))
        G_batch = -(Y_b - P_batch)
        # Iterate through each layer
        for layer in range(k - 1, 0, -1):
            grad_W[layer] = (1 / N) * (
                G_batch @ X_layers[layer].T) + 2 * lamb * W[layer]
            grad_b[layer] = (1 / N) * (G_batch @ np.ones((N, 1)))
            G_batch = W[layer].T @ G_batch
            G_batch = G_batch * (X_layers[layer] > 0)

        #Fix gradinets for the first input layer
        grad_W[0] = 1 / N * G_batch @ X_layers[0].T + lamb * W[0]
        grad_b[0] = 1 / N * G_batch @ np.ones((N, 1))
        return grad_W, grad_b


def ComputeGradsNum(X,
                    Y,
                    lambda_,
                    W,
                    b,
                    gamma,
                    beta,
                    mean,
                    var,
                    batch_normalization,
                    h=0.000001):
    """Python version of provided Matlab code for testing the gradients numerically."""
    # Create lists for saving the gradients by layers
    grad_W = [W_l.copy() for W_l in W]
    grad_b = [b_l.copy() for b_l in b]
    if batch_normalization:
        grad_gamma = [gamma_l.copy() for gamma_l in gamma]
        grad_beta = [beta_l.copy() for beta_l in beta]

    # Compute initial cost and iterate layers k
    c = ComputeCost(X, Y, W, b, lambda_, gamma, beta, mean, var,
                    batch_normalization)
    k = len(W)
    for l in range(k):

        # Gradients for bias
        for i in range(b[l].shape[0]):
            b_try = [b_l.copy() for b_l in b]
            b_try[l][i, 0] += h
            c2 = ComputeCost(X, Y, W, b_try, lambda_, gamma, beta, mean, var,
                             batch_normalization)
            grad_b[l][i, 0] = (c2 - c) / h

        # Gradients for weights
        for i in range(W[l].shape[0]):
            for j in range(W[l].shape[1]):
                W_try = [W_l.copy() for W_l in W]
                W_try[l][i, j] += h
                c2 = ComputeCost(X, Y, W_try, b, lambda_, gamma, beta, mean,
                                 var, batch_normalization)
                grad_W[l][i, j] = (c2 - c) / h

        if l < (k - 1) and batch_normalization:

            # Gradients for gamma
            for i in range(gamma[l].shape[0]):
                gamma_try = [gamma_l.copy() for gamma_l in gamma]
                gamma_try[l][i, 0] += h
                c2 = ComputeCost(X, Y, W, b, lambda_, gamma_try, beta, mean,
                                 var, batch_normalization)
                grad_gamma[l][i, 0] = (c2 - c) / h

            # Gradients for betas
            for i in range(beta[l].shape[0]):
                beta_try = [beta_l.copy() for beta_l in beta]
                beta_try[l][i, 0] += h
                c2 = ComputeCost(X, Y, W, b, lambda_, gamma, beta_try, mean,
                                 var, batch_normalization)
                grad_beta[l][i, 0] = (c2 - c) / h

    if batch_normalization:
        return grad_W, grad_b, grad_gamma, grad_beta
    else:
        return grad_W, grad_b


def test_gradients_num(X,
                       Y,
                       W,
                       b,
                       lambda_,
                       gamma=None,
                       beta=None,
                       batch_norm=False):
    """Function for checking the analytical gradietn with the numerical ones."""
    # Compute the gradients analytically
    k = len(W)
    if batch_norm is True:
        P, S_BN, S, X_layers, mean, varinace = EvaluateClassifier(
            X,
            W,
            b,
            gamma=gamma,
            beta=beta,
            mean=None,
            variance=None,
            batch_normalization=True)

        grad_W_numerical, grad_b_numerical, grad_gamma_numerical, grad_beta_numerical = ComputeGradsNum(
            X,
            Y,
            lambda_,
            W,
            b,
            gamma,
            beta,
            mean=None,
            var=None,
            batch_normalization=batch_norm)

        grad_W_analytical, grad_b_analytical, grad_gamma_analytical, grad_beta_analytical = ComputeGradients(
            X,
            Y,
            P,
            S_BN=S_BN,
            S=S,
            X_layers=X_layers,
            W=W,
            b=b,
            lamb=lambda_,
            gamma=gamma,
            beta=beta,
            mean=mean,
            variance=varinace,
            batch_norm=True)

        print("For weights, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_W_analytical[i] - grad_W_numerical[i]))
            for i in range(len(grad_W_analytical))
        ])

        print("For bias, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_b_analytical[i] - grad_b_numerical[i]))
            for i in range(len(grad_b_analytical))
        ])
        print("For bias, the maximum relative errors for each layer is:")
        print([
            np.max(np.abs(grad_b_analytical[i] - grad_b_numerical[i])) /
            np.maximum(
                np.array([
                    np.max(
                        np.abs(grad_b_analytical[i]) +
                        np.abs(grad_b_numerical[i]))
                ]), 0.000001) for i in range(len(grad_b_analytical))
        ])

        print("For weights, the maximum relative errors for each layer is:")
        print([
            np.max(np.abs(grad_W_analytical[i] - grad_W_numerical[i])) /
            np.maximum(
                np.array([
                    np.max(np.abs(grad_W_analytical[i] + grad_W_numerical[i]))
                ]), 0.0000001) for i in range(len(grad_W_analytical))
        ])
        #  ---------------- beta and gamma ------------------
        print("For gamma, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_gamma_analytical[i] - grad_gamma_numerical[i]))
            for i in range(len(grad_gamma_analytical))
        ])

        print("For beta, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_beta_analytical[i] - grad_beta_numerical[i]))
            for i in range(len(grad_beta_analytical))
        ])
        print("For beta, the maximum relative errors for each layer is:")
        print([
            np.max(np.abs(grad_beta_analytical[i] - grad_beta_numerical[i])) /
            np.maximum(
                np.array([
                    np.max(
                        np.abs(grad_beta_analytical[i]) +
                        np.abs(grad_beta_numerical[i]))
                ]), 0.0000001) for i in range(len(grad_beta_analytical))
        ])

        print("For gamma, the maximum relative errors for each layer is:")
        print([
            np.max(np.abs(grad_gamma_analytical[i] - grad_gamma_numerical[i]))
            / np.maximum(
                np.array([
                    np.max(
                        np.abs(grad_gamma_analytical[i] +
                               grad_gamma_numerical[i]))
                ]), 0.0000001) for i in range(len(grad_gamma_analytical))
        ])

    else:
        grad_W_numerical, grad_b_numerical = ComputeGradsNum(
            X,
            Y,
            lambda_,
            W,
            b,
            gamma,
            beta,
            mean=None,
            var=None,
            batch_normalization=batch_norm)
        print("test gradients.")
        print("In test_gradients, W len:", len(W))
        P, X_layers = EvaluateClassifier(X, W, b)
        grad_W_analytical, grad_b_analytical = ComputeGradients(
            X,
            Y,
            P,
            S_BN=None,
            S=None,
            X_layers=X_layers,
            W=W,
            b=b,
            lamb=lambda_,
            gamma=None,
            beta=None,
            mean=None,
            variance=None,
            batch_norm=batch_norm)

        # Absolute errors check
        print("For weights, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_W_analytical[i] - grad_W_numerical[i]))
            for i in range(len(grad_W_analytical))
        ])

        print("For bias, the maximum absolute errors for each layer is:")
        print([
            np.max(np.abs(grad_b_analytical[i] - grad_b_numerical[i]))
            for i in range(len(grad_b_analytical))
        ])

        grad_W_analytical = np.array(grad_W_analytical)
        grad_b_analytical = np.array(grad_b_analytical)

        #Relative errors check
        print("For bias, the maximum relative errors for each layer is:")
        print([
            np.max(np.abs(grad_b_analytical[i] - grad_b_numerical[i])) /
            np.max(np.abs(grad_b_analytical[i] + grad_b_numerical[i]))
            for i in range(len(grad_b_analytical))
        ])

        print("For weights, the maximum relative errors for each layer is: ")
        print([
            np.max(np.abs(grad_W_analytical[i] - grad_W_numerical[i])) /
            np.max(np.abs(grad_W_analytical[i] + grad_W_numerical[i]))
            for i in range(len(grad_W_analytical))
        ])


"""
#------Testing the gradients------
X = X_train[0:30, 0:10]
Y = Y_train[:, 0:10]
#X = X_train
#Y = Y_train
lambda_ = 0
input_dimension = X.shape[0]
hidden_dimensions = [50, 50]
output_dimension = Y_train.shape[0]
W, b, gamma, beta = init_weights(input_dimension, hidden_dimensions,
                                 output_dimension)

# test gradients without batch norm
test_gradients_num(X,
                   Y,
                   W,
                   b,
                   lambda_,
                   gamma=None,
                   beta=None,
                   batch_norm=False)

# test gradients with batch norm
test_gradients_num(X,
                   Y,
                   W,
                   b,
                   lambda_,
                   gamma=gamma,
                   beta=beta,
                   batch_norm=True)
"""


def miniBatchGD(X_trainn,
                Y_trainn,
                y_trainn,
                W,
                b,
                X_validation,
                Y_validation,
                y_validation,
                dict,
                lmbda,
                gamma=None,
                beta=None,
                batch_normalization=False,
                alpha=0.9):
    # take out params from dictionary
    n_batch = dict['n_batch']  # batch size
    n_epochs = dict['n_epochs']
    eta_max = dict['eta_max']
    eta_min = dict['eta_min']  # arrcording to instructions
    t = dict['t']
    n_s = dict['n_s']
    N = X_trainn.shape[1]  # Number of data images
    d = X_trainn.shape[0]

    #Dictionary for the results of the trained network
    nn_dict = {
        'epochs': [],
        'train_loss': [],
        'train_cost': [],
        'train_accuracy': [],
        'validation_cost': [],
        'validation_acuracy': [],
        'train_loss_mean': [],
        'train_accuracy_mean': [],
        'validation_acuracy_mean': [],
    }

    t = 0  # time that we update the learning rates over
    eta = eta_min
    for epoch in range(n_epochs):
        np.random.seed(epoch)
        # permutation of indicies
        shuffled_indexes = np.random.permutation(N)
        for j in range(N // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            # batch of shuffled indicies
            X_batch = X_trainn[:, shuffled_indexes[j_start:j_end]]
            Y_batch = Y_trainn[:, shuffled_indexes[j_start:j_end]]

            if batch_normalization == True:
                P, S_BN, S, X_layers, mean, variance = EvaluateClassifier(
                    X_batch,
                    W=W,
                    b=b,
                    gamma=gamma,
                    beta=beta,
                    batch_normalization=True)
                if t == 0:
                    mean_avg = mean_train
                    var_avg = variance
                else:
                    mean_avg = [
                        alpha * mean_avg[layer] + (1 - alpha) * mean[layer]
                        for layer in range(len(mean))
                    ]
                    var_avg = [
                        alpha * var_avg[layer] + (1 - alpha) * variance[layer]
                        for layer in range(len(variance))
                    ]
                grad_W, grad_b, grad_gamma, grad_beta = ComputeGradients(
                    X_batch,
                    Y_batch,
                    P,
                    S_BN=S_BN,
                    S=S,
                    X_layers=X_layers,
                    W=W,
                    b=b,
                    lamb=lmbda,
                    gamma=gamma,
                    beta=beta,
                    mean=mean,
                    variance=variance,
                    batch_norm=True)

            else:
                P, X_layers = EvaluateClassifier(X_batch, W, b)
                grad_W, grad_b = ComputeGradients(
                    X_batch,
                    Y_batch,
                    P,
                    S_BN=None,
                    S=None,
                    X_layers=X_layers,
                    W=W,
                    b=b,
                    lamb=lmbda,
                    gamma=None,
                    beta=None,
                    mean=None,
                    variance=None,
                    batch_norm=batch_normalization)

            W = [W[layer] - eta * grad_W[layer] for layer in range(len(W))]
            b = [b[layer] - eta * grad_b[layer] for layer in range(len(b))]
            if batch_normalization == True:
                gamma = [
                    gamma[layer] - eta * grad_gamma[layer]
                    for layer in range(len(gamma))
                ]
                beta = [
                    beta[layer] - eta * grad_beta[layer]
                    for layer in range(len(beta))
                ]
            if t <= n_s:
                eta = eta_min + t / n_s * (eta_max - eta_min)
            elif t <= 2 * n_s:
                eta = eta_max - (t - n_s) / n_s * (eta_max - eta_min)
            t = (t + 1) % (2 * n_s)
        if batch_normalization == True:
            nn_dict['epochs'].append(epoch + 1)

            nn_dict['train_accuracy'].append(
                ComputeAccuracy(X_trainn,
                                y_trainn,
                                W,
                                b,
                                gamma=gamma,
                                beta=beta,
                                mean=mean_avg,
                                variance=var_avg,
                                batch_norm=True))

            nn_dict['train_cost'].append(
                ComputeCost(X_trainn,
                            Y_trainn,
                            W,
                            b,
                            lmbda,
                            gamma=gamma,
                            beta=beta,
                            mean=mean_avg,
                            variance=var_avg,
                            batch_norm=True))

            nn_dict['validation_acuracy'].append(
                ComputeAccuracy(X_validation,
                                y_validation,
                                W,
                                b,
                                gamma=gamma,
                                beta=beta,
                                mean=mean_avg,
                                variance=var_avg,
                                batch_norm=True))
            nn_dict['validation_cost'].append(
                ComputeCost(X_validation,
                            Y_validation,
                            W,
                            b,
                            lmbda,
                            gamma=gamma,
                            beta=beta,
                            mean=mean_avg,
                            variance=var_avg,
                            batch_norm=True))
        else:

            nn_dict['epochs'].append(epoch + 1)

            nn_dict['train_accuracy'].append(
                ComputeAccuracy(X_trainn, y_trainn, W, b))

            nn_dict['train_cost'].append(
                ComputeCost(X_trainn, Y_trainn, W, b, lmbda))

            nn_dict['validation_acuracy'].append(
                ComputeAccuracy(X_validation, y_validation, W, b))
            nn_dict['validation_cost'].append(
                ComputeCost(X_validation, Y_validation, W, b, lmbda))

    nn_dict['train_accuracy_mean'] = (np.mean(nn_dict['train_accuracy']))

    nn_dict['validation_acuracy_mean'] = ((np.mean(
        nn_dict['validation_acuracy'])))

    if batch_normalization == True:
        train_accuracy = ComputeAccuracy(X_trainn,
                                         y_trainn,
                                         W,
                                         b,
                                         gamma=gamma,
                                         beta=beta,
                                         mean=mean_avg,
                                         variance=var_avg,
                                         batch_norm=True)
        validation_accuracy = ComputeAccuracy(X_validation,
                                              y_validation,
                                              W,
                                              b,
                                              gamma=gamma,
                                              beta=beta,
                                              mean=mean_avg,
                                              variance=var_avg,
                                              batch_norm=True)
        print("The accuracy on the training set is: ", train_accuracy)
        print("The accuracy on the validation set is: ", validation_accuracy)
        return W, b, gamma, beta, mean_avg, var_avg, nn_dict
    else:
        train_accuracy = ComputeAccuracy(X_trainn, y_trainn, W, b)
        validation_accuracy = ComputeAccuracy(X_validation, y_validation, W, b)
        print("The accuracy on the training set is: ", train_accuracy)
        print("The accuracy on the validation set is: ", validation_accuracy)
        return W, b, nn_dict


def plot_learning_curve(nn_dictionary):
    """ Function for plotting the accuracy and loss for the training/validation data over epoches"""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))

    epochs = nn_dictionary['epochs']

    #train_loss = nn_dictionary['train_loss']
    train_accuracy = nn_dictionary['train_accuracy']
    train_cost = nn_dictionary['train_cost']

    validation_acuracy = nn_dictionary['validation_acuracy']
    validation_cost = nn_dictionary['validation_cost']

    ax[0].plot(epochs, validation_cost, label="Validation loss")
    ax[0].plot(epochs, train_cost, label="Train loss")
    ax[0].legend()
    ax[0].set(xlabel='Epoch', ylabel='Total Loss')
    ax[0].grid()

    ax[1].plot(epochs, validation_acuracy, label="Validation Acuracy")
    ax[1].plot(epochs, train_accuracy, label="Train Accuracy")
    ax[1].legend()
    ax[1].set(xlabel='Epoch', ylabel='Accuracy ')
    ax[1].grid()

    plt.show()


def plot_lambda_search(GDparams_search, lambdas):
    """ Function for plotting the accuracy and loss for the training/validation data over the lambda search"""
    #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    validation_acuracy_mean = GDparams_search['validation_acuracy_mean']
    #print("lambdas shape:", np.shape(lambdas))
    ax.plot(lambdas, validation_acuracy_mean, 'o', label="Validation accuracy")
    ax.legend()
    ax.set(xlabel='Log Lambas', ylabel='Accuracy')
    ax.grid()
    plt.show()


"""
#  smaller dataset
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
    'n_epochs': 20,
    'eta_max': 1e-1,
    'eta_min': 1e-5,  # arrcording to instructions
    't': 0,
    'n_s': 2250,
    'etas_varying': [],
}
"""
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

# Subset the validation set to 5000 images
np.random.seed(0)
indexes_validation = np.random.choice(range(images.shape[1]),
                                      5000,
                                      replace=False)
X_val_search = images[:, indexes_validation]
Y_val_search = labels_one_hot[:, indexes_validation]
y_val_search = [labels[i] for i in indexes_validation]

# Subset the training set by removing th 5000 validation images
images_train = np.delete(images, indexes_validation, 1)

#labels_one_hot_train
Y_train_search = np.delete(labels_one_hot, indexes_validation, 1)

#labels_train
y_train_search = [
    labels[i] for i in range(images.shape[1]) if i not in indexes_validation
]

# Normalize the new data by centering and dividing inputs by 255.
X_train_search, mean_train_search, std_train_search = normalize_data(
    images_train, None, None)
X_val_search = normalize_data(X_val_search, mean_train_search,
                              std_train_search)[0]

# New dictionary for lambda search
GDparams_search = {
    'n_batch': 100,
    'eta_min': 1e-5,
    'eta_max': 1e-1,
    'eta': 0.1,
    'n_epochs': 20,
    't': 0,
    'n_s': 2250,
    'etas_varying': [],
    'loss_train_mean': [],
    'acc_train_mean': [],
    'loss_val_mean': [],
    'acc_val_mean': [],
    'validation_acuracy_mean': []
}

#GDparams_search['n_s'] = int(2 * images_train.shape[1] /
#                             GDparams_search['n_batch'])
GDparams_search['n_s'] = int(5 * 45000 / GDparams_search['n_batch'])
input_dimension = X_train_search.shape[0]
#input_dimension = 20
#hidden_dimensions = [50, 50] # 3 layer neural network
hidden_dimensions = [50, 30, 20, 20, 10, 10, 10, 10]  # 9 layer neural network
output_dimension = Y_train_search.shape[0]

W, b, gamma, beta = init_weights(input_dimension, hidden_dimensions,
                                 output_dimension)
lamb = 0.005
#return W, b, gamma, beta, mean_avg, var_avg, nn_dict
#W, b, nn_dict = miniBatchGD(X_train_search, Y_train_search, y_train_search, W,
#                            b, X_val_search, Y_val_search, y_val_search,
#                            GDparams_search, lamb)
"""
# for plotting the evolutions
W, b, gamma, beta, mean_avg, var_avg, nn_dict = miniBatchGD(
    X_train_search,
    Y_train_search,
    y_train_search,
    W,
    b,
    X_val_search,
    Y_val_search,
    y_val_search,
    GDparams_search,
    lamb,
    gamma=gamma,
    beta=beta,
    batch_normalization=True)


plot_learning_curve(nn_dict)
"""


def lambda_search(W, b, gamma, beta):
    """Function for testing performace over range of lambda values"""
    n_samples = 10
    lamb_max, lamb_min = -1.7, -2.75
    lambdas = []
    for j in range(n_samples):
        np.random.seed(j)

        #uniform distribution
        l = lamb_min + (lamb_max - lamb_min) * np.random.rand()
        print("log lambda is: ", l)

        #lamb = list(10**l)  # log scale
        lamb = 10**l  # log scale
        lambdas.append(l)

        #W, b, = init_weights(input_dimension, hidden_dimensions,
        #                     output_dimension)
        W, b, gamma, beta, mean_avg, var_avg, nn_dict = miniBatchGD(
            X_train_search,
            Y_train_search,
            y_train_search,
            W,
            b,
            X_val_search,
            Y_val_search,
            y_val_search,
            GDparams_search,
            lamb,
            gamma=gamma,
            beta=beta,
            batch_normalization=True)

        accuracy = ComputeAccuracy(X_val_search,
                                   y_val_search,
                                   W,
                                   b,
                                   gamma=gamma,
                                   beta=beta,
                                   mean=mean_avg,
                                   variance=var_avg,
                                   batch_norm=True)

        print("l =  %s, lambda = 10**l = %s,  Validation accuracy mean : %s " %
              (l, lamb, accuracy))
        #print("final accuracy: ", acc)

        GDparams_search['validation_acuracy_mean'].append(accuracy)
        #print("Train accuracy mean: ", nn_dict['train_accuracy_mean'])
        #print("Train accuracy mean: ", nn_dict['train_accuracy_mean'])

    print("lambdas shape: ", np.shape(lambdas))
    plot_lambda_search(GDparams_search, lambdas)


#------------------ Train with best Lambda found ---------------------------
#lambda_search(W, b, gamma, beta)

best_lambda = 4.8739 * 10**(-3)
#W1, b1, W2, b2 = init_weights(input_dimension, hidden_dimension,
#                              output_dimension)
"""
W, b, gamma, beta, mean_avg, var_avg, nn_dict = miniBatchGD(
    X_train_search,
    Y_train_search,
    y_train_search,
    W,
    b,
    X_val_search,
    Y_val_search,
    y_val_search,
    GDparams_search,
    best_lambda,
    gamma=gamma,
    beta=beta,
    batch_normalization=True)

plot_learning_curve(nn_dict)

"""


#----------weight intitaliztion sensitivity ---------
def weight_sensitivity(X_train,
                       Y_train,
                       y_train,
                       X_val,
                       Y_val,
                       y_val,
                       sigma,
                       GD_params_search,
                       hidden_dimensions,
                       batch_normalization=False):
    lamb = 0.005
    input_dimension = X_train.shape[0]
    output_dimensions = Y_train.shape[0]
    results = np.array(["Sigma", "Batch Normalization", "Test Accuracy"])
    #for sigma in sigmas:
    #sigma = sigmas
    W, b, gamma, beta = init_weights(input_dimension,
                                     hidden_dimensions,
                                     output_dimensions,
                                     he_initialization=False,
                                     seed=0,
                                     std=sigma)
    if batch_normalization == True:
        W_, b_, gamma_, beta_, mean_avg_, var_avg_, nn_dict = miniBatchGD(
            X_train,
            Y_train,
            y_train,
            W,
            b,
            X_val,
            Y_val,
            y_val,
            GD_params_search,
            lamb,
            gamma=gamma,
            beta=beta,
            batch_normalization=True)

    else:
        W_, b_, nn_dict = miniBatchGD(X_train,
                                      Y_train,
                                      y_train,
                                      W,
                                      b,
                                      X_val,
                                      Y_val,
                                      y_val,
                                      GD_params_search,
                                      lamb,
                                      gamma=None,
                                      beta=None,
                                      batch_normalization=False)
    plot_learning_curve(nn_dict)
    #save test accuracy


#miniBatchGD(X_trainn,
#                Y_trainn,
#                y_trainn,
#                W,
#                b,
#                X_validation,
#                Y_validation,
#                y_validation,
#                dict,
#                lmbda,
#                gamma=None,
#                beta=None,
#                batch_normalization=False,
#                alpha=0.9):

GDparams_search['n_s'] = int(2 * 45000 / GDparams_search['n_batch']
                             )  # change to fewer iteration for initial weights

#sigma = 1e-1
#sigma = 1e-3
sigmas = 1e-4
weight_sensitivity(X_train_search,
                   Y_train_search,
                   y_train_search,
                   X_val_search,
                   Y_val_search,
                   y_val_search,
                   sigmas,
                   GDparams_search,
                   hidden_dimensions,
                   batch_normalization=True)
