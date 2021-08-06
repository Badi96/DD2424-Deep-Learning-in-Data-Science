import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
from keras.preprocessing.text import Tokenizer


class Vanilla_RNN:
    def __init__(self, book_data, m=100, eta=0.1, seq_length=25, sigma=0.01):
        # K=1,
        self.m = m
        self.eta = eta
        #self.K = K
        #self.char_to_ind = dict()
        #self.ind_to_char = dict()
        #self.book_data = open('goblet_book.txt', 'r').read()
        self.book_data = book_data
        #self.book_chars = set(self.book_data)
        #for index, chatacter in enumerate(book_data):
        #    self.char_to_ind[chatacter] = index
        #    self.ind_to_char[index] = chatacter

        self.tokenizer = Tokenizer(char_level=True, lower=False)
        self.tokenizer.fit_on_texts(self.book_data)
        #sequence_of_int = tokenizer.texts_to_sequences(book_data)
        #print(sequence_of_int)
        self.char_to_ind = self.tokenizer.word_index
        self.ind_to_char = self.tokenizer.index_word

        self.K = len(self.char_to_ind)
        self.seq_length = seq_length
        self.b = np.zeros((m, 1))  # Bias vector b
        self.c = np.zeros((self.K, 1))  # Bias vector c
        self.U = np.random.rand(m, self.K) * sigma  # Weight matrix U
        self.W = np.random.rand(m, m) * sigma  # Weight matrix W
        self.V = np.random.rand(self.K, m) * sigma  # Weight matrix V

        #self.gradients = {
        #    'dLdU': np.zeros((self.m, self.K)),
        #    'dLdW': np.zeros((self.m, self.m)),
        #    'dLdV': np.zeros((self.K, self.m)),
        #    'dLdb': np.zeros((self.m, 1)),
        #    'dLdc': np.zeros((self.K, 1))
        #}
        self.gradients = {
            'U': np.zeros((self.m, self.K)),
            'W': np.zeros((self.m, self.m)),
            'V': np.zeros((self.K, self.m)),
            'b': np.zeros((self.m, 1)),
            'c': np.zeros((self.K, 1))
        }
        self.numerical_gradinets = copy.deepcopy(self.gradients)
        self.grad_diff = copy.deepcopy(self.gradients)
        self.m_theta = copy.deepcopy(self.gradients)

        #k = len(self.book_chars)
        #self.char_to_ind = dict()
        #self.ind_to_char = dict()
        #for index, chatacter in enumerate(book_chars):
        #    self.char_to_ind[chatacter] = index
        #    self.ind_to_char[index] = chatacter

    def softmax(self, o_t):
        p = np.exp(o_t)
        if np.sum(p, axis=0) == 0:
            #remove later
            print('WARNING: zero in p')
        else:
            p = p / np.sum(p, axis=0)
        #return p / np.sum(p, axis=0)
        return p

    """
    def one_hot_encoding(self, char_index):
        # Use CHAR_TO_INDEX when calling this function!!
        char_one_hot = np.zeros((self.K, 1))
        char_one_hot[char_index, 0] = 1
        return char_one_hot
    """

    def one_hot_encoding(self, ch):
        print("size of ch", ch)

        x = []
        for c in ch:
            x0 = np.zeros((self.K, 1))
            x0[self.char_to_ind[c] - 1] = 1
            x.append(x0)
        y = np.array(x)
        y = np.squeeze(y)
        y = y.T
        if len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))
        return y

    def synthesize(self, x0, h0, length):
        Y = []
        x_next = x0
        for i in range(length):
            a_t = self.W @ h0 + self.U @ x_next + self.b
            h_t = np.tanh(a_t)
            o_t = self.V @ h_t + self.c
            print("W shape: ", self.W.shape)
            print("h0 shape: ", h0.shape)

            print("U shape: ", self.U.shape)
            print("x_next shape: ", x_next.shape)
            print("b shape: ", self.b.shape)
            #a_t = np.dot(self.W, h0) + np.dot(self.U, x_next) + self.b
            #h_t = np.tanh(a_t)
            #o_t = np.dot(self.V, h_t) + self.c
            p_t = self.softmax(o_t)
            c = np.cumsum(p_t)
            r = np.random.rand()
            ii = np.where(c - r > 0)[0][0]
            x_next = self.one_hot_encoding(self.ind_to_char[ii + 1])
            #xnext = np.random.multinomial(
            #    1,
            #    np.squeeze(p),
            #)[:, np.newaxis]
            # save one-hpt encoding of created next sequence
            #Y[:, [i]] = xnext[:, [0]]
            Y.append(x_next)
        Y = np.array(Y)
        Y = np.squeeze(Y)
        Y = Y.T
        #Y = (np.squeeze(np.array(Y))).T
        return Y

    """
    def synthesize(self, x0, h0, length, stop_character_one_hot=None):
        Y = np.zeros(shape=(self.output_size, length))
        x = x0
        h_prev = h0
        for t in range(length):
            a = self.W @ h_prev + self.U @ x + self.b
            h = self.tanh(a)
            o = self.V @ h + self.c
            p = self.softmax(o)
            # Create next sequence input randomly from predicted output distribution
            x = np.random.multinomial(1, np.squeeze(p))[:, np.newaxis]
            # Save the one-hot encoding of created next sequence input
            Y[:, [t]] = x[:, [0]]
            # Break loop if created next sequence input is equal to given stop character
            if all(x == stop_character_one_hot):
                Y = Y[:, 0:(t + 1)]
                break
            # Update previous hidden state for next sequence iteration
            h_prev = h
        return Y
    """
    """

    def forward_pass(self, x0, y0, h0):
        h = np.zeros((h0.shape[0], x0.shape[1]))
        a = copy.deepcopy(h)
        probabilities = np.zeros(x0.shape)
        for t in range(x0.shape[1]):
            if t == 0:
                a[:,
                  t] = (self.W @ h0[:, np.newaxis] +
                        self.U @ x0[:, t][:, np.newaxis] + self.b).flatten()
            else:
                a[:,
                  t] = (self.W @ self.h[:, t - 1][:, np.newaxis] +
                        self.U @ x0[:, t][:, np.newaxis] + self.b).flatten()
            h[:, t] = np.tanh(a[:, t])
            self.o = self.V @ h[:, t][:, np.newaxis] + self.c
            p = self.softmax(self.o)
            probabilities[:, t] = p.flatten()
        return probabilities, h, a
    """

    def fwd_pass(self, x0, y0, h0):
        # No y0 NEEDED!
        h = [h0]
        loss = 0
        #at = []
        probability_t = []
        sequence_length = x0.shape[1]
        for t in range(sequence_length):
            #try @ and see if it works..
            a = np.dot(self.W, h[t]) + np.dot(self.U,
                                              np.reshape(x0[:, t],
                                                         (-1, 1))) + self.b
            #    params['U'], np.reshape(x0[:, t], (-1, 1))) + params['b']
            #if t == 0:
            #    a = self.W @ h0 + self.U @ x0[:, t] + self.b
            #else:
            #a = self.W @ h[t] + self.U @ x0[:, t] + self.b
            h.append(np.tanh(a))
            o = np.dot(self.V, h[t + 1]) + self.c
            #o = self.V @ h[t + 1] + self.c
            probability_t.append(self.softmax(o))
            #loss += np.log(y0[:, [t]].T @ probability_t[t])[0, 0]
            loss -= np.log(y0[:, [t]].T @ probability_t[t])[0, 0]

        return loss, h, probability_t

    def back_propagation(self, x, y, h, p):
        dLdo = []

        for t in range(x.shape[1]):
            dLdo.append(-(np.reshape(y[:, t], (-1, 1)).T - p[t].T))
            #self.gradients['dLdV'] += np.dot(dLdo[t].T, h[t + 1].T)
            #self.gradients['dLdc'] += dLdo[t].T
            self.gradients['V'] += np.dot(dLdo[t].T, h[t + 1].T)
            self.gradients['c'] += dLdo[t].T
        dLda = np.zeros((1, self.m))

        for t in range(x.shape[1] - 1, -1, -1):
            dLdh = np.dot(dLdo[t], self.V) + np.dot(dLda, self.W)
            dLda = np.dot(dLdh, np.diag(1 - h[t + 1][:, 0]**2))

            #self.gradients['dLdW'] += np.dot(dLda.T, h[t].T)
            #self.gradients['dLdU'] += np.dot(dLda.T,
            #                                 np.reshape(x[:, t], (-1, 1)).T)
            #self.gradients['dLdb'] += dLda.T

            self.gradients['W'] += np.dot(dLda.T, h[t].T)
            self.gradients['U'] += np.dot(dLda.T,
                                          np.reshape(x[:, t], (-1, 1)).T)
            self.gradients['b'] += dLda.T
        #return None
        return self.gradients

    def generate_text(self, Y):
        # one hot encoding to text
        ind = np.argmax(Y, axis=0)
        string = []
        for i in range(ind.shape[0]):
            string.append(rnn.ind_to_char[ind[i] + 1])
        return ''.join(string)

    """
    def compute_cost(self, y0, pt):
        #h, pt = self.fwd_pass(x0, y0, h0, params)

        # Cross entropy loss
        loss = 0
        for t in range(len(pt)):
            y = np.reshape(y0.T[t], (-1, 1))
            loss -= sum(np.log(np.dot(y.T, pt[t])))
            if loss == np.inf:
                print(
                    'WARNING: Loss going to inf, handling by assigning zero value'
                )
                loss = 0
        return loss
    """


def ComputeGradsNum(RNN_object, X, Y, h0, h=1e-4):
    # Iterate parameters and compute gradients numerically
    GRADS = dict()
    for parameter in ['b', 'c', 'U', 'W', 'V']:
        GRADS[parameter] = np.zeros_like(vars(RNN_object)[parameter])
        for i in range(vars(RNN_object)[parameter].shape[0]):
            for j in range(vars(RNN_object)[parameter].shape[1]):
                RNN_try = copy.deepcopy(RNN_object)
                vars(RNN_try)[parameter][i, j] += h
                loss2, _, _ = RNN_try.fwd_pass(X, Y, h0)
                vars(RNN_try)[parameter][i, j] -= 2 * h
                loss1, _, _ = RNN_try.fwd_pass(X, Y, h0)
                GRADS[parameter][i, j] = (loss2 - loss1) / (2 * h)
    return GRADS


f = open("goblet_book.txt", "r")
book_data = f.read()
#print(book_data)
f.close()
rnn = Vanilla_RNN(book_data)

# 3. synthesize text from randomly initialized rnn"
x0 = rnn.one_hot_encoding('.')
#0 = rnn.one_hot_encoding(book_data)

h0 = np.random.randn(rnn.m, 1)
n = 10

Y = rnn.synthesize(x0, h0, n)
print(rnn.generate_text(Y))

# 4. Implement the forward & backward pass of back-prop
X = rnn.book_data[0:rnn.seq_length]
Y = rnn.book_data[1:rnn.seq_length + 1]
x0 = rnn.one_hot_encoding(X)
y0 = rnn.one_hot_encoding(Y)
h0 = np.zeros((rnn.m, 1))

_, h, p = rnn.fwd_pass(x0, y0, h0)
#rnn.back_propagation(x0, y0, h, p)

newGRADS = rnn.back_propagation(x0, y0, h, p)

newGRADS_num = ComputeGradsNum(rnn, x0, y0, h0)

for parameter in ['b', 'c', 'U', 'W', 'V']:
    error = abs(newGRADS_num[parameter] - newGRADS[parameter])
    mean_error = np.mean(error < 1e-6)
    max_error = error.max()
    print('For '+parameter+', the % of absolute errors <1e-6 is '+str(mean_error*100)+ \
          ' and the maximum is '+str(max_error))
"""book_data = open('goblet_book.txt', 'r').read()

book_chars = set(book_data)
rnn = Vanilla_RNN(book_data)

seq_length = 25
X_chars = rnn.book_data[0:seq_length]
Y_chars = rnn.book_data[1:(1 + seq_length)]

myRNN = Vanilla_RNN(book_chars)

X = rnn.book_data[0:myRNN.seq_length]
Y = rnn.book_data[1:myRNN.seq_length + 1]
x0 = rnn.one_hot_encoding(X)
y0 = rnn.one_hot_encoding(Y)
h0 = np.zeros((rnn.m, 1))

h0 = np.zeros(shape=(rnn.m, 1))
loss, h, p = rnn.fwd_pass(x0, y0, h0)

newGRADS = rnn.back_propagation(x0, y0, h0, p)

newGRADS_num = ComputeGradsNum(rnn, x0, y0, h0)

for parameter in ['b', 'c', 'U', 'W', 'V']:
    error = abs(newGRADS_num[parameter] - newGRADS[parameter])
    mean_error = np.mean(error < 1e-6)
    max_error = error.max()
    print('For '+parameter+', the % of absolute errors <1e-6 is '+str(mean_error*100)+ \
          ' and the maximum is '+str(max_error))
"""
