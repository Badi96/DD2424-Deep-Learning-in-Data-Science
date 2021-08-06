import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
import copy


class RNN(object):
    def __init__(self, book_data):
        self.book_data = book_data

        self.tokenizer = Tokenizer(char_level=True, lower=False)
        self.tokenizer.fit_on_texts(book_data)
        #sequence_of_int = tokenizer.texts_to_sequences(book_data)
        #print(sequence_of_int)
        self.char_to_ind = self.tokenizer.word_index
        self.ind_to_char = self.tokenizer.index_word

        self.m = 100
        self.K = len(self.char_to_ind)

        self.sig = 0.01

        U = np.random.randn(self.m, self.K) * self.sig
        W = np.random.randn(self.m, self.m) * self.sig
        V = np.random.randn(self.K, self.m) * self.sig

        b = np.zeros((self.m, 1))
        c = np.zeros((self.K, 1))

        #change to self.U = ..., self.V = ... , etc.
        self.params = {'U': U, 'V': V, 'W': W, 'b': b, 'c': c}

        self.gradients = {
            'dLdU': np.zeros((self.m, self.K)),
            'dLdW': np.zeros((self.m, self.m)),
            'dLdV': np.zeros((self.K, self.m)),
            'dLdb': np.zeros((self.m, 1)),
            'dLdc': np.zeros((self.K, 1))
        }

        self.numerical_gradients = copy.deepcopy(self.gradients)
        # self.numerical_gradients = {
        #     'dLdU': np.zeros((self.m,self.K)),
        #     'dLdW': np.zeros((self.m,self.m)),
        #     'dLdV': np.zeros((self.K,self.m)),
        #     'dLdb': np.zeros((self.m, 1)),
        #     'dLdc': np.zeros((self.K, 1))
        # }

        self.grad_diff = copy.deepcopy(self.gradients)

        self.eta = 0.1
        self.seq_length = 25

        # self.m_theta = []
        # for key in self.params:
        #     self.m_theta.append(np.zeros_like(self.params[key]))
        self.m_theta = copy.deepcopy(self.gradients)

    def one_hot(self, ch):
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

    def synthesize_txt(self, x0, h0, n):
        #Y = x0
        Y = []
        xnext = x0
        for i in range(n):
            a = np.dot(self.params['W'], h0) + np.dot(self.params['U'],
                                                      xnext) + self.params['b']
            ht = np.tanh(a)
            o = np.dot(self.params['V'], ht) + self.params['c']
            p = self.softmax(o)
            c = np.cumsum(p)
            r = np.random.rand()
            ii = np.where(c - r > 0)[0][0]
            xnext = self.one_hot(self.ind_to_char[ii + 1])
            #Y = np.concatenate((Y,x0), axis=1)
            Y.append(xnext)
        Y = np.array(Y)
        Y = np.squeeze(Y)
        Y = Y.T
        return Y

    def softmax(self, o):
        p = np.exp(o)
        if np.sum(p, axis=0) == 0:
            print('WARNING: zero in p')
        else:
            p = p / np.sum(p, axis=0)
        return p

    def generate_text(self, Y):
        # one hot encoding to text
        ind = np.argmax(Y, axis=0)
        string = []
        for i in range(ind.shape[0]):
            string.append(rnn.ind_to_char[ind[i] + 1])
        return ''.join(string)

    def fwd_pass(self, x0, y0, h0, params):

        h = [h0]
        #at = []
        pt = []

        for t in range(x0.shape[1]):
            a = np.dot(params['W'], h[t]) + np.dot(
                params['U'], np.reshape(x0[:, t], (-1, 1))) + params['b']
            h.append(np.tanh(a))
            o = np.dot(params['V'], h[t + 1]) + params['c']
            pt.append(self.softmax(o))
        return h, pt

        # g = - (np.dot(y0.T, p))
        # g = - (y0 - p).T
        # L = np.sum(g.T, 1)/N
        # c = np.cumsum(p)
        # r = np.random.rand()
        # ii = np.where(c-r > 0)[0][0]

    def back_prop(self, x, y, h, p, params):
        self.gradients = {
            'dLdU': np.zeros((self.m, self.K)),
            'dLdW': np.zeros((self.m, self.m)),
            'dLdV': np.zeros((self.K, self.m)),
            'dLdb': np.zeros((self.m, 1)),
            'dLdc': np.zeros((self.K, 1))
        }

        dLdo = []

        for t in range(x.shape[1]):
            dLdo.append(-(np.reshape(y[:, t], (-1, 1)).T - p[t].T))
            #dLdo = np.reshape(dLdo, (-1,1)).T
            self.gradients['dLdV'] += np.dot(dLdo[t].T, h[t + 1].T)
            self.gradients['dLdc'] += dLdo[t].T

        dLda = np.zeros((1, self.m))

        for t in range(x.shape[1] - 1, -1, -1):
            dLdh = np.dot(dLdo[t], params['V']) + np.dot(dLda, params['W'])
            dLda = np.dot(dLdh, np.diag(1 - h[t + 1][:, 0]**2))

            self.gradients['dLdW'] += np.dot(dLda.T, h[t].T)
            self.gradients['dLdU'] += np.dot(dLda.T,
                                             np.reshape(x[:, t], (-1, 1)).T)
            self.gradients['dLdb'] += dLda.T

            #print("")

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

    def check_gradients(self):
        step = 1e-4
        for key in self.params:
            print('Computing numerical gradient for key: ' + key)
            for i in range(self.params[key].shape[1]):
                for j in range(self.params[key].shape[0]):
                    params_copy = copy.deepcopy(self.params)
                    params_copy[key][j][i] -= step
                    _, pt = self.fwd_pass(x0, y0, h0, params_copy)
                    c1 = self.compute_cost(y0, pt)

                    params_copy = copy.deepcopy(self.params)
                    params_copy[key][j][i] += step
                    _, pt = self.fwd_pass(x0, y0, h0, params_copy)
                    c2 = self.compute_cost(y0, pt)

                    self.numerical_gradients['dLd' +
                                             key][j][i] = (c2 - c1) / (2 *
                                                                       step)

        for key in self.gradients:
            self.grad_diff[
                key] = self.gradients[key] - self.numerical_gradients[key]
        print('')
        print('Absolute difference b/w actual and numerical gradients:')
        print('dLdU: ' + str(np.mean(self.grad_diff['dLdU'])))
        print('dLdW: ' + str(np.mean(self.grad_diff['dLdW'])))
        print('dLdV: ' + str(np.mean(self.grad_diff['dLdV'])))
        print('dLdb: ' + str(np.mean(self.grad_diff['dLdb'])))
        print('dLdc: ' + str(np.mean(self.grad_diff['dLdc'])))

    def clip_gradients(self):
        for key in self.gradients:
            self.gradients[key] = np.maximum(
                np.minimum(self.gradients[key], 5), -5)

    def grad_update(self, update_type):
        if update_type == 'ada':  #Ada Grad
            for key in self.params:
                self.m_theta['dLd' +
                             key] = self.m_theta['dLd' +
                                                 key] + self.gradients['dLd' +
                                                                       key]**2
                denom = (self.m_theta['dLd' + key] + 1e-10)**-0.5
                self.params[key] = self.params[key] - self.eta * np.multiply(
                    denom, self.gradients['dLd' + key])

    def stochastic_gradient_descent(self, nEpochs):
        h0 = np.zeros((self.m, 1))
        smooth_loss_list = []
        loss_list = []
        smooth_loss = 0
        n_updates = 0

        for epoch in range(nEpochs):
            print('\n')
            print('---------------------------------')
            print('EPOCH: ' + str(epoch))
            e = 0
            while e < len(self.book_data) - self.seq_length:
                X = self.book_data[e:e + self.seq_length]
                Y = self.book_data[e + 1:e + self.seq_length + 1]
                x0 = self.one_hot(X)
                y0 = self.one_hot(Y)

                if e == 0:
                    hprev = h0

                # do forward pass
                h, pt = self.fwd_pass(x0, y0, hprev, self.params)

                # do back propagation to calculate and update gradients
                self.back_prop(x0, y0, h, pt, self.params)
                # clip gradients
                self.clip_gradients()
                # use gradients to update parameters
                self.grad_update('ada')
                # compute loss
                loss = self.compute_cost(y0, pt)
                if loss is not 0:
                    if smooth_loss == 0:
                        smooth_loss = loss
                    else:
                        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                smooth_loss_list.append(smooth_loss)
                loss_list.append(loss)
                # update e
                e = e + self.seq_length

                n_updates += 1

                if n_updates % 10000 == 0:
                    print('Smoothed loss after ' + str(n_updates) +
                          'th step: ' + str(smooth_loss))
                if n_updates % 10000 == 0:
                    inp = rnn.one_hot(self.book_data[e])
                    op = rnn.synthesize_txt(inp, hprev, 200)
                    print('\nGenerated text till now: ')
                    print(rnn.generate_text(op))
                    print('\n')

                # update hprev
                hprev = copy.deepcopy(h[-1])

        return loss_list, smooth_loss_list


f = open("goblet_book.txt", "r")
book_data = f.read()
#print(book_data)
f.close()

rnn = RNN(book_data)

# 3. synthesize text from randomly initialized rnn"
x0 = rnn.one_hot('.')
h0 = np.random.randn(rnn.m, 1)
n = 10
Y = rnn.synthesize_txt(x0, h0, n)
print(rnn.generate_text(Y))

# 4. Implement the forward & backward pass of back-prop
X = rnn.book_data[0:rnn.seq_length]
Y = rnn.book_data[1:rnn.seq_length + 1]
x0 = rnn.one_hot(X)
y0 = rnn.one_hot(Y)
h0 = np.zeros((rnn.m, 1))

h, p = rnn.fwd_pass(x0, y0, h0, rnn.params)
rnn.back_prop(x0, y0, h, p, rnn.params)

# # 4.1 Gradient check
# rnn.check_gradients()

# 5 Train RNN using AdaGrad
loss_list, smooth_loss_list = rnn.stochastic_gradient_descent(50)
print('')