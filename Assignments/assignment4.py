import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import squeeze


class Vanilla_RNN:
    def __init__(self, book_data, m=100, eta=0.1, seq_length=25, sigma=0.01):
        # K=1,
        self.m = m
        self.eta = eta
        self.book_data = book_data
        self.unique_book_data = set(book_data)
        self.char_to_ind = {}
        self.ind_to_char = {}

        for index, item in enumerate(self.unique_book_data):
            self.char_to_ind[item] = index
            self.ind_to_char[index] = item

        self.K = len(self.char_to_ind)
        self.seq_length = seq_length
        self.b = np.zeros((m, 1))  # Bias vector b
        self.c = np.zeros((self.K, 1))  # Bias vector c
        self.U = np.random.rand(m, self.K) * sigma  # Weight matrix U
        self.W = np.random.rand(m, m) * sigma  # Weight matrix W
        self.V = np.random.rand(self.K, m) * sigma  # Weight matrix V

        # dictionary for gradients
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

    def softmax(self, a):
        s = np.clip(a, -700, 700)
        return np.exp(s) / np.sum(np.exp(s), axis=0)

    def one_hot_encoding(self, vector):
        matrix = np.zeros((len(self.char_to_ind), len(vector)))
        for i in range(len(vector)):
            matrix[self.char_to_ind[vector[i]], i] = 1
        return matrix

    def synthesize(self, x_0, h_0, num_samples):
        x = np.copy(x_0)
        h = np.copy(h_0)[:, np.newaxis]
        samples = np.zeros((x_0.shape[0], num_samples))
        for t in range(num_samples):
            a = self.W @ h + self.U @ x + self.b
            h = np.tanh(a)
            o = self.V @ h + self.c
            p = self.softmax(o)
            # Select random character according to probabilities
            chosen_characters = np.random.choice(range(x.shape[0]),
                                                 1,
                                                 p=p.flatten())
            x = np.zeros(x.shape)
            x[chosen_characters] = 1
            samples[:, t] = x.flatten()

        return samples

    def forward_pass(self, x, h_0):
        #print("shape of x is: ", x.shape)
        h = np.zeros((h_0.shape[0], x.shape[1]))
        a = np.zeros((h_0.shape[0], x.shape[1]))
        probability_t = np.zeros(x.shape)
        for t in range(x.shape[1]):
            if t == 0:
                a[:, t] = (self.W @ h_0[:, np.newaxis] +
                           self.U @ x[:, t][:, np.newaxis] + self.b).flatten()
            else:
                a[:, t] = (self.W @ h[:, t - 1][:, np.newaxis] +
                           self.U @ x[:, t][:, np.newaxis] + self.b).flatten()
            h[:, t] = np.tanh(a[:, t])
            o = self.V @ h[:, t][:, np.newaxis] + self.c
            p = self.softmax(o)
            probability_t[:, t] = p.flatten()

        return probability_t, h, a

    def back_propagation(self, y, p, h, h_prev, a, x):
        #gradient lists
        grad_h = list()
        grad_a = list()
        # Computation the final gradient of o
        grad_o = -(y - p).T
        # Compute the final gradients of h and a
        grad_h.append(grad_o[-1][np.newaxis, :] @ self.V)
        grad_a.append(
            (grad_h[-1] @ np.diag(1 - np.power(np.tanh(a[:, -1]), 2))))
        # Computing the remaining gradients (o, h, and a)
        for t in reversed(range(y.shape[1] - 1)):
            grad_h.append(grad_o[t][np.newaxis, :] @ self.V +
                          grad_a[-1] @ self.W)
            grad_a.append(
                grad_h[-1] @ np.diag(1 - np.power(np.tanh(a[:, t]), 2)))

        grad_a.reverse()  # Reverse a gradient so it goes forward_passs
        grad_a = np.vstack(grad_a)  # stack gradien in array sequence
        #store gradients in dictionary
        self.gradients['V'] = grad_o.T @ h.T
        h_aux = np.zeros(h.shape)  # Auxiliar h matrix that includes h_prev
        h_aux[:, 0] = h_prev
        h_aux[:, 1:] = h[:, 0:-1]
        self.gradients['W'] = grad_a.T @ h_aux.T
        self.gradients['U'] = grad_a.T @ x.T
        self.gradients['b'] = np.sum(grad_a, axis=0)[:, np.newaxis]
        self.gradients['c'] = np.sum(grad_o, axis=0)[:, np.newaxis]
        return self.gradients

    def compute_loss(self, y, p):
        return -np.sum(np.log(np.sum(y * p, axis=0)))

    def generate_text(self, Y):
        string = []
        for i in range(Y.shape[1]):
            string.append(self.ind_to_char[int(np.argmax(Y[:, i]))])
        return ''.join(string)

    def clip_gradients(self):
        for key in self.gradients:
            self.gradients[key] = np.maximum(
                np.minimum(self.gradients[key], 5), -5)

    def ada_grad(self):
        """ ADAGRad algorithm for optimization"""
        for param in ['b', 'c', 'U', 'W', 'V']:
            self.m_theta[
                param] = self.m_theta[param] + self.gradients[param]**2
            denom = (self.m_theta[param] + 1e-10)**-0.5
            vars(self)[param] = vars(self)[param] - self.eta * np.multiply(
                denom, self.gradients[param])

    def SGD(self, num_epoch):
        print("in SGD")
        #h0 = np.zeros((self.m, 1))
        smooth_loss_list = []
        loss_list = []
        smooth_loss = 0
        num_iterations = 0
        max_iterations = 100000
        hprev = np.zeros((self.m))
        for epoch in range(num_epoch):

            print("-----------------")
            print("epoch: ", epoch)
            e = 0
            if num_iterations >= max_iterations:
                break
            while e < len(self.book_data) - self.seq_length:
                if num_iterations >= max_iterations:
                    break
                #inputs
                X = self.book_data[e:e + self.seq_length]
                #Labeled outputs
                Y = self.book_data[e + 1:e + self.seq_length + 1]
                x0 = self.one_hot_encoding(X)
                y0 = self.one_hot_encoding(Y)
                #forward_pass propagation of the RNN
                probability_t, h, a = self.forward_pass(x0, hprev)
                loss = self.compute_loss(y0, probability_t)

                #backward propagation for gradients
                self.gradients = self.back_propagation(y0, probability_t, h,
                                                       hprev, a, x0)
                #optimizing the RNN
                self.clip_gradients()
                self.ada_grad()

                # for saving the best model
                if num_iterations == 0:
                    smooth_loss = loss
                    best_rnn = copy.deepcopy(rnn)
                    best_loss = copy.deepcopy(smooth_loss)
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                    if smooth_loss < best_loss:
                        best_rnn = copy.deepcopy(rnn)
                        best_loss = smooth_loss
                loss_list.append(loss)
                smooth_loss_list.append(smooth_loss)

                e = e + self.seq_length
                if num_iterations == 0:
                    print('loss at iteration ' + str(num_iterations) +
                          ' is: ' + str(loss))
                    input_ = rnn.one_hot_encoding(self.book_data[e])
                    op = rnn.synthesize(input_, hprev, 200)
                    print('\nGenerated text till now: ')
                    print(rnn.generate_text(op))
                    print('\n')

                num_iterations += 1
                if num_iterations % 5000 == 0:
                    print('smooth loss at iteration ' + str(num_iterations) +
                          'is: ' + str(smooth_loss))
                if num_iterations % 10000 == 0:
                    print('smooth loss at iteration ' + str(num_iterations) +
                          'is: ' + str(smooth_loss))
                    input_ = rnn.one_hot_encoding(self.book_data[e])
                    op = rnn.synthesize(input_, hprev, 200)
                    print('\nGenerated text till now: ')
                    print(rnn.generate_text(op))
                    print('\n')
                #save last h for next iteration
                hprev = h[:, -1]
            hprev = np.zeros(hprev.shape)  #or np.shape(hprev)
        print("Best loss: ", best_loss)
        h_prev = np.zeros(rnn.m)
        input_ = rnn.one_hot_encoding(self.book_data[-1])
        op = best_rnn.synthesize(input_, h_prev, 1000)
        print('\nBest model Generated text: ')
        print(best_rnn.generate_text(op))
        print('\n')
        return loss_list, smooth_loss_list, num_iterations, best_rnn, best_loss


def ComputeGradsNum(RNN_object, X, Y, h0, h=1e-4):
    # Iterate parameters and compute gradients numerically
    GRADS = dict()
    for parameter in ['b', 'c', 'U', 'W', 'V']:
        GRADS[parameter] = np.zeros_like(vars(RNN_object)[parameter])
        for i in range(vars(RNN_object)[parameter].shape[0]):
            for j in range(vars(RNN_object)[parameter].shape[1]):
                RNN_try = copy.deepcopy(RNN_object)
                vars(RNN_try)[parameter][i, j] += h
                p_t2, _, _ = RNN_try.forward_pass(X, h0)
                loss2 = RNN_try.compute_loss(Y, p_t2)
                vars(RNN_try)[parameter][i, j] -= 2 * h
                p_t1, _, _ = RNN_try.forward_pass(X, h0)
                loss1 = RNN_try.compute_loss(Y, p_t1)
                #loss1, _, _ = RNN_try.fwd_pass(X, Y, h0)
                GRADS[parameter][i, j] = (loss2 - loss1) / (2 * h)
    return GRADS


def plot_losses(losses_list, smooth_losses_list, num_iterations):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    #ax[0].plot(num_iterations, np.squeeze(losses_list), label="Loss curve")
    ax[0].plot(np.arange(len(losses_list)),
               np.squeeze(losses_list),
               label="Loss curve")
    ax[0].legend()
    ax[0].set(xlabel='Iterations', ylabel='Total Loss')
    ax[0].grid()
    ax[1].plot(np.arange(len(smooth_losses_list)),
               np.squeeze(smooth_losses_list),
               label="Smooth Loss curve")
    ax[1].legend()
    ax[1].set(xlabel='Iterations', ylabel='Smooth Loss')
    ax[1].grid()
    plt.show()


f = open("goblet_book.txt", "r")
book_data = f.read()
f.close()
rnn = Vanilla_RNN(book_data)
epochs = 3
# 5. RNN using AdaGrad for 100 000 iteration and get fianl (best) result
loss_list, smooth_loss_list, num_iterations, best_rnn, best_loss = rnn.SGD(
    epochs)
print("Best loss is: ", best_loss)

plot_losses(loss_list, smooth_loss_list, num_iterations)
"""
#######
# 3. synthesize text from randomly initialized rnn"
e = 0
X = rnn.book_data[e:e + rnn.seq_length]
#Labeled outputs
Y = rnn.book_data[e + 1:e + rnn.seq_length + 1]
x0 = rnn.one_hot_encoding(X)
y0 = rnn.one_hot_encoding(Y)
#hidden state
h0 = np.random.randn(rnn.m)

# 4. Implement the forward_pass & backward pass of back-prop
X = rnn.book_data[0:rnn.seq_length]
Y = rnn.book_data[1:rnn.seq_length + 1]
x0 = rnn.one_hot_encoding(X)
y0 = rnn.one_hot_encoding(Y)
h0 = np.zeros((rnn.m))

p, h, a = rnn.forward_pass(x0, h0)

newGRADS = rnn.back_propagation(y0, p, h, h0, a, x0)
newGRADS_num = ComputeGradsNum(rnn, x0, y0, h0)
#check gradients
for parameter in ['b', 'c', 'U', 'W', 'V']:
    error = abs(newGRADS_num[parameter] - newGRADS[parameter])
    mean_error = np.mean(error < 1e-6)
    max_error = error.max()
    print('For '+parameter+', the % of absolute errors <1e-6 is '+str(mean_error*100)+ \
          ' with the maximum being '+str(max_error))
"""
