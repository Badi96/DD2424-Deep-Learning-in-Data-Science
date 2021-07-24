import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
figure = 0
book_data = open('goblet_book.txt', 'r').read()
print(type(book_data))
print(len(book_data))

#print(book_data[0:1000])

book_chars = set(book_data)
k = len(book_chars)
char_to_ind = dict()
ind_to_char = dict()

for index, chatacter in enumerate(book_chars):
    char_to_ind[chatacter] = index
    ind_to_char[index] = chatacter
print(char_to_ind)


class Vanilla_RNN:
    def __init__(self,
                 book_chars,
                 K=1,
                 m=100,
                 eta=0.1,
                 seq_length=25,
                 sigm=0.01):
        self.m = m
        self.eta = eta
        self.K = K
        self.seq_length = seq_length
        self.b = np.zeros((m, 1))  # Bias vector b
        self.c = np.zeros((k, 1))  # Bias vector c
        self.u = np.random.rand(m, k) * sigm  # Weight matrix u
        self.w = np.random.rand(m, m) * sigm  # Weight matrix w
        self.v = np.random.rand(k, m) * sigm  # Weight matrix v

        self.gradients = {
            'dLdU': np.zeros((self.m, self.K)),
            'dLdW': np.zeros((self.m, self.m)),
            'dLdV': np.zeros((self.K, self.m)),
            'dLdb': np.zeros((self.m, 1)),
            'dLdc': np.zeros((self.K, 1))
        }
        self.numerical_gradinets = copy.deepcopy

        #?
        self.grad_diff = copy.deepcopy(self.gradients)
        self.m_theta = copy.deepcopy(self.gradients)

        for index, chatacter in enumerate(book_chars):
            self.char_to_ind[chatacter] = index
            self.ind_to_char[index] = chatacter
