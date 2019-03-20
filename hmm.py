import string
import numpy as np
import copy
import math
import sys
from queue import Queue
import threading
import time

class HMM:
    def __init__(self, states, uniqueWords):
        self.numStates = states
        self.numObvs = len(uniqueWords)
        self.a = np.random.rand(self.numStates, self.numStates)
        self.b = np.zeros((self.numStates, self.numObvs))
        self.pi = np.random.rand(self.numStates, 1)
        self.word_map = self.make_word_map(uniqueWords)
        self.index_map = self.make_index_map()
        # fix a
        for i in range(self.numStates):
            sum = np.sum(self.a[i,:])
            for j in range(self.numStates):
                self.a[i,j] = self.a[i,j] / sum

        # fix pi
        self.pi = self.pi / np.sum(self.pi)

        #fill b
        sum = 0
        for i in range(self.numStates):
            for word in uniqueWords:
                self.b[i, self.word_index(word)] = uniqueWords[word]
                sum += uniqueWords[word]
        self.b = self.b / sum

    def trainHMM(self, textCorpus, maxIter=1000, threshold=0.00001):
        text_by_index = self.text_to_index(textCorpus)
        for it in range(maxIter):
            start = time.time()
            print("Iteration {}".format(it))
            # save old stuff
            old_a = copy.copy(self.a)
            old_b = copy.copy(self.b)
            old_pi = copy.copy(self.pi)
            gammas = []
            xis = []

            # put all lines in queue
            for seq, n in zip(text_by_index, range(len(text_by_index))):
                if n % 10000 == 0:
                    print("Line {}".format(n))
                alpha = self.forwardBaumWelch(seq)
                beta = self.backwardBaumWelch(seq)
                gammas.append(self.getGamma(alpha, beta, seq))
                xis.append(self.getXi(alpha, beta, seq))

            print("finished e-step")

            # maximization

            # pi
            print("maxing pi")
            self.pi = np.zeros((self.numStates, 1))
            for i in range(self.numStates):
                for T in range(len(textCorpus)):
                    self.pi[i,0] += gammas[T][i,0]
            sum = np.sum(self.pi)
            self.pi[:,0] = self.pi[:,0] / sum

            # a
            print("maxing a")
            for i in range(self.numStates):
                denom = 0
                for seq, T in zip(text_by_index, range(len(text_by_index))):
                    denom += np.sum(gammas[T][i,0:-2])

                for j in range(self.numStates):
                    numer = 0
                    for seq, T in zip(text_by_index, range(len(text_by_index))):
                        numer += np.sum(xis[T][i,j,0:-2])
                    self.a[i,j] = numer / denom

            # b
            print("maxing b")
            self.b = np.zeros((self.numStates, self.numObvs))
            for i in range(self.numStates):
                denom = 0
                for seq, T in zip(text_by_index, range(len(text_by_index))):
                    denom += np.sum(gammas[T][i,:])
                    for word, t in zip(seq, range(len(seq))):
                        self.b[i, word] += gammas[T][i,t]

                self.b[i,:] = self.b[i,:] / denom

            print("done maxing")
            diffA = np.absolute(np.linalg.norm(self.a) - np.linalg.norm(old_a))
            diffB = np.absolute(np.linalg.norm(self.b) - np.linalg.norm(old_b))
            diffPi = np.absolute(np.linalg.norm(self.pi) - np.linalg.norm(old_pi))
            total_diff = diffA + diffB + diffPi
            print("Total diff: {}".format(total_diff))
            print("Total time: {}".format(time.time() - start))
            if total_diff <= threshold:
                break

        if self.sanityCheck():
            return self.a, self.b, self.pi
        return None, None, None


    def sanityCheck(self):
        for i in range(self.numStates):
            sumA = 0.0
            for j in range(self.numStates):
                sumA += self.a[i,j]
            if (int(round(sumA)) != 1):
                print("A: i: {}, sumA: {}".format(i, int(round(sumA))))
                return False

        for i in range(self.numStates):
            sumB = 0.0
            for j in range(self.numObvs):
                sumB += self.b[i,j]
            if (int(round(sumB)) != 1):
                print("B: i: {}, sumB: {}".format(i, int(round(sumB))))
                return False

        sumPi = 0.0
        for i in range(self.numStates):
            sumPi += self.pi[i,0]
        if (int(round(sumPi)) != 1):
            print("Pi: sumPi: {}".format(sumPi))
            return False
        return True


    def forwardBaumWelch(self, seq):
        alpha = np.zeros((self.numStates, len(seq)))
        alpha[:,0] = self.pi[:,0] * self.b[:,seq[0]]

        for t in range(1, len(seq)):
            for i in range(self.numStates):
                sum = 0
                for j in range(self.numStates):
                    sum += alpha[j,t-1] * self.a[j,i]
                alpha[i,t] = self.b[i,seq[t-1]] * sum
        return alpha


    def backwardBaumWelch(self, seq):
        beta = np.zeros((self.numStates, len(seq)))
        beta[:,-1] = 1

        for t in range(len(seq) - 2, -1, -1):
            for i in range(self.numStates):
                for j in range(self.numStates):
                    beta[i,t] += beta[j,t+1] * self.a[i,j] * self.b[j,seq[t+1]]
        return beta

    def getGamma(self, alpha, beta, seq):
        gamma = np.zeros((self.numStates, len(seq)))
        denom = 0
        for i in range(self.numStates):
            for t in range(len(seq)):
                gamma[i,t] = alpha[i,t] * beta[i,t]
                denom += gamma[i,t]
        gamma = gamma / denom
        return gamma

    def getXi(self, alpha, beta, seq):
        xi = np.zeros((self.numStates, self.numStates, len(seq)))
        denom = 0
        for i in range(self.numStates):
            for j in range(self.numStates):
                for t in range(len(seq)-1):
                    xi[i,j,t] = alpha[i,t] * self.a[i,j] * beta[j,t+1] * self.b[j,seq[t+1]]
                    denom += xi[i,j,t]
        xi = xi / denom
        return xi

    # keep track of word integer relationship for numpy arrays
    def make_word_map(self, uniqueWords):
        d = {}
        i = 0
        for word in uniqueWords:
            if word not in d:
                d[word] = i
                i += 1
        return d

    # return the integer associated with the word
    def word_index(self, word):
        return self.word_map[word]

    def index_word(self, index):
        return self.index_map[index]

    def make_index_map(self):
        map = {}
        for word in self.word_map:
            map[self.word_map[word]] = word

    def text_to_index(self, text):
        text_by_index = []
        for seq in text:
            seq_by_index = []
            for word in seq:
                seq_by_index.append(self.word_index(word))
            text_by_index.append(seq_by_index)
        return text_by_index
