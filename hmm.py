import string
import numpy as np
import copy
import math
import sys
import threading
import time

class HMM:
    def __init__(self, use, states=None, obs=None, a=None, b=None, pi=None, index=None, word=None):
        # training constructor
        if use == "t":
            self.numStates = states
            self.numObvs = len(obs)
            self.a = np.random.rand(self.numStates, self.numStates)
            self.b = np.zeros((self.numStates, self.numObvs))
            self.pi = np.random.rand(self.numStates,)
            self.word_map = self.make_word_map(obs)
            self.index_map = self.make_index_map()

            # fix a
            for i in range(self.numStates):
                self.a[i,:] = self.a[i,:] / np.sum(self.a[i,:])

            # fix pi
            self.pi = self.pi / np.sum(self.pi)

            #fill b
            sum = 0
            for i in range(self.numStates):
                for word in obs:
                    self.b[i, self.word_index(word)] = obs[word]
                    sum += obs[word]
            self.b = self.b / sum

        else:
            self.numStates = states
            self.a = a
            self.b = b
            self.pi = pi
            self.word_map = word
            self.index_map = index
            self.numObvs = len(self.word_map)



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
                if n % 50000 == 0:
                    print("Line {}".format(n))
                alpha = self.forwardBaumWelch(seq)
                beta = self.backwardBaumWelch(seq)
                gammas.append(self.getGamma(alpha, beta, seq))
                xis.append(self.getXi(alpha, beta, seq))

            print("finished e-step")

            # maximization

            # pi
            print("maxing pi")
            self.pi = np.zeros((self.numStates,))
            for i in range(self.numStates):
                for T in range(len(textCorpus)):
                    self.pi[i] += gammas[T][i,0]
            sum = np.sum(self.pi)
            self.pi = self.pi / sum

            # a
            print("maxing a")
            for i in range(self.numStates):
                denom = 0
                for T, seq in zip(range(len(text_by_index)), text_by_index):
                    for t in range(len(seq) - 1):
                        denom += gammas[T][i,t]

                for j in range(self.numStates):
                    numer = 0
                    for T, seq in zip(range(len(text_by_index)), text_by_index):
                        for t in range(len(seq)):
                            numer += xis[T][i,j,t]
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

            if not self.sanityCheck():
                print("Error training hmm")
                return None, None, None
        return self.a, self.b, self.pi


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
            sumPi += self.pi[i]
        if (int(round(sumPi)) != 1):
            print("Pi: sumPi: {}".format(sumPi))
            return False
        return True


    def forwardBaumWelch(self, seq):
        alpha = np.zeros((self.numStates, len(seq)))
        alpha[:,0] = self.pi[:] * self.b[:,seq[0]]

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

        for t in range(len(seq)):
            denom = 0
            for i in range(self.numStates):
                gamma[i,t] = alpha[i,t] * beta[i,t]
                denom += gamma[i,t]
            gamma[:,t] = gamma[:,t] / denom
        return gamma

    def getXi(self, alpha, beta, seq):
        xi = np.zeros((self.numStates, self.numStates, len(seq)))
        for t in range(len(seq)-1):
            denom = 0
            for i in range(self.numStates):
                for j in range(self.numStates):
                    xi[i,j,t] = alpha[i,t] * self.a[i,j] * beta[j,t+1] * self.b[j,seq[t+1]]
                    denom += xi[i,j,t]
            xi[:,:,t] = xi[:,:,t] / denom
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
            index = self.word_map[word]
            map[index] = word
        return map

    def text_to_index(self, text):
        text_by_index = []
        for seq in text:
            seq_by_index = []
            for word in seq:
                seq_by_index.append(self.word_index(word))
            text_by_index.append(seq_by_index)
        return text_by_index

    def getWordMaps(self):
        return self.word_map, self.index_map

    def genText(self, length):
        state = np.random.choice(self.numStates, p=self.pi[:])
        seq = ""
        for i in range(length):
            index = np.random.choice(self.numObvs, p=self.b[state,:])
            seq += self.index_word(index) + " "
            state = np.random.choice(self.numStates, p=self.a[state,:])
        return seq
