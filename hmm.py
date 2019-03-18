import string
import numpy as np
import copy
import math

class HMM:
    def __init__(self, states, uniqueWords):
        self.states = states
        self.observations = len(uniqueWords)
        self.transition = np.full((self.states, self.states), 1/self.states)
        self.emmission = np.full((self.states, self.observations), 1/self.observations)
        self.initialization = np.full((self.states, 1), 1/self.states)
        self.word_map = self.make_word_map(uniqueWords)

    def trainHMM(self, textCorpus, maxIter=1000, threshold=0.00001):

        for i in range(maxIter):
            # E - step
            gammas = []
            chis = []
            for sentence, n in zip(textCorpus, range(len(textCorpus))):
                if len(sentence) < 3:
                    continue
                print("Iteration: {}, Percent Complete: {:.2f}%".format(i, 100 * n / len(textCorpus)), end='\r', flush=True)
                alphas = self.forwardBaumWelch(sentence)
                betas = self.backwardBaumWelch(sentence)
                gammas.append(self.getGammas(alphas, betas, sentence))
                chis.append(self.getChis(alphas, betas, sentence))

            # save old copies of matrices commparison
            prvTransition = copy.copy(self.transition)
            prvEmmision = copy.copy(self.emmission)
            prvInitilization = copy.copy(self.initialization)

            # M - step
            # update initialization
            for i in range(self.states):
                for t in range(len(textCorpus)):
                    self.initialization[i,0] = sum(gammas[t][0][i])
            self.initialization = self.initialization / (len(textCorpus))

            # update transition
            for i in range(self.states):
                for j in range(self.states):
                    chiSums = 0
                    gammaSums = 0
                    for sentence, T in zip(textCorpus, range(len(textCorpus))):
                        for t in range(len(sentence) - 1):
                            chiSums += chis[T][t][i,j]
                            gammaSums += gammas[T][t][i]
                    self.transition[i,j] = chiSums / gammaSums

            # update emmission
            for i in range(self.states):
                for word in range(len(self.emmission)):
                    numGammaSum = 0
                    denGammaSum = 0
                    for sentence, T in zip(textCorpus, range(len(textCorpus))):
                        for t in range(len(sentence)):
                            if self.word_index(sentence[t]) == word:
                                numGammaSum += gammas[T][t][i]
                            denGammaSum += gammas[T][t][i]
                    self.emmission[i, word] = numGammaSum / denGammaSum


            diffTransition = np.absolute(np.linalg.norm(self.transition, ord='fro') -
                                         np.linalg.norm(prvTransition, ord='fro'))
            diffEmmission = np.absolute(np.linalg.norm(self.emmission, ord='fro') -
                                         np.linalg.norm(prvEmmision, ord='fro'))
            diffInitialization = np.absolute(np.linalg.norm(self.initialization, ord='fro') -
                                         np.linalg.norm(prvInitilization, ord='fro'))
            totalDiff = diffTransition + diffEmmission + diffInitialization
            print("Diff A: " + str(diffTransition))
            print("Diff B: " + str(diffEmmission))
            print("Diff I: " + str(diffInitialization))
            if totalDiff < threshold:
                break
        print(self.transition)
        print(self.emmission)
        print(self.initialization)

    def forwardBaumWelch(self, sentence):
        alphas = np.zeros((self.states, len(sentence)))
        alphas[:, [0]] = self.initialization * self.emmission[:, [self.word_index(sentence[0])]]
        for t in range(1, len(sentence)):
            for i in range(self.states):
                alphas[i, t] = self.emmission[i, [self.word_index(sentence[t])]] * sum(alphas[:,[t-1]] * self.transition[:, [i]])
        return alphas

    def backwardBaumWelch(self, sentence):
        betas = np.zeros((self.states, len(sentence)))
        betas[:, -1] = 1
        for t in range(len(sentence)-2, -1, -1):
            for i in range(self.states):
                betas[i, t] = sum(betas[:,[t+1]] * np.transpose(self.transition[[i], :]) * self.emmission[:, [self.word_index(sentence[t+1])]])
        return betas

    # derived from formula for gamma at:
    # https://genome.sph.umich.edu/w/images/b/b7/Biostat615-fall2011-lecture20.pdf
    def getGammas(self, alphas, betas, sentence):
        gammas = []
        for t in range(len(sentence)):
            gammas.append(np.zeros((self.states, 1)))
            for i in range(self.states):
                gammas[t][i] = alphas[i,t] * betas[i,t]
            gammas[t] = gammas[t] / sum(gammas[t])
        return gammas

    # derived from formula for chis at:
    # https://genome.sph.umich.edu/w/images/b/b7/Biostat615-fall2011-lecture20.pdf
    def getChis(self, alphas, betas, sentence):
        chis = []
        for t in range(len(sentence) - 1):
            chis.append(np.zeros((self.states, self.states)))
            denominator = 0
            for i in range(self.states):
                for j in range(self.states):
                    chis[t][i,j] = (alphas[i,t] * self.transition[i,j] *
                                    self.emmission[j, [self.word_index(sentence[t+1])]] *
                                    betas[j, t + 1])
                    denominator += chis[t][i,j]
            chis[t] = chis[t] / float(denominator)
        return chis

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
