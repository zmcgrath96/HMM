import sys
import string
import hmm
import numpy as np

def main(args):

    if len(args) < 2:
        printUsage()

    # training
    if args[1] == "-t":
        train()
    elif args[1] == "-g":
        # load stored matrices
        if len(args) != 3:
            printUsage()
        shakeHMM = load_trained_hmm()
        print(shakeHMM.genText(int(args[2])))


def printUsage():
    print("Usage:\n\t-t\n\t\tBegins hmm training and saves matices to output folder")
    print("\t-g <number of words>\n\t\tGenerates words using saved matrices")
    print("\t-p <number of words to predict> [sequence of words]")
    print("\t\tPredicts next words gicen sequence of words")
    sys.exit(1)

def train():
    file = open("./data/alllines.txt", "r")
    text = ""
    table = str.maketrans("", "", string.punctuation)
    textCorpus = []
    uniqueWords = {}

    for line in file:
        words = line.translate(table).replace("\n", "").lower().split(" ")
        # lines less than 2 cause errors.
        # lines > 60 typically converge to 0 probablity
        if len(words) <= 2 or len(words) > 60:
            continue
        textCorpus.append(words)
        for word in words:
            if word not in uniqueWords:
                uniqueWords[word] = 0
            uniqueWords[word] += 1
    file.close()

    shakeHMM = hmm.HMM(use="t", states=3, obs=uniqueWords)
    word, index = shakeHMM.getWordMaps()
    saveDict(word, "output/word_map.out")
    saveDict(index, "output/index_map.out")
    a, b, pi = shakeHMM.trainHMM(textCorpus, maxIter=1000, threshold=0.0001)
    if a is not None:
        np.savetxt("output/a.out", a)
        np.savetxt("output/b.out", b)
        np.savetxt("output/pi.out", pi)

def saveDict(dict, location):
    file = open(location, "w")
    for key in dict:
        file.write(str(key) + " " + str(dict[key]) + "\n")
    file.close()

def load_trained_hmm():
    a = np.loadtxt("output/a.out")
    shape_a = a.shape
    if len(shape_a) != 2 or shape_a[0] != shape_a[1]:
        print("Error: State transition matrix a is of dimensions {}".format(shape_a))
        sys.exit(1)
    numStates = shape_a[0]

    b = np.loadtxt("output/b.out")
    shape_b = b.shape
    if len(shape_b) != 2:
        print("Error: Matrix b is not 2-d")
        sys.exit(1)
    if shape_b[0] != numStates:
        print("Error: a and b matrix do not have the same number of states")
        sys.exit(1)
    numObvs = shape_b[1]

    pi = np.loadtxt("output/pi.out")
    shape_pi = pi.shape
    if  len(shape_pi) != 1:
        print("Error: Matrix pi is not 1-d")
        sys.exit(1)
    if shape_pi[0] != numStates:
        print("Error: pi and a matrix do not have the same number of states")
        sys.exit(1)

    word = {}
    fileW = open("output/word_map.out")
    for line in fileW:
        line = line.replace("\n", "")
        line_split = line.split(" ")
        if len(line_split) != 2:
            print("Error processing word_map.out: line was not a key value pair")
            sys.exit(1)
        word[line_split[0]] = int(line_split[1])
    fileW.close()
    if len(word) != numObvs:
        print("Error: word_map and b matrix do not have the same number of observations")
        sys.exit(1)

    index = {}
    fileI = open("output/index_map.out")
    for line in fileI:
        line = line.replace("\n", "")
        line_split = line.split(" ")
        if len(line_split) != 2:
            print("Error processing index_map.out: line was not a key value pair")
            sys.exit(1)
        index[int(line_split[0])] = line_split[1]
    fileI.close()
    if len(index) != numObvs:
        print("Error: index_map and b matrix do not have the same number of observations")
        sys.exit(1)
    return hmm.HMM(use="g", states=numStates, a=a, b=b, pi=pi, index=index, word=word)

if __name__ == '__main__':
    main(sys.argv)
