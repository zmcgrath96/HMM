import sys
import string
import hmm
import numpy as np

def main(args):

    file = open("./data/alllines.txt", "r")
    text = ""
    table = str.maketrans("", "", string.punctuation)
    textCorpus = []
    uniqueWords = {}

    for line in file:
        words = line.translate(table).replace("\n", "").lower().split(" ")
        # lines less than 2 cause errors.
        # lines > 50 typically converge to 0 probablity
        if len(words) <= 2 or len(words) > 50:
            continue
        textCorpus.append(words)
        for word in words:
            if word not in uniqueWords:
                uniqueWords[word] = 0
            uniqueWords[word] += 1

    file.close()

    shakeHMM = hmm.HMM(3, uniqueWords)
    a, b, pi = shakeHMM.trainHMM(textCorpus, maxIter=1000, threshold=0.0001)
    np.savetxt("output/a.out", a)
    np.savetxt("output/b.out", b)
    np.savetxt("output/pi.out", pi)


if __name__ == '__main__':
    main(sys.argv)
