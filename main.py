import sys
import string
import hmm

def main(args):

    file = open("./data/alllines.txt", "r")
    text = ""
    table = str.maketrans("", "", string.punctuation)
    textCorpus = []
    uniqueWords = set({})
    for line, n in zip(file, range(100000)):
        words = line.translate(table).replace("\n", "").lower().split(" ")
        textCorpus.append(words)
        for word in words:
            uniqueWords.add(word)
        if n > 50000:
            break
    file.close()

    shakeHMM = hmm.HMM(2, uniqueWords)
    shakeHMM.trainHMM(textCorpus)


if __name__ == '__main__':
    main(sys.argv)
