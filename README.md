# HMM
Repository for EECS 738 Project Hidden M... Mod..

#### Authors
Zachary McGrath, Kevin Ray (github.com/kray10)

## Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/HMM
```

Install dependencies
```
$HMM> pip3 install -r requirements.txt

```
## Theory
Machine learning is a mechanism used to find relationships between points in a dataset. Complex language is simply a relationship between words and their part of speech. For example, in order for a sequence of words to constitute a sentence, it must have certain characteristics, such as verbs, nouns, adjectives, etc. Utilizing machine learning to find these relationships between states and words can be very effective. The Hidden Markov Model (HMM) can be very effective in describing these relationships. HMM relies on hidden states and observations. For the sake of modeling language, the parts of speech can be seen as the hidden states, and the sequence of observations the words in the sentence.
## Running

### Training (not recommended, already done)
```
python3 hmm_shake.py -t
```
This will read in the complete works of Shakespeare and output to file the transition, observation, and initialization matrices.

### Generating Text
```
python3 hmm_shake.py -g <length>
```
This will read in the saved training matrices and use them to output the console a string of the given length.

### Sentence Prediction
```
python3 hmm_shake.py -p <number of words to predict> [sequence of words]
```
This will read in saved training matrices and use them to output to the console the number of words to predict

## Process

### Training
Training our Hidden Markov Model made use of the Baulm-Welch algorithm. This forward/backward, expectation maximization algorithm takes in the text corpus (training data set, https://www.kaggle.com/kingburrito666/shakespeare-plays) and computes three different probability distributions: Initial state, Emission, and Transition.

The initial state distribution, or pis, is the distribution that describes the probability of starting in a certain state. The transition distribution describes the probability of moving from one hidden state to another. The emission distribution describes the probability of observing a certain observation given the current hidden state.

It was determined that for this project we would train using ten states. There were two main reason we chose this number.
1. Due to the complexity of sentence structure, we wanted to include a large enough number of states to encapsulate this complexity. According to https://en.oxforddictionaries.com/grammar/word-classes-or-parts-of-speech, there are nine typical parts of speech. Since Shakespeare's writings are hundreds of years old and not technical in nature, we decided to use ten states to encapsulate those parts of speech with an additional state to hopefully capture any variation from those parts of speech.

2. With the complexity of the Baum-Welch algorithm, training with a number of states greater than ten begin to be increasingly difficult. With ten states and a convergence of 0.0001 between all matrices, the run time reached approxiamtely 41 hours on the KU EECS servers.

During traing, only lines from the text courpus with a length between three and sixty were used. The lower limit was implemented due to an indexing issue in the forward and backward algorithms that we implemented. The upper limit was used due to constraints of the python language. As line length approached seventy characters, the values of the alpha and beta arrays would become so small that python would convert them to zero. Since the alpha and beta arrays are somewhat symetrical, when the were multiplied together the resulting gamma array would be all zeroes, implying that a sentence of that length would never occur. The easiest solution was to restrict the length of the sentences for training to avoid python rounding the probablities.

### Generating Text
Generating a length of text from the data set used to train is rather straightforward. An initial state is chosen at random from the probability distribution generated in the training process. Taking that random state, a loop then takes a random sample from the emission matrix probability distribution of that initial state. The next state uses the Viterbi algortithm using the new sequence of observations to predict the next state most likely to occur. This cycle continues for the n words desired

### Sentence Prediction
The Veterbi algorithm was used for the text prediction. After the training matrices are loaded into the HMM, the sequence of observations are passed in. The Veterbi algorithm then determines the most likely sequence of states that resulted in this sequence. This allowed us to determine the most likely final state of the sequence. From this we could determine the most likely state that would be transitioned to next and what observation is most likely to be seen in that state. That sequence could then be done again until the desired number of words had been predicted.

## Results
One issue that was encountered during sentence completion process was that the predictions would eventually converge sequence "of the lord". This is most likely due to the small number of states that we were able to process and the fact that a cycle of states exists within our max probabilities. Since this cycle exists both in the seven and ten state training output, it appears that this cycle is inherent to the training data we processed. Therefore, it seems to fit within the expected results of any prediction that eventually converges to a cycle of states.

A random sample selection was the chosen method for text generation for two reasons. The first is that if max likelihood was chosen, the initial state selection from the pis (as previously described), would always result in the same initial state. Secondly, the same problem as with cycles as referenced above would occur. After entering the cycle, using max likelihood, the same three words would be repeated. Taking a random sample from the distribution would A) ensure that a cycle would not be entered and B) still have a high likelihood of seeing some sort of sequence learned in the training set.

## Optimizations
Tagging parts of speech and modeling the more complex relationships of verbs, nouns, adjectives, etc. could lead to improvements in the modeling and prediction of speech. Being able to model these states and what sequence they happen in could lead to less jerky, more accurate representations of language.
