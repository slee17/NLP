# The following code was written by Professor Julie Medero.

from nltk import bigrams, word_tokenize
import re
import math
from nltk.corpus import gutenberg
from nltk.probability import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist

wordRE = re.compile("\w")

class AddAlphaBigramModel():
    def __init__(self, alpha=0.1):
        self.vocabulary=set()
        self.V = 0
        self.bigrams=ConditionalFreqDist([])
        self.unigrams=FreqDist([])
        self.alpha = 0.1
    def train(self):
        self.vocabulary=set()
        
        this_bigrams=[]
        self.unigrams = FreqDist([])
        
        for fileid in gutenberg.fileids():
            for sentence in gutenberg.sents(fileid):
                words=["<s>",] + [x.lower() for x in sentence if wordRE.search(x)] + ["</s>",]
                this_bigrams += bigrams(words)
                self.vocabulary.update(words)
                self.unigrams.update(words)
        self.bigrams=ConditionalFreqDist(this_bigrams)
        self.V = len(self.vocabulary)
        
    def bigram_prob(self, w1, w2):
        numerator = self.bigrams[w1][w2] + self.alpha
        denominator = self.bigrams[w1].N() + (self.alpha * self.V)
        retval= math.log(numerator / denominator)

        return retval

    def unigram_prob(self, w):
        numerator = self.unigrams[w] + self.alpha
        denominator = self.unigrams.N() + (self.alpha * self.V)
        return math.log(numerator/denominator)
    
    def __contains__(self, w):
        return w in self.vocabulary

