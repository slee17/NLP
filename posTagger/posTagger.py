import nltk, sys
from nltk.corpus import brown
from numpy import argmax, zeros, array, float32, ones
from collections import defaultdict, Counter, deque
from nltk.tag.api import TaggerI

from nltk import ConditionalFreqDist, ConditionalProbDist, MLEProbDist, FreqDist

def simple_brown_generator():
    def cleaned_tag(tag):
        tag=tag.split("-")[0]
        tag=tag.split("+")[0]
        tag=tag.rstrip("$*")
        return tag        
    for sent in brown.tagged_sents():
        yield [(word.lower(), cleaned_tag(tag)) for word, tag in sent]

class HMMTagger(TaggerI):
    def __init__(self, vocab_size=1000, clean_tags=False, vocab=set(), tags=[], oov = "<OOV>"):
        self.tag_word_counts= ConditionalFreqDist()
        self.tag_tag_counts = ConditionalFreqDist()

        self.tag_word_probs = None 
        self.tag_tag_probs = None 
        
        self.clean_tags = clean_tags
        self.vocab_size = vocab_size
        
        self.oov = oov
        self.vocab = vocab
        self.tags = tags

    def train(self, sents):
        def replaceOOV(word):
            """Given a word, returns the word if it is in the vocabulary, oov otherwise."""
            if word in self.vocab: return word
            else: return self.oov
        
        # build the vocabulary from sents
        if not(self.vocab):
            self.vocab = set([x[0] for x in FreqDist(w for w,t in sum(sents,[])).most_common(self.vocab_size)])
        self.vocab.update([self.oov,])
        self.vocab = list(self.vocab)
        
        # build self.tags from sents
        if not(self.tags):
            self.tags = list(set(t for w, t in sum(sents,[])))
            
        # update self.tag_tag_counts and self.tag_word_counts to have <tag, tags> and <tag, vocab> key-value set, respectively
        for tag in self.tags:
            self.tag_tag_counts[tag].update(self.tags)
            self.tag_word_counts[tag].update(self.vocab)              

        # iterate through the tuples in sents and update self.tag_word_counts and self.tag_tag_counts
        # such that self.tag_word_counts counts the number of times a word in the vocabulary appears given the tag,
        # and self.tag_tag_counts counts the number of a times a tag appears following a given tag
        for sent in sents:
            replaced_sent = [(replaceOOV(w),t) for w,t in sent]
            prev_tag = "<START>"
            for word, tag in replaced_sent:
                self.tag_word_counts[tag].update((word,))
                self.tag_tag_counts[prev_tag].update((tag,))
                prev_tag = tag
        
        # self.tag_word_probs represents the conditional probability of a word given its tag
        # and self.tag_tag_probs represents the conditional probability of a tag given its previous tag
        # self.tag_word_costs and self.tag_tag_costs calculate the cost (represented as log probability)
        # of choosing a word given a tag and choosing a tag given its previous tag, respecitvely
        self.tag_word_probs = ConditionalProbDist(self.tag_word_counts, MLEProbDist)
        self.tag_word_costs = self.cpdToArray(self.tag_word_probs, self.tags, self.vocab)
        self.tag_tag_probs = ConditionalProbDist(self.tag_tag_counts, MLEProbDist)
        self.tag_tag_costs = self.cpdToArray(self.tag_tag_probs, self.tags, self.tags)
        
    def cpdToArray(self, cpd, key_list, value_list):
        return array([[cpd[key].logprob(value) for value in value_list] for key in key_list])
    
    def token_to_index(self, token):
        return self.vocab.index(token)
    
    def tag_to_index(self, tag):
        return self.tags.index(tag)
        
    def tag(self, tokens):
        """Given a list of tokens as input, determines the most appropriate tag sequence for the given token sequence,
        and returns a corresponding list of tagged tokens of the form (token, tag)."""
        numRows = len(self.tags)
        numCols = len(tokens)
        # initialize tables for dynamic programming
        table = array([[0] * numCols] * numRows, dtype=float32)
        trace = array([[None] * numCols] * numRows)
        
        # fill in the base cases, i.e. the first column
        for row in range(numRows):
            currentTag = self.tags[row]
            currentWord = tokens[0] if tokens[0] in self.vocab else '<OOV>'
            table[row][0] = self.tag_tag_probs['<START>'].prob(currentTag) * self.tag_word_probs[currentTag].prob(currentWord)
            trace[row][0] = '<START>'
        
        # fill the rest of the table
        # iterate through by columns
        for col in range(1, numCols):
            for row in range(numRows):
                currentTag = self.tags[row]
                currentWord = tokens[col] if tokens[col] in self.vocab else '<OOV>'
                maxProbability = 0.0;
                maxPrevRow = 0
                
                # iterate through the previous column and find the maximum probability
                # as well as the previous tag that led to the maximum probability
                for prevRow in range(numRows):
                    prevTag = self.tags[prevRow]
                    probability = table[prevRow][col-1] * self.tag_tag_probs[prevTag].prob(currentTag)                                     * self.tag_word_probs[currentTag].prob(currentWord)
                    if probability > maxProbability:
                        maxProbability = probability
                        maxPrevRow = prevRow
                        
                table[row][col] = maxProbability
                trace[row][col] = maxPrevRow
        
        returnList = []
        # retrace and construct the tag list
        maxIndex = argmax(table, axis=0)[-1]
        # insert the last (token, tag) pair
        returnList.insert(0, (tokens[-1], self.tags[maxIndex]))
        # loop through the trace table and prepend each (token, tag) pair
        i = numCols - 1
        index = trace[maxIndex][numCols-1]
        while i > 0:
            returnList.insert(0, (tokens[i-1], self.tags[index]))
            i -= 1
            index = trace[index][i]
        
        return returnList

if __name__ == '__main__':
    all_sents = list(simple_brown_generator())

    NUM_TRAIN = 57000
    vocab_size = 1000
    vocab = set([x[0] for x in FreqDist([w for sent in all_sents for w,t in sent]).most_common(vocab_size)])
    tags = list(set([t for sent in all_sents for w, t in sent]))

    decoder=HMMTagger(vocab=vocab, tags=tags)
    decoder.train(all_sents[:NUM_TRAIN])

    # print(decoder.tag(['What', 'is', 'going', 'on', '.']))
    # print(decoder.tag(['Andrew', 'is', 'running', 'through', 'the', 'bush']))
    # print(decoder.tag(['Squirrels', 'are', 'fast']))
    print(decoder.evaluate(all_sents[NUM_TRAIN:])) # returns a score of 0.6813092161929372