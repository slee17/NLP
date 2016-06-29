"""
This module handles binary classification of arbitrary texts using a Naive Bayes classifier.

Sample Usage
    See the main function.

Design Decisions and Analysis
    Pos_Class matters: when pos_class = "DEM", the f1 score is 0.886, whereas the f1 score
    when pos_class = "REP" is 0.848. This makes sense, since a f1 score depends on values of
    precision and recall, and the values of precision and recall depend on the pos_class.
    
    Building the vocabulary: this module builds the vocabulary from the training data (a set of documents).
    After extracting all words from the training data, the module 1) removes words that appear less than
    four times in each document, 2) removes stopwords, and 3) stems. The size of the resulting vocabulary was 6,603.

    Choosing the alpha value: an alpha value of 1 gives an f1 score of 0.823, whereas an alpha value of
    0.4 gives an f1 score 0.873. 0.4 was chosen after experimenting with several different values
    (though this value can be changed by user input to the constructor).

    Size of tokens as an additional feature: this feature came out on top as the most distinguishing
    feature with the coefficient of -1.1010.
    
    Frequent bigrams as an additional feature: This module distinguishes bigrams that appear more than once in
    each of the document and cumulated those. This resulted in [('loan', 'student'), ('secur', 'work'),
    ('interest', 'friend'), ('rais', 'tax'), ('senat', 'servic'), ('think', 'choic'), ('nation', 'question'),
    ('famili', 'rate')]. The f1 score wasn't affected by much by this feature, supposedly because these bigrams
    are used frequently by Republicans and Democrats alike.
"""

import numpy
import glob
import nltk
import sys

from pathlib import Path
from pandas import DataFrame

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, make_scorer, confusion_matrix

from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from nltk.collocations import *

class TextClassifier():
    def __init__(self, path_object, alpha_value = 0.4):
        """Sets the starting state of the classifier."""
        self.clf = MultinomialNB(alpha = alpha_value)
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        # given a path, creates vocabulary and bigrams based on the files in the path and its subdirectories
        p = sorted(path_object.glob('*/*'))
        tokens_cumulated = []
        bigrams_cumulated = []
        for path in p:
            if path.is_file():
                tokens = self.get_tokens(path)
                tokens_cumulated += tokens
                bigrams_cumulated += self.get_bigrams(tokens, n = 2)
        self.vocabulary = list(set(tokens_cumulated)) # hacky way of getting the distinct words but in such a way that they are accessible by index
        self.bigrams = list(set(bigrams_cumulated))
    
    def get_tokens(self, path_object):
        """Given a POSIX path object, handles opening a file, and doing any tokenizing/normalization.
        Returns a list of tokens."""
        tokens = []
        tokenizer = nltk.RegexpTokenizer(r'\w+') # r'\w+' this tokenizer removes punctuations from the text
        stemmer = nltk.stem.PorterStemmer()
        stopword = stopwords.words('english')
        with path_object.open() as f:
            text = tokenizer.tokenize(BeautifulSoup(f.read(), 'html.parser').get_text())
            # keep only the tokens that appear more than twice in this document
            processed = [k for (k,v) in Counter(text).items() if v > 3]
            # keep only the tokens that are lower case words and are not stopwords (according to nltk.stopwords)
            tokenized = [stemmer.stem(x) for x in processed if x.isalpha() and x.islower() and x not in stopword]
            tokens += tokenized
        f.close()
        return tokens
    
    def get_bigrams(self, tokens, n = 1):
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(n) # only the bigrams that appear >=n times
        bigram = finder.nbest(self.bigram_measures.pmi, sys.maxsize) # get them all (as opposed to the top n bigrams)
        return bigram
    
    def extract_features(self, path_object):
        """Given a POSIX path object, returns a numpy array of features for the document contained in the corresponding file."""
        tokens = self.get_tokens(path_object)
        features = []
        features += self.extract_word_counts(tokens, path_object)
        features += self.extract_token_length(tokens)
        # features += self.extract_vocabulary_size(tokens)
        features += self.extract_bigrams(tokens)
        return numpy.asarray(features)
    
    def extract_word_counts(self, tokens, path_object):
        """Returns a list of lists."""
        cnt = Counter(tokens)
        features = [0] * len(self.vocabulary)
        for item in cnt.items():
            features[self.vocabulary.index(item[0])] = item[1]
        return features
    
    def extract_token_length(self, tokens):
        return [len(tokens)]
    
    def extract_vocabulary_size(self, tokens):
        return [len(set(tokens))]
    
    def extract_bigrams(self, tokens, n_bigrams = 2):
        bigram = self.get_bigrams(tokens, n = n_bigrams)
        cnt = Counter(bigram)
        features = [0] * len(self.bigrams)
        for item in cnt.items():
            if item in self.bigrams:
                features[self.bigrams.index(item[0])] = item[1]
        return features
    
    def get_feature_names(self):
        """Returns a list of feature names (in the same order that the features are returned by extract_features),
        for use in analyzing the system."""
        features = self.vocabulary
        features.append("Size of Tokens")
        # features.append("Size of Vocabulary")
        features += [str(x) for x in self.bigrams]
        return features
    
    def do_cross_validation(self, filepaths, labels, k = 10, pos_classes = ['DEM', 'REP']):
        """Given a list of POSIX file paths and a list of labels, returns a 10-fold cross-validation score
        (the average f1-score)."""

        # sanity check
        assert len(filepaths) == len(labels)
        
        # construct text and label
        text = []
        label = []
        for i in range(len(filepaths)):
            text.append(self.extract_features(filepaths[i]))
            label.append(labels[i])
        text = numpy.asarray(text)
        label = numpy.asarray(label)
        
        # calculate scores
        avg_scores = []
        
        for pos_class in pos_classes:
            # create a new scorer
            f1_scorer = make_scorer(f1_score, pos_label = pos_class)

            # compute the average f1 score
            kfold = cross_validation.KFold(len(filepaths), n_folds = k, shuffle = True)

            scores = cross_validation.cross_val_score(self.clf, text, label, cv = kfold, n_jobs = 1, scoring = f1_scorer)
            avg_scores.append(sum(scores)/len(scores))
        
        # return the average of scores from different pos_class
        return sum(avg_scores)/len(avg_scores)
    
    def do_feature_comparison(self, p = Path('2008_election'), n = 20):
        """Trains the Naive Bayes classifier on all of the available documents
        and returns the 20 most heavily-weighted features."""

        # build training data
        paths = sorted(p.glob('*/*'))
        text = []
        label = []
        for path in paths:
            text.append(self.extract_features(path))
            label.append((str(path).split('/')[1][:3])) # this program assumes that the files are under p/class/filename
                                                        # where the first three leters of class is used as the identifier
        
        clf_trained = self.clf.fit(text, label)
        
        feature_names = self.get_feature_names()
        coefs = sorted(zip(self.clf.coef_[0], feature_names), reverse = True)
        
        return coefs[:20]


if __name__=="__main__":

    # For interaction with the program
    tc = TextClassifier(Path('2008_election'))

    # TESTS FOR GET_TOKENS AND GET_FEATURE_NAMES
    # tc.get_tokens(Path('2008_election/Dem_01/73120'))
    # tc.get_feature_names(Path('2008_election/Dem_01/73120'))

    paths = []
    labels = []

    # SET UP FOR RUNNING THE PROGRAM (BUILD PATHS AND LABELS)
    p = Path('2008_election')
    paths += sorted(p.glob('DEM*/*'))

    for i in range(len(paths)):
        labels.append('DEM')

    republicans = sorted(p.glob('REP*/*'))
    paths += republicans
    for i in range(len(republicans)):
        labels.append('REP')

    # ALTERNATE PATH TO USE FOR TESTING
    #for path in paths:
    #    a = 
    #paths = [Path('2008_election/DEM_01/73120'),
    #        Path('2008_election/DEM_01/76232'),
    #        Path('2008_election/DEM_01/76302'),
    #        Path('2008_election/DEM_01/76361'),
    #        Path('2008_election/DEM_01/76456'),
    #        Path('2008_election/REP_02/62273'),
    #        Path('2008_election/REP_02/77103'),
    #        Path('2008_election/REP_02/77104'),
    #        Path('2008_election/REP_02/77105'),
    #        Path('2008_election/REP_02/77106'),
    #        Path('2008_election/REP_02/77107')]
    #labels = ['Democrats', 'Democrats', 'Democrats', 'Democrats', 'Democrats',
    #          'Republican', 'Republican', 'Republican', 'Republican', 'Republican', 'Republican']"""

    # TESTS FOR EXTRACT_FEATURES, GET_TOKENS, AND EXTRACT_BIGRAMS, DO_CROSS_VALIDATION, AND DO_FEATURE_COMPARISON
    #print(tc.extract_features(Path('2008_election/DEM_01/73120')))
    #tokens = tc.get_tokens(Path('2008_election/DEM_01/73120'))
    #print(tc.extract_bigrams(tokens))
    #tc.get_bigrams(['hello', 'world', 'foo', 'bar', 'hi', 'hello', 'world'])
    print(tc.do_cross_validation(paths, labels))
    most_predictive = tc.do_feature_comparison()
    for (coef, feature) in most_predictive:
        print ("\t%.4f\t%-15s" % (coef, feature))