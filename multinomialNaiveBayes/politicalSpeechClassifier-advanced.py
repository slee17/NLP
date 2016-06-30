"""
Based on the paper ``Efficient Estimation of Word Representations in Vector Space'' by
Mikolov, Chen, Corrado, and Dean, this module improves on politicalSpeechClassifier by adding the
Continuous Bag of Words architecture as a feature to the classifier.

Sample Usage
    See the main function.

Design Decisions and Analysis
    The feature added to the classifier by the extract_most_similar function improved the f1-score
    of the system to 0.829.
"""
import numpy
import glob
import nltk
import sys

import gensim
import logging
import os
import pathlib

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from pathlib import Path
from pandas import DataFrame

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from random import shuffle
from nltk.corpus import stopwords
from nltk.collocations import *

class LabeledLineSentence(object):
    """Allows for reading in multiple text files during training."""
    def __init__(self, path_object):
        """Create a dictionary that defines the files to read and
        the label prefixes sentences from that document should take on, e.g. {'files': 'labels'}"""
        p = sorted(path_object.glob('*/*'))
        self.sources = defaultdict()
    
        for path in p:
            if path.is_file():
                self.sources[str(path)] = str(path).split('/')[1][:3]
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [source]) # prefix + '_%s' % item_no
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [source])) # prefix + '_%s' % item_no
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

class TextClassifier():
    def __init__(self, path_object, alpha_value = 0.4, train_new = True):
        """Sets the starting state of the classifier."""
        self.clf = MultinomialNB(alpha = alpha_value)
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        self.directory = path_object
        
        # given a path, creates vocabulary and bigrams based on the files in the path and its subdirectories
        p = sorted(path_object.glob('*/*'))
    
        tokens_cumulated = []
        bigrams_cumulated = []
        labels = []
        
        for path in p:
            if path.is_file():
                labels.append(str(path).split('/')[1][:3])
                tokens = self.get_tokens(path)
                tokens_cumulated += tokens
        
        # self.model = gensim.models.Word2Vec([tokens_cumulated], min_count = 1, workers = 4)

        # get unique words from the tokens and add OOV at the end
        self.vocabulary = numpy.append(numpy.unique(numpy.asarray(tokens_cumulated)), 'OOV').tolist()
        
        if train_new:
            sentences = LabeledLineSentence(path_object)
            model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
            model.build_vocab(sentences.to_array())

            for epoch in range(10):
                model.train(sentences.sentences_perm())

            model.save('./imdb.d2v')
        
        else:
            model = Doc2Vec.load('./imdb.d2v')
            
        self.model = model
        
        # for token in tokens_cumulated[1:]:
        #    numpy.add(self.model[token], vector)
        # vector = self.model[tokens_cumulated[0]]
        # self.vector = vector
        # self.bigrams = list(set(bigrams_cumulated))

    def get_tokens(self, path_object):
        """Given a POSIX path object, handles opening a file, and doing any tokenizing/normalization.
        Returns a list of tokens."""
        tokens = []
        tokenizer = nltk.RegexpTokenizer(r'\w+') # r'\w+' this tokenizer removes punctuations from the text
        stemmer = nltk.stem.PorterStemmer()
        stopword = stopwords.words('english')
        with path_object.open() as f:
            # html parsing
            text = tokenizer.tokenize(BeautifulSoup(f.read(), 'html.parser').get_text())
            # keep only the tokens that appear more than once in this document
            infrequent = [k for (k, v) in Counter(text).items() if v < 2]
            text = [x for x in text if x not in infrequent]
            # processed = [k for (k,v) in Counter(text).items() if v > 2]
            # keep only the tokens that are lower case words and that are not stopwords (according to nltk.stopwords)
            tokenized = [stemmer.stem(x).lower() for x in text if x.isalpha() and x not in stopword]
            tokens += tokenized
        f.close()
        return tokens
    
    def get_bigrams(self, tokens, n = 1):
        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(n) # only the bigrams that appear >= n times
        bigram = finder.nbest(self.bigram_measures.pmi, sys.maxsize) # get them all (as opposed to the top n bigrams)
        return bigram
    
    def extract_features(self, path_object):
        """Given a POSIX path object, returns a numpy array of features for the document contained in the corresponding file."""
        tokens = self.get_tokens(path_object)
        features = []
        features += self.extract_word_counts(tokens, path_object)
        features += self.extract_token_length(tokens)
        # features += self.extract_vocabulary_size(tokens)
        # features += self.extract_bigrams(tokens)
        # features += self.extract_vector(tokens)
        features += self.extract_most_similar(path_object)
        return numpy.asarray(features)
    
    def extract_word_counts(self, tokens, path_object):
        """Returns a list of lists."""
        cnt = Counter(tokens)
        features = [0] * len(self.vocabulary)
        for item in cnt.items():
            if item[0] in self.vocabulary:
                features[self.vocabulary.index(item[0])] = item[1]
            else:
                features[-1] += 1 # add to OOV
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
    
    def extract_vector(self, tokens):
        try:
            document_vector = self.model[tokens[0]]
            for token in tokens[1:]:
                numpy.add(self.model[token], document_vector)
        except IndexError:
            document_vector = []
        return document_vector
    
    def extract_most_similar(self, path_object):
        """Returns the class of the document most similar to the current document, according to
        their vector representations."""
        docvec = self.model.docvecs[str(path_object)]
        similars = self.model.docvecs.most_similar([docvec])
        DEM_COUNT = 0
        REP_COUNT = 0
        for similar in similars:
            if 'DEM' in similar[0]:
                DEM_COUNT += 1
            else:
                REP_COUNT += 1

        if DEM_COUNT > REP_COUNT:
            return [0] # DEM
        return [1] # REP
    
    def get_feature_names(self):
        """Returns a list of feature names (in the same order that the features are returned by extract_features),
        for use in analyzing the system."""
        features = self.vocabulary
        features.append("Size of Tokens")
        # features.append("Size of Vocabulary")
        # features += [str(x) for x in self.bigrams]
        # for i in range(100):
        #    features.append("Vector Dimension %d" % i)
        features.append("Most Similar Document")
        return features
    
    def do_cross_validation(self, filepaths, labels, k = 10, pos_classes = ['DEM', 'REP']):
        """Given a list of POSIX file paths and a list of labels, returns a 10-fold cross-validation score (the average f1-score)."""
        # sanity check
        assert len(filepaths) == len(labels)
        
        # construct text and label
        text = []
        label = []
        for i in range(len(filepaths)):
            text.append(self.extract_features(filepaths[i]))
            label.append(labels[i])
        text = numpy.asarray(text, dtype=object)
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
        
        diffs = abs(self.clf.feature_log_prob_[0:1,:] - self.clf.feature_log_prob_[1:2,:])
        weights = diffs[0,:].tolist()
        return sorted(zip(weights, feature_names), reverse = True)[:20]

if __name__=="__main__":
    tc = TextClassifier(Path('2008_election'), train_new = True)

    paths = []
    labels = []

    p = Path('2008_election')
    paths += sorted(p.glob('DEM*/*'))

    for i in range(len(paths)):
        labels.append('DEM')

    republicans = sorted(p.glob('REP*/*'))
    paths += republicans
    for i in range(len(republicans)):
        labels.append('REP')

    # print f1-score of the system and its most predictive features
    print(tc.do_cross_validation(paths, labels))
    most_predictive = tc.do_feature_comparison()
    for (coef, feature) in most_predictive:
        print ("\t%.4f\t%-15s" % (coef, feature))