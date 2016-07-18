from nltk.twitter import Streamer, TweetWriter, credsfromfile
from nltk.twitter.common import json2csv
from nltk.corpus import stopwords, opinion_lexicon

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn import metrics, cross_validation

import pandas as pd
import numpy as np
import csv
import json

def find_matching_tweets(num_tweets=100, fname="matching_tweets.csv", shownum=50):
    """Given the number of tweets to retrieve, queries that number of tweets with
    the keyword "Trump" and saves the tweet id and text as a csv file "fname". Prints
    out the shownum amount of tweets using panda. Does not remove retweets."""
    oauth = credsfromfile()
    # create and register a streamer
    client = Streamer(**oauth)
    writer = TweetWriter(limit=num_tweets)
    client.register(writer)
    # get the name of the newly-created json file
    input_file = writer.timestamped_file()
    client.filter(track="trump") # case-insensitive
    client.sample()

    with open(input_file) as fp:
        # these two fields for now
        json2csv(fp, fname, ['id', 'text', ])

    # pretty print using pandas
    tweets = pd.read_csv(fname, encoding="utf8")
    return tweets.head(shownum)

def parse(filename, delimiter = '\t'):
    """Given a filename of a csv file and a delimiter, returns an array of json objects."""
    json_objects = []
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter = delimiter)
        for row in reader:
            input_object = json.dumps(row)
            json_object = json.loads(input_object)
            # check the entry's label
            label = json_object['label']
            if label == 'positive':
                count_positive += 1
            elif label == 'negative':
                count_negative += 1
            else:
                count_neutral += 1
            json_objects.append(json_object)
    
    # for analyzing the data
    """print ("Total number of entries: %d" % len(json_objects))
    print ("Positive: %d, %f%%" % (count_positive, count_positive/len(json_objects)*100))
    print ("Negative: %d, %f%%" % (count_negative, count_negative/len(json_objects)*100))
    print ("Neutral: %d, %f%%" % (count_neutral, count_neutral/len(json_objects)*100))"""
    
    return json_objects

def format_json(json_objects):
    """Given an array of json objects, returns the contents of the tweets and their labels
    as separate arrays."""
    texts = []
    labels = []
    for json_object in json_objects:
        texts.append([json_object['text']])
        labels.append(json_object['label'])
    return np.asarray(texts), np.asarray(labels)

def run_baseline_systems(strategy, training_file, test_file):
    # generate training and text data
    training_json_objects = parse(training_file, delimiter = '\t')
    training_texts, training_labels = format_json(training_json_objects)
    test_json_objects = parse(test_file, delimiter = ',')
    test_texts, test_labels = format_json(test_json_objects)
    
    clf = DummyClassifier(strategy=strategy)
    
    # calculate the 10-fold f1 score
    f1_scores = cross_validation.cross_val_score(clf, texts, labels, cv=10) # scoring='f1_weighted' ??
    f1_score = sum(f1_scores)/len(f1_scores)
    
    # calculate the score on the test set
    clf.fit(training_texts, training_labels)
    test_score = clf.score(test_texts, test_labels)
    
    return f1_score, test_score

def run_NB(training_file, test_file):
    # generate training and text data
    training_json_objects = parse(training_file, delimiter = '\t')
    training_texts, training_labels = format_json(training_json_objects)
    test_json_objects = parse(test_file, delimiter = ',')
    test_texts, test_labels = format_json(test_json_objects)
    
    training_texts = [element[0] for element in training_texts]
    test_texts = [element[0] for element in test_texts]
    
    count_vectorizer = CountVectorizer(analyzer="word", stop_words='english', vocabulary=list(set(opinion_lexicon.words())))
    counts = count_vectorizer.transform(training_texts)
    
    classifier = MultinomialNB()
    
    # calculate the 10-fold f1 score
    k_fold = KFold(n=len(training_texts), n_folds=10)
    scores = cross_validation.cross_val_score(classifier, counts, training_labels, cv=k_fold) # scoring=f1_scorer
    f1_score = sum(scores)/len(scores)

    # calculate the score on the test set
    classifier.fit(counts, training_labels)
    test_counts = count_vectorizer.transform(test_texts)
    predictions = classifier.predict(test_counts)
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct_predictions += 1
    test_score = correct_predictions/len(predictions)
    
    return f1_score, test_score

def advanced_classifier(training_file, test_file):
    # generate training and text data
    training_json_objects = parse(training_file, delimiter = '\t')
    training_texts, training_labels = format_json(training_json_objects)
    test_json_objects = parse(test_file, delimiter = ',')
    test_texts, test_labels = format_json(test_json_objects)
    
    training_texts = parse_text(training_texts)
    test_texts = parse_text(test_texts)
    
    count_vectorizer = CountVectorizer(analyzer="word", stop_words='english', vocabulary=list(set(opinion_lexicon.words())))
    counts = count_vectorizer.transform(training_texts)
    
    classifier = MultinomialNB()
    
    # calculate the 10-fold f1 score
    k_fold = KFold(n=len(training_texts), n_folds=10)
    scores = cross_validation.cross_val_score(classifier, counts, training_labels, cv=k_fold) # scoring=f1_scorer
    f1_score = sum(scores)/len(scores)

    # calculate the score on the test set
    classifier.fit(counts, training_labels)
    test_counts = count_vectorizer.transform(test_texts)
    predictions = classifier.predict(test_counts)
    
    # sideline features
    for i in range(len(predictions)):
        if includes_hyperlink(test_texts[i]):
            predictions[i] = 'neutral'
        if includes_positive_hashtag(test_texts[i]):
            predictions[i] = 'positive'
    
    # calculate the score on the test set
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct_predictions += 1
    test_score = correct_predictions/len(predictions)
    
    return f1_score, test_score

def parse_text(data):
    """Given a list of lists of strings, returns a list of strings with each of the string parsed and normalized."""
    stops = set(stopwords.words('english'))
    texts = [element[0].lower().replace('.', '') for element in data if element[0] not in stops]
    parsed_tweets = []
    for text in texts:
        parsed_tweet= []
        list_of_words = text.split()
        for word in list_of_words:
            if not word.startswith('@') and word != 'rt':
                if 'http' in word:
                    parsed_tweet.append('hyperlink')
                elif word.endswith('!'):
                    parsed_tweet.append(word[:-1])
                    parsed_tweet.append('!')
                else:
                    parsed_tweet.append(word)
        parsed_tweets.append(' '.join(parsed_tweet))
                
    return parsed_tweets

def includes_hyperlink(tweet):
    """Given a tweet represented as a string, returns True if the tweet contains a hyperlink."""
    return 'hyperlink' in tweet

def includes_positive_hashtag(tweet):
    """Given a tweet represented as a string, returns True if the tweet contains a positive hashtag."""
    hashtags = ['#trump2016', '#makeamericagreatagain', '#maga', '#alwaystrump', '#onlytrump']
    for hashtag in hashtags:
        if hashtag in tweet:
            return True
    return False


if __name__ == '__main__':
    print (advanced_classifier('./data/training_full.csv', './data/matching_tweets_utf.csv'))