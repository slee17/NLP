# A script to get content from Wikipedia, extract text, and do simple analysis

import requests
import bs4
import nltk
import re

from nltk.stem.porter import PorterStemmer
from collections import defaultdict

# Some helper functions

def get_words(s):
    s = s.lower()
    listWords = s.split(' ')
    return listWords

def count_words(words):
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1
    return frequency

from operator import itemgetter

def words_by_frequency(words):
    """Given a list of words, returns a list of `(word, count)` tuples sorted by count such that
    the first item in the list is the most frequent item."""
    frequency = count_words(words)
    tupleList = list(frequency.items())
    tupleList.sort(key=itemgetter(1), reverse=True)
    return tupleList


# Acquire a Wikipedia File using the requests module (http://requests.readthedocs.org/en/latest/user/quickstart/)
# and the Wikipedia API (https://en.wikipedia.org/w/api.php) to download a copy of the Wikipedia article 

def get_wikipedia_page(article_title, language):
    """Takes as input the name of a Wikipedia page and returns the HTML extract of that Wikipedia page."""
    parameters = {'format': 'json', 'action': 'query', 'prop': 'extracts', 'exintro': 'explaintext', 'titles': article_title, 'lang': language}
    # build a response object
    r = requests.get('https://'+language+'.wikipedia.org/w/api.php', params = parameters)
    return (list(r.json()['query']['pages'].values())[0]['extract'])

def text_from_html(html_str):
    """Takes an html string as input and returns the text content of that html, using BeautifulSoup."""
    return bs4.BeautifulSoup(html_str, "lxml").get_text() # use the "lxml" parser

def extract_wikipedia_page(article_title, language):
    """Takes the title of a Wikipedia article and a language as input, and returns the article's extract as plain text."""
    return (text_from_html(get_wikipedia_page(article_title, language)))

def compare_english_simple(article_title):
    """Given a title of an article, returns the number of tokens, types, and stems
    in both the English version and the simple English version."""
    english = extract_wikipedia_page(article_title, "en")
    simple = extract_wikipedia_page(article_title, "simple")
    num_tokens_english = len(english)
    num_tokens_simple = len(simple)
    types_english = count_words(get_words(english))
    types_simple = count_words(get_words(simple))
    
    porter_stemmer = PorterStemmer()
    
    stem_english = defaultdict(int)
    stem_simple = defaultdict(int)
    for key in types_english.keys():
        stem_english[porter_stemmer.stem(key)] += 1
    for key in types_simple.keys():
        stem_simple[porter_stemmer.stem(key)] += 1
    
    print ("Number of Tokens in English " + article_title + ": %d" % num_tokens_english)
    print ("Number of Tokens in Simple English " + article_title + ": %d" % num_tokens_simple)
    print ("Number of Types in English " + article_title + ": %d" % len(types_english))
    print ("Number of Types in Simple English " + article_title + ": %d" % len(types_simple))
    print ("Number of Stems in English " + article_title + ": %d" % len(stem_english))
    print ("Number of Stems in Simple English " + article_title + ": %d" % len(stem_simple))

if __name__ == "__main__":
    extract_wikipedia_page("Stephen Curry", language="en")
    compare_english_simple("Water")