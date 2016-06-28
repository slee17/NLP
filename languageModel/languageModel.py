# Bag-of-words, language models, Witten-Bell Discounting, Kneser-Ney Discounting

import nltk

from collections import Counter

from nltk.corpus import brown
from nltk.util import bigrams, trigrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


def brown_tokens(category):
    """Given a category of the brown corpus, returns all the tokens in the corpus such that
    all tokens are lowercased and tokens that don't include any alpha-numeric characters are removed."""
    words = [word.lower() for word in brown.words(categories=category) if word.isalnum()]
    return words

def count(category):
    """Given a category of the brown corpus, returns a counter that counts the number of times
    each word in the corpus appears."""
    return Counter(brown_tokens(category))

def test_appear_token(category1, category2):
    """Given two categories category1 and category2, returns the percentage of tokens that appear in category1
    but don't appear in category2."""
    num_false = 0
    category1_words = brown_tokens(category1)
    category2_counts = count(category2)
    
    for word in category1_words:
        if word not in category2_counts:
            num_false += 1
            
    return num_false/len(category1_words)

def test_appear_type(category1, category2):
    """Given two categories category1 and category2, returns the percentage of types that appear in category1
    but don't appear in category2."""
    num_false = 0
    category1_types = set(brown_tokens(category1))
    category2_counts = count(category2)
    
    for word in category1_types:
        if word not in category2_counts:
            num_false += 1
            
    return num_false/len(category1_types)

def brown_bigrams(category):
    """Takes as input the name of a brown category, and returns a list of all of the bigrams in the category."""
    words = ["<s>"]
    words += [word.lower() for word in brown.words(categories=category) if word.isalnum()]
    words.append("</s>")
    return list(bigrams(words))

def brown_trigrams(category):
    """Takes as input the name of a brown category, and returns a list of all of the trigrams in the category."""
    words = ["<s>"]
    words += [word.lower() for word in brown.words(categories=category) if word.isalnum()]
    words.append("</s>")
    return list(trigrams(words))

def test_appear_ngrams(test_category, training_categories, n):
    """Given the test category (test_category), an array of training categories (training_categories), and n,
    returns the percentage of n-grams(tokens, not types) that appear in category 1 but don't appear in category 2."""
    training_tokens = []
    if n == 2:
        for category in training_categories:
            training_tokens += (brown_bigrams(category))
        test_tokens = brown_bigrams(test_category)
    elif n == 3:
        for category in training_categories:
            training_tokens += (brown_trigrams(category))
        test_tokens = brown_trigrams(test_category)
    else:
        return
        
    num_false = 0
    training_counter = Counter(training_tokens)
    
    for token in test_tokens:
        if token not in training_counter:
            num_false += 1
            
    return num_false/len(test_tokens)

def test_appear_chunks(categories, chunk_size, n):
    """Given an array of categories, a chunk size, and n,
    returns the percentage of n-grams(tokens, not types) that appear in the first n-1 chunks of all categories
    but don't appear in the nth chunk."""
    
    # training_categories = [fist_category_bigrams, second_category_bigrams, ...]
    training_tokens = []
    test_tokens = []
    
    if n == 2:
        for category in categories:
            tokens = brown_bigrams(category)
            training_tokens += tokens[:int((chunk_size-1)*len(tokens)/chunk_size)]
            test_tokens += tokens[int((chunk_size-1)*len(tokens)/chunk_size):]
            
    elif n == 3:
        for category in categories:
            tokens = brown_trigrams(category)
            training_tokens += tokens[:int((chunk_size-1)*len(tokens)/chunk_size)]
            test_tokens += tokens[int((chunk_size-1)*len(tokens)/chunk_size):]
    else:
        return
        
    num_false = 0
    training_counter = Counter(training_tokens)
    
    for token in test_tokens:
        if token not in training_counter:
            num_false += 1
            
    return num_false/len(test_tokens)
      
def bigram_to_cfd(category):
    """Given a category in the brown corpus, returns a conditional frequency distribution
    where the context is the first word in each bigram and the element is the second word in each bigram."""
    bigrams = brown_bigrams(category)
    return ConditionalFreqDist((bigram[0], bigram[1]) for bigram in bigrams)

def words_by_followers(category):
    """Given a category from the brown corpus, lowercases everything, builds a conditional frequency distribution,
    and then returns another frequency distribution where the keys are the conditions and the counts
    are the number of different words that can follow each context."""
    cfdist = bigram_to_cfd(category)
    fdist = FreqDist()
    for context in cfdist.keys():
        fdist[context] = len(cfdist[context])
    return fdist

def words_by_followers(category):
    """Given a category from the brown corpus, lowercases everything,
    and returns a frequency distribution where the keys are words
    and the counts are the number of different contexts that each word can appear in."""
    bigrams = brown_bigrams(category)
    cfdist = ConditionalFreqDist((bigram[1], bigram[0]) for bigram in bigrams)
    fdist = FreqDist()
    for context in cfdist.keys():
        fdist[context] = len(cfdist[context])
    return fdist


if __name__ == "__main__":
    ## Token comparison
    print (test_appear_token("adventure", "editorial"))
    print (test_appear_token("romance", "editorial"))
    print (test_appear_type("adventure", "editorial"))
    print (test_appear_type("romance", "editorial"))
    
    # 15.72% of the tokens that appear in the adventure category do not appear in the editorial category.
    # 13.29% of the tokens that appear in the romance category don't appear in the editorial category.
    # 58.21% of the types that appear in the adventure category do not appear in the editorial category.
    # 54.53% of the types that appear in the romance category don't appear in the editorial category.
    #
    # The big difference between the percentage of tokens and the percentage of types is surprising:
    # the latter is almost 4 times the former.
    # This difference shows that, while the editorial category uses a considerable number of
    # (types of) words that do not occur in either "adventure" or "romance", the frequency of those words are very low;
    # whereas words that occur in both "adventure" and "editorial" or both "romance" and "editorial" occur very often.

    print ()

    ## Bigram analysis
    print (test_appear_ngrams("adventure", ["editorial"], 2))
    print (test_appear_ngrams("romance", ["editorial"], 2))
    print (test_appear_ngrams("adventure", ["editorial"], 3))
    print (test_appear_ngrams("romance", ["editorial"], 3))
    # 68.27% of the bigrams that appear in the adventure category do not appear in the editorial category.
    # 66.25% of the bigrams that appear in the romance category do not appear in the editorial category.
    # 96.17% of the trigrams that appear in the adventure category do not appear in the editorial category.
    # 95.72% of the trigrams that appear in the romance category do not appear in the editorial category.
    #
    # The percentage of bigrams that appear in one category but not in another is, as expected,
    # much higher than the percentage of words that appear in one category but not in another;
    # similarly, the percentage of trigrams that apepar in one category but not in another is much higher
    # than the percentage of bigrams. This is because the longer the phrase, the less likely it is
    # for the phrase to appear in a certain text.

    print ()
    
    ## Train on pairs of categories and test on the third category
    print (test_appear_ngrams("romance", ["editorial", "adventure"], 2))
    print (test_appear_ngrams("adventure", ["editorial", "romance"], 2))
    print (test_appear_ngrams("editorial", ["adventure", "romance"], 2))
    print ()
    print (test_appear_ngrams("romance", ["editorial", "adventure"], 3))
    print (test_appear_ngrams("adventure", ["editorial", "romance"], 3))
    print (test_appear_ngrams("editorial", ["adventure", "romance"], 3))
    # 52.50% of the bigrams that appear in the romance category do not appear in the editorial and adventure categories.
    # 54.66% of the bigrams that appear in the adventure category do not appear in the editorial and romance categories.
    # 64.68% of the bigrams that appear in the editorial category do not appear in the adventure and romance categories.
    #
    # 90.08% of the trigrams that appear in the romance category do not appear in the editorial and adventure categories.
    # 90.58% of the trigrams that appear in the adventure category do not appear in the editorial and romance categories.
    # 94.09% of the trigrams that appear in the editorial category do not appear in the adventure and romance categories.
    #
    # In both bigrams and trigrams, the best combination (i.e. the combination that leads to the least percentage of
    # bigrams or trigrams that were in the test set but not in the training set) seems to be when
    # the editorial category and the adventure category are used as a training set,
    # and the romance category as the test set. This is presumably because of the wide variety of words/bigrams/trigrams
    # that appear in the editorial category: because the editorial category is used for training,
    # we are less likely to be surprised when testing on the romance category, which contains fewer types of words/bigrams/trigrams.
    # In fact, whenever the editoral category is used in training (i.e. bigrams with romance and editorial as the training set and adventure as the test set,
    # and trigrams with romance and editorial as the training set and adventure as the test set),
    # the result was lower percentage of previously unseen bigrams/trigrams than when the editorial category was used as the test set.

    print ()
    
    ## Train on equal-sized "chunks"
    print(test_appear_chunks(["editorial", "adventure", "romance"], 4, 2))
    print(test_appear_chunks(["editorial", "adventure", "romance"], 4, 3))
    # 53.51% of the bigrams that appear in the first three chunks of editorial, adventure, and romance do not appear in the last chunk.
    # 
    # 89.68% of the trigrams that appear in the first three chunks of editorial, adventure, and romance do not appear in the last chunk.
    # 
    # These resulsts are considerably lower (especially for the trigrams) compared to the results from the previous question (52.50%, 54.66%, 64.68% for bigrams, 90.08%, 90.58%, 94.09% for trigrams). This is presumably due to the fact that bigrams/trigrams are likely to appear repeatedly within the same text. In other words, our training set now does a better job of predicting the test set, since there is a higher chance that a certain bigram/trigram occurs again in a text than appears across different texts.

    print ()

    ## Witten-Bell Discounting
    # Some contexts are more likely to be followed by new words than others.
    len(bigram_to_cfd("romance")['was'])
    words_by_followers("adventure").pprint()
    words_by_followers("romance").pprint()
    words_by_followers("editorial").pprint()

    print (len(words_by_followers("adventure").hapaxes()))
    print (len(words_by_followers("romance").hapaxes()))
    print (len(words_by_followers("editorial").hapaxes()))
    print ()
    print (len(words_by_followers("adventure")))
    print (len(words_by_followers("romance")))
    print (len(words_by_followers("editorial")))

    words_by_followers("adventure").pprint()

    print ()
    
    ## Kneser-Ney Discounting
    # Some contexts are more likely to be *preceded* by new words than others.
    words_by_followers("romance").pprint()
    words_by_followers("editorial").pprint()

    print (len(words_by_followers("adventure").hapaxes()))
    print (len(words_by_followers("romance").hapaxes()))
    print (len(words_by_followers("editorial").hapaxes()))