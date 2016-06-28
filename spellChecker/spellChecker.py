"""
Spell Checker

This module implements a spell checker using the editDistanceFinder as the error (channel) model
and AddAlphaBigramModel as the language model. It checks for real word errors as well as non word errors,
by calculating the probability of each candidate in check_words as 0.2 * cm_score + 0.4 * unigram_score + 0.4 * bigram_score.
This was a decision based on the fact that, in real word errors, information about the word itself
(i.e. the context of the word such as the words that follow or come before it, and the unigram possibility of the word)
is more important than the edit distance of the original word and a candidate.

Example
    from LanguageModel import AddAlphaBigramModel
    from EditDistance import EditDistanceFinder

    lm = AddAlphaBigramModel(alpha=.1)
    lm.train()

    cm = EditDistanceFinder()
    cm.train("wikipedia_misspellings.txt")

    s=SpellChecker(cm, lm)

    print(s.check_sentence("they did not yb any menas".split()))

    >>> [['they'], ['did'], ['not'], ['by', 'b', 'ye', 'y', 'yo', 'ob', 'ya', 'ab'], ['any'], 
    >>>  ['means', 'mens', 'mena', 'zenas', 'menan', 'mends']]

    print(s.autocorrect_sentence("they did not yb any menas".split()))
    >>> ['they', 'did', 'not', 'by', 'any', 'means']

    print(s.suggest_sentence("they did not yb any menas".split(), max_suggestions=2))
    >>> ['they', 'did', 'not', ['by', 'b'], 'any', ['means', 'mens']]
"""

import string

from collections import OrderedDict
from LanguageModel import AddAlphaBigramModel
from EditDistance import EditDistanceFinder

class SpellChecker():
    def __init__(self, channel_model, language_model, max_distance):
        """Given an EditDistanceFinder, a AddAlphaBigramModel, and an int as input, initializes a SpellChecker."""
        self.channel_model = channel_model # EditDistanceFinder
        self.language_model = language_model # AddAlphaBigramModel
        self.max_distance = max_distance
    
    def bigram_score(self, prev_word, word, next_word):
        """Given three words as input (a "previous" word, a "focus" word, and a "next" word),
        returns the average of the bigram score of the bigrams (prev_word, word) and (word, next_word) according to the AddAlphaBigramModel."""
        return (self.language_model.bigram_prob(prev_word, word) + self.language_model.bigram_prob(word, next_word))/2
        
    def unigram_score(self, word):
        """Given a word as input, returns the unigram probability of the word according to the AddAlphaBigramModel."""
        return self.language_model.unigram_prob(word)
        
    def cm_score(self, error_word, corrected_word):
        """Given an error word and a possible correction as input, returns the EditDistanceFinder's probability of the corrected word
        having been transformed into the error word."""
        # Be careful about the order of the arguments to you call to the EditDistanceFinder, in light of how the noisy channel model works!
        return self.channel_model.prob(corrected_word, error_word) # check correctness
    
    def inserts(self, word, filtered = True):
        """Given a word as input, returns a list of words (that are in the AddAlphaBigramModel) that are within one insert of word."""
        within_one = []
        for i in range(len(word)+1):
            for char in string.ascii_lowercase:
                new_word = word[:i] + char + word[i:]
                if filtered and new_word in self.language_model:
                        within_one.append(new_word)
                elif not filtered:
                    within_one.append(new_word)
        return within_one
        
    def deletes(self, word, filtered = True):
        """Given a word as input, returns a list of words (that are in the AddAlphaBigramModel) that are within one deletion of word."""
        within_one = []
        for i in range(len(word)):
            new_word = word[:i] + word[i+1:]
            if filtered and new_word in self.language_model:
                within_one.append(new_word)
            elif not filtered:
                within_one.append(new_word)
        return within_one
        
    def substitutions(self, word, filtered = True):
        """Given a word as input, returns a list of words (that are in the AddAlphaBigramModel) that are within one substitution of word."""
        within_one = []        
        for i in range(len(word)):
            for char in string.ascii_lowercase.replace(word[i], ''): # loop through the alphabet except the character we are substituting
                new_word = word[:i] + char + word[i+1:]
                if filtered and new_word in self.language_model:
                    within_one.append(new_word)
                elif not filtered:
                    within_one.append(new_word)
        return within_one
        
    def transpositions(self, word, filtered = True):
        """Given a word as input, return a list of words (that are in the AddAlphaBigramModel) that are within one transposition of word."""
        # transposition of the same characters?
        within_one = []
        for i in range(len(word)-1):
            new_word = word[:i] + word[i+1] + word[i] + word[i+2:]
            if filtered and new_word in self.language_model:
                within_one.append(new_word)
            elif not filtered:
                within_one.append(new_word)
        return within_one
    
    def generate_candidates(self, word):
        """Given a word as input, return a list of candidate words that are within self.max_distance edits of word by calling inserts, deletes, substitutions, and transpositions."""
        valid_candidates = []
        for word in self.__generate_candidates_recursive__(word, 0):
            if word in self.language_model:
                valid_candidates.append(word)
        return valid_candidates
        
    def __generate_candidates_recursive__(self, word, i):
        if i == self.max_distance:
            return []
        else:
            so_far = []
            return_list = []
            so_far += self.inserts(word, filtered = False)
            so_far += self.deletes(word, filtered = False)
            so_far += self.substitutions(word, filtered = False)
            so_far += self.transpositions(word, filtered = False)
            for w in so_far:
                return_list += self.__generate_candidates_recursive__(w, i+1)
            return so_far + return_list
        
    def check_non_words(self, sentence):
        """Given a list of words as input, returns a list of lists. Words that are in the language model should map to a list with just that word,
        while words that are not in the language model map to a list of possible corrections."""
        return_list = []
        for word in sentence:
            if word in self.language_model:
                return_list.append([word])
            else:
                return_list.append(self.generate_candidates(word))
        return return_list
   
    def check_words(self, sentence):
        """Given a list of valid words as input, generates suggested corrections for real word spelling errors."""
        """check_sentence should call check_words after check_non_words,
        so functions like autocorrect_sentence and suggest_sentence should work off of the combination of the two."""
        candidates_list = []
        max_score = float("-infinity")
        most_probable_sentence = []
        # build candidates_list which is a list of lists where each lists is composed of candidates for each word in the sentence
        for word in sentence:
            # if real word
            if word in self.language_model:
                candidates = self.generate_candidates(word)
                if word not in candidates:
                    candidates.append(word)
                candidates_list.append(candidates)
            # if non-word
            else:
                candidates_list.append([word]) # just a list with the non-word
            
        for i in range(len(sentence)):
            for j in range(len(candidates_list[i])):
                current_word = candidates_list[i][j]
                # find out the bigram score of this word
                if i == len(sentence) - 1 and i == 0: # one word sentence
                    bigram_score = self.bigram_score("<s>", current_word, "<s>")
                elif i == len(sentence) - 1: # end of sentence
                    bigram_score = self.bigram_score(sentence[i - 1], current_word, "<s>")
                elif i == 0: # start of sentence
                    bigram_score = self.bigram_score("<s>", current_word, sentence[i + 1])
                else:
                    bigram_score = self.bigram_score(sentence[i - 1], current_word, sentence[i + 1])
                # calculate the overall score
                score = 0.2 * self.cm_score(sentence[i], current_word) + 0.4 * self.unigram_score(current_word) + 0.4 * bigram_score
                if score > max_score:
                    max_score = score
                    most_probable_sentence = sentence[:i] + [current_word] + sentence[i+1:]
        return most_probable_sentence
        
    def check_sentence(self, sentence):
        """Given a tokenized sentence (as a list of words) as input, calls check_non_words then check_words, and returns the resulting list-of-lists.
        """
        non_word_list = self.check_non_words(sentence)
        real_word_list = self.check_words(sentence)
        merged_list = []
        for i in range(len(sentence)):
            merged = []
            if type(non_word_list[i]) is str and type(real_word_list[i]) is str:
                merged = [non_word_list[i], real_word_list[i]]
            elif type(non_word_list[i]) is str:
                merged = [non_word_list[i]] + real_word_list[i]
            elif type(real_word_list[i]) is str:
                merged = [real_word_list[i]] + non_word_list[i]
            else:
                merged = non_word_list[i] + real_word_list[i]
            merged_unique = list(set(merged))
            if len(merged_unique) == 1:
                merged_list.append(merged_unique[0])
            else:
                merged_list.append(merged_unique)
        return merged_list
    
    def autocorrect_sentence(self, sentence):
        """Given a tokenized sentence (as a list of words) as input, calls check_sentence on the sentence, and returns a new list of tokens
        where each non-word has been replaced by its most likely spelling correction."""
        list_of_words, probabilities_list = self.__score_probabilities__(sentence) # check_sentence is called inside the helper function
        return_list = []
        for i in range(len(list_of_words)):
            if len(list_of_words[i]) == 1: # no candidate suggestion
                return_list += list_of_words[i]
            else:
                max_probability_index = probabilities_list[i].index(max(probabilities_list[i])) # find the index at which the most probable candidate is
                return_list.append(list_of_words[i][max_probability_index])
        return return_list
                    
    def suggest_sentence(self, sentence, max_suggestions):
        """Given a tokenized sentence (as a list of words) as input, call check_sentence on the sentence, and return a new list where:
        - Real words are just strings in the list
        - Non-words are lists of up to max_suggestions suggested spellings, ordered by the model's preference for them."""
        list_of_words, probabilities_list = self.__score_probabilities__(sentence)
        return_list = []
        for i in range(len(list_of_words)):
            if len(list_of_words[i]) == 1: # no candidate suggestion
                return_list.append(list_of_words[i])
            else:
                ordered_candidates = [x for (y, x) in sorted(zip(probabilities_list[i], list_of_words[i]), reverse=True)]
                return_list.append(ordered_candidates[:max_suggestions])
        return return_list
    
    def __score_probabilities__(self, sentence):
        """Given a tokenized sentence (as a list of words) as input, calls check_sentence on the sentence, and returns a new list of lists
        where each list consists of the probability of each candidate in the list according to the models' preference.
        The probability of a word that does not have candidates is 0.0."""
        list_of_words = self.check_sentence(sentence)
        return_list = []
        current_index = 0 # the index of the word we are currently considering
        for words in list_of_words:
            # if no edit suggestions
            if len(words) == 1:
                return_list += [0.0]
            # otherwise
            else:
                error_word = sentence[current_index] # the original word
                scores = []
                # iterate over the candidates
                i = 0
                while i < len(words):
                    if current_index == len(sentence) - 1 and current_index == 0: # one word sentence
                        bigram_score = self.bigram_score("<s>", words[i], "<s>")
                    elif current_index == len(sentence) - 1: # end of sentence
                        bigram_score = self.bigram_score(sentence[current_index - 1], words[i], "<s>")
                    elif current_index == 0: # start of sentence
                        bigram_score = self.bigram_score("<s>", words[i], sentence[current_index + 1])
                    else:
                        bigram_score = self.bigram_score(sentence[current_index - 1], words[i], sentence[current_index + 1])
                    scores.append(self.cm_score(error_word, words[i]) + 0.5 * bigram_score + 0.5 * self.unigram_score(words[i]))
                    i += 1
                return_list.append(scores)
            current_index += 1
        return list_of_words, return_list

if __name__ == "__main__":
    lm = AddAlphaBigramModel(alpha=.1)
    lm.train()

    cm = EditDistanceFinder()
    cm.train("wikipedia_misspellings.txt")

    s = SpellChecker(cm, lm, 1)
    print (s.check_sentence(["how", "are", "yoo", "sir"]))
    print ()
    print (s.check_sentence(["they", "did", "not", "yb", "any", "menas"]))
    print ()
    print (s.autocorrect_sentence(["they", "did", "not", "yb", "any", "menas"]))
    print ()
    print (s.autocorrect_sentence(["menas"]))
    print ()
    print (s.__score_probabilities__(["they", "did", "not", "yb", "any", "menas"]))
    print ()
    print (s.suggest_sentence(["they", "did", "not", "yb", "any", "menas"], 3))
    print ()
    print (s.generate_candidates("hi"))
    print ()
    print (s.check_words(["how", "are", "yo", "sir"]))
    print ()
    print (s.check_words(["man", "and", "woman", "are"]))
    print ()
    print(s.suggest_sentence("they did not yb any menas".split(), max_suggestions=2))