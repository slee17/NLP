"""
Weighted Minimum Edit Distance

This module finds the weighted minimum edit distance between two strings, and displays the resulting character alignments. 

Sample usage:
    my_aligner = EditDistanceFinder()
    my_aligner.train("wikipedia_misspellings.txt")
    dist, alignments = my_aligner.align("caught","cought")
    print("Distance between 'caught' and 'cought' is", dist)
    my_aligner.show_alignment(alignments)
    print()
    dist, alignments = my_aligner.align("cougt","cought")
    print("Distance between 'caught' and 'cought' is", dist)
    my_aligner.show_alignment(alignments)

"""

import numpy as np
from nltk import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist

class EditDistanceFinder():
    def __init__(self):
        """Initializes the del_probs and ins_probs variables to empty MLE probability distributions,
        and the sub_probs to an empty conditional probability distribution."""
        self.del_probs = MLEProbDist(FreqDist()) # a MLE probability distribution representing how likely each character is to be deleted
        self.ins_probs = MLEProbDist(FreqDist()) # a MLE probability distribution representing how likely each character is to be inserted
        self.sub_probs = ConditionalProbDist(ConditionalFreqDist(), MLEProbDist) # a Conditional Probability Distribution representing how likely a given character is to be replaced by another character
    
    def ins_cost(self, x):
        """Given a single character as input,
        returns a cost (between 0 and 1) of inserting that character."""
        ins_prob = self.ins_probs.prob(x)
        return float(1 - ins_prob)
    
    def del_cost(self, x):
        """Given two characters as input,
        returns a cost (between 0 and 1) of substituting the first character with the second character."""
        del_prob = self.del_probs.prob(x)
        return float(1 - del_prob)
    
    def sub_cost(self, x, y):
        """Given two characters as input,
        returns a cost (between 0 and 1) of substituting the first character (x) with the second character (y)."""
        if x == y:
            return 0.0
        else:
            return 2.0 * (1.0-float(self.sub_probs[x].prob(y))) # order of x and y
        
    def align(self, start, end):
        """Given two words, returns a distance (as a float) and the corresponding character alignments
        (as a list of tuples of characters)."""
        numRows = len(start)+1
        numColumns = len(end)+1
        dptable = np.array(([[0]*numColumns]*numRows), dtype=object)
        
        # each cell in the dp table will consist of (cost, char befor modification, char after modification)
        # e.g. if the last action was to delete 'a' and the resulting cost is 10, (10, a, %)
        
        # base cases
        dptable[numRows-1, 0] = (0.0, '%', '%')
        ## fill in the bottom row
        for i in range(1, numColumns):
            char = end[i-1]
            cost = dptable[numRows-1, i-1][0] + self.ins_cost(char)
            dptable[numRows-1, i] = (cost, '%', char)
        ## fill in the first column
        for j in range(numRows-2, -1, -1):
            char = start[numRows-j-2]
            cost = dptable[j+1, 0][0] + self.del_cost(char)
            dptable[j, 0] = (cost, char, '%')
        
        # fill in the rest of the table
        newStart = "%" + start
        newEnd = "%" + end
        for row in range(numRows-2, -1, -1):
            for col in range(1, numColumns):
                sub_cost = dptable[row+1][col-1][0] + self.sub_cost(newStart[len(newStart)-row-1], newEnd[col])
                del_cost = dptable[row+1][col][0] + self.del_cost(newStart[len(newStart)-row-1])
                ins_cost = dptable[row][col-1][0] + self.ins_cost(newEnd[col])
                min_cost = min(sub_cost, del_cost, ins_cost)
                # find the move with the least cost and set fromChar and toChar accordingly
                if sub_cost == min_cost:
                    fromChar = newStart[len(newStart)-row-1]
                    toChar = newEnd[col]
                elif del_cost == min_cost:
                    fromChar = newStart[len(newStart)-row-1]
                    toChar = "%"
                elif ins_cost == min_cost:
                    fromChar = "%"
                    toChar = newEnd[col]
                dptable[row, col] = (min_cost, fromChar, toChar)
        
        # backtrace
        row = 0
        col = numColumns-1
        path = []
        while (row != numRows-1 or col != 0):
            fromChar = dptable[row][col][1]
            toChar = dptable[row][col][2]
            path.insert(0, (fromChar, toChar))
            # trace the last action and move to the prior cell
            ## if the prior move was to substitute
            if (fromChar == toChar) or (fromChar != '%' and toChar != '%'):
                row += 1
                col -= 1
            ## if the prior move was to insert
            elif (fromChar == '%'):
                col -= 1
            ## if the prior move was to delete
            else:
                row += 1
            
        return (dptable[0, numColumns-1][0], path)
    
    def show_alignment(self, alignment): # user has to feed an align result
        """Takes the alignments returned by align and print them in a friendly way."""
        string1 = [a[0] for a in alignment]
        string2 = [a[1] for a in alignment]
        print ("String1:", ' '.join(string1))
        print ("String2:", ' '.join(string2))
        return
    
    def train(self, file):
        """Given a file name, reads in the file and split it into a list of tuples,
        e.g. [(misspelling1, correctspelling1), (misspelling2, correctspelling2), ...],
        then iteratively call train_alignments and train_costs repeatedly until the model converges."""
        pairs = [(pair[0], pair[1]) for pair in [sentence.strip('\n').split(',') for sentence in open(file).readlines()]]
        prior = None
        converged = False
        while not converged:
            print ("Converging...")
            alignments = self.train_alignments(pairs)
            self.train_costs(alignments)
            # check for convergence
            if alignments == prior:
                converged = True
            prior = alignments
        return
    
    def train_alignments(self, misspellings):
        """Given a list of misspellings like the one returned by train, calls align on each of the (misspelling, correctspelling) pairs,
        and returns a single list with all of the character alignments from all of the pairs."""
        align_list = []
        for i in range(len(misspellings)):
            align_list += self.align(misspellings[i][0], misspellings[i][1])[1]
        return align_list
    
    def train_costs(self, alignments):
        """Given a list of character alignments, uses it to estimate the likelihood of different types of errors."""
        # find all of the deletions, insertions, and substitutions in the alignment list
        deletions = []
        insertions = []
        substitutions = []
        for alignment in alignments:
            fromChar = alignment[0]
            toChar = alignment[1]
            if ((fromChar == toChar) or (fromChar != '%' and toChar != '%')):
                substitutions.append(alignment)
            elif fromChar == '%':
                insertions.append(toChar)
            else: # toChar == '%'
                deletions.append(fromChar)
        
        # use the result above to update the probability distributions scores in del_probs, ins_probs, and sub_probs
        self.del_probs = MLEProbDist(FreqDist(deletions))
        self.ins_probs = MLEProbDist(FreqDist(insertions))
        self.sub_probs = ConditionalProbDist(ConditionalFreqDist([(pair[0], pair[1]) for pair in substitutions]), MLEProbDist)
        return


if __name__ == "__main__":
    # test align 1
    aligner_1 = EditDistanceFinder()
    dist, alignments = aligner_1.align("caught", "cought")
    print("Distance between 'caught' and 'cought' is", dist)
    aligner_1.show_alignment(alignments)

    print()

    # test align 2
    aligner_2 = EditDistanceFinder()
    dist, alignments = aligner_2.align("intention", "execution")
    print("Distance between 'intention' and 'execution' is", dist)
    aligner_2.show_alignment(alignments)

    print ()

    # test align 3
    aligner_3 = EditDistanceFinder()
    dist, alignments = aligner_3.align("ant", "aunt")
    #print (aligner_3.align("ant", "aunt"))
    print("Distance between 'ant' and 'aunt' is", dist)
    aligner_3.show_alignment(alignments)

    print ()

    # test all (including training functions)
    my_aligner = EditDistanceFinder()
    my_aligner.train("wikipedia_misspellings.txt")

    print ()

    dist, alignments = my_aligner.align("caught","cought")
    print("Distance between 'caught' and 'cought' is", dist)
    my_aligner.show_alignment(alignments)
    print()
    dist, alignments = my_aligner.align("cougt","cought")
    print("Distance between 'cougt' and 'cought' is", dist)
    my_aligner.show_alignment(alignments)
    print()
    dist, alignments = my_aligner.align("chocolate","healthy")
    print("Distance between 'chocolate' and 'healthy' is", dist)
    my_aligner.show_alignment(alignments)
    print()