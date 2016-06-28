# The following code was written by Professor Julie Medero.

from nltk.probability import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist
import numpy
import string
from math import log

class EditDistanceFinder():  
    def __init__(self):
        self.char_probs = ConditionalProbDist([],MLEProbDist)
        self.bichar_freqs = ConditionalFreqDist([])
        self.transp_freqs = FreqDist()
        self.DOWN,self.LEFT,self.DIAG,self.DOUBLE_DIAG = range(4)
        self.INSERT, self.DELETE, self.SUBST, self.TRANSP = range(4)
        
    def train(self, fname):
        misspellings=[]
        for line in open(fname):
            line=line.strip()
            if not(line): continue
            w1, w2 = line.split(",")
            misspellings.append((w1.strip(),w2.strip()))
       
        last_alignments = None
        done = False
        while not done:
            print("Iteration")
            alignments, bigrams = self.train_alignments(misspellings)
            self.train_costs(alignments, bigrams)
            done = (alignments == last_alignments)
            last_alignments = alignments
            
    def train_alignments(self, misspellings):
        alignments = []
        self.bichar_freqs = FreqDist()

        for error, corrected in misspellings:
            distance, this_alignments = self.align(corrected, error)
            alignments += this_alignments
            bigrams = [corrected[i:i+2] for i in range(len(corrected)-1)]
            self.bichar_freqs.update(bigrams)
            
        return alignments,bigrams
    
    def train_costs(self, alignments,bigrams):
        add_one_aligns = [(a,b) for a in string.ascii_lowercase for b in string.ascii_lowercase]
        single_aligns = [(a,b) for a,b in alignments if len(a) < 2]
        
        char_aligns = ConditionalFreqDist(single_aligns + add_one_aligns)
        self.char_probs = ConditionalProbDist(char_aligns, MLEProbDist)
        
        double_aligns = [a for a,b in alignments if len(a) >= 2]
        self.transp_freqs = FreqDist(double_aligns)

    def align(self, w1, w2, verbose=False):
        M = len(w1) +1
        N = len(w2) +1
        table = numpy.zeros((M,N))
        backtrace = numpy.zeros((M,N))
    
        for i in range(1,M):
            w1_char = w1[i-1]
            table[i,0] = table[i-1,0] + self.del_cost(w1_char)
            backtrace[i,0] = self.DOWN
        for j in range(1,N):
            w2_char = w2[j-1]
            backtrace[0,j] = self.LEFT
            table[0,j] = table[0,j-1] + self.ins_cost(w2_char)   
    
        for i in range(1,M):
            w1_char = w1[i-1]
            for j in range(1,N):
                w2_char = w2[j-1]

                this_del = table[i-1,j] + self.del_cost(w1_char)
                this_ins = table[i,j-1] + self.ins_cost(w2_char)
                this_sub = table[i-1,j-1] + self.sub_cost(w1_char,w2_char)
                
                if j > 1 and i > 1 and w1[i-1] == w2[j-2] and w1[i-2]==w2[j-1] and w1[i-1] != w1[i-2]:
                    this_transp = table[i-2,j-2] + self.transp_cost(w1_char, w2_char)
                else:
                    this_transp = 999999
            
                min_cost = min(this_del, this_ins, this_sub, this_transp)
                table[i,j] = min_cost

                if this_sub == min_cost:
                    backtrace[i,j] = self.DIAG
                elif this_transp == min_cost:
                    backtrace[i,j] = self.DOUBLE_DIAG
                elif this_ins == min_cost:
                    backtrace[i,j] = self.LEFT
                else: # insert
                    backtrace[i,j] = self.DOWN

                
        alignments = []
        i = M - 1    
        j = N - 1
        while (j or i):
            this_backtrace = backtrace[i,j]
            if this_backtrace == self.DIAG: # sub
                alignments.append((w1[i-1],w2[j-1]))
                i -= 1
                j -= 1
            elif this_backtrace == self.DOUBLE_DIAG:
                alignments.append((w1[i-2:i],w2[j-2:j]))
                i -= 2
                j -= 2
            elif this_backtrace == self.DOWN: # delete
                alignments.append((w1[i-1],"%"))
                i -= 1
            elif this_backtrace == self.LEFT: # insert
                alignments.append(("%",w2[j-1]))
                j -= 1

        alignments.reverse()
        if verbose:
            print(table)
        return table[M-1,N-1], alignments

    def transp_cost(self, char1, char2):
        ## how often do char1 and char2 get transposed?
        return 1 - self.transp_prob(char1,char2)
   
    def del_cost(self, char):
        return 1-self.char_probs[char].prob('%')
    def ins_cost(self, char):
        return 1-self.char_probs['%'].prob(char)
    def sub_cost(self, char1, char2):
        return 1-self.char_probs[char1].prob(char2)
    
    def transp_prob(self, char1, char2):
        numerator = self.transp_freqs[char1] + .1
        denominator = self.bichar_freqs[char1] + .1*26*26
        return numerator / denominator
    
    def prob(self, w1, w2):
        score, alignment = self.align(w1, w2)
        total_prob = 0
        for a, b in alignment:
            if len(a) > 1:
                total_prob += log(self.transp_prob(a[0],a[1]))
            else:
                total_prob += self.char_probs[a].logprob(b)
        return total_prob
    
    def show_alignment(self, alignments):
        print("String1:", " ".join([x[0] for x in alignments]))
        print("String2:", " ".join([x[1] for x in alignments]))