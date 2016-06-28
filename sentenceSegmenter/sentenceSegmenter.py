# A simple sentence segmenter

# Sample usage:
# 
# hyp = sentence_segment("adventure.txt")
# run_evaluation("adventure.txt", hyp, "adventure-eos.txt")

def concordance(words, i, eval=''):
    concordance = {
        'prechars': ' '.join(words[max(0,i-20):i])[-30:],
        'center': words[i],
        'postchars': ' '.join(words[i+1:])[:30],
        'label': eval
    }
        
    print("{prechars: >30}  {center: <20}  {postchars: <30}  ({label})".format(**concordance))    

def evaluate(guesses, truths, words, verbose=0):
    """
    verbose:
      0 = show nothing
      1 = show incorrect
      2 = show all
    """
    
    hyp = set(guesses)
    ref = set(truths)
    full = set(range(len(words)))
    tp = hyp & ref
    tn = full-(hyp|ref)   
    fp = hyp-ref
    fn = ref-hyp
    
    if verbose == 2:
        for i in tp: concordance(words, i, 'TP')
        for i in tn: concordance(words, i, 'TN')
    if verbose >= 1:
        for i in fp: concordance(words, i, 'FP')
        for i in fn: concordance(words, i, 'FN')    
            
    return {"tp":len(tp), "fp":len(fp), "fn":len(fn), "tn":len(tn)}

def run_evaluation(word_file, hyp, ref_file, verbose=0):
    """Given a word file, guesses for EOS, and a reference label file (that contains all the line numbers of the end of sentence),
    reports the true positives, true negatives, false positives and false negatives."""
    words = [x.rstrip() for x in open(word_file)]
    ref = [int(x.rstrip()) for x in open(ref_file)]
    
    eval_results = evaluate(hyp, ref, words, verbose)

    print("TP: {tp:7d}\tFN: {fn:7d}".format(**eval_results))
    print("FP: {fp:7d}\tTN: {tn:7d}".format(**eval_results))

def sentence_segment(filename):
    """Given the name of a text file, returns a list of the line numbers that mark the end of sentences in the file."""
    end_of_sentence = []
    words = []
    
    with open(filename) as f:
        for line in f:
            words.append(line.strip("\n"))
              
    for i in range(1, len(words)):
        # Case 1: sentence ends in ;!? (unambiguous marks of ends of sentences, note: duplicated in training files)
        if words[i] in ";!?" and words[i-1] in ";!?":
            end_of_sentence.append(i)
        
        # Case 2: sentence ends in .
        if words[i] == ".": # this takes care of abbreviations, which do not exactly match "."
            end_of_sentence.append(i)

        # Case 3: sentence ends in :
        if words[i] == ":":
            if words[i+1][0].isupper():
                end_of_sentence.append(i)
        
        # Special case 1: -- followed by capital letters or ``
        if i < len(words)-1 and words[i] == "--":
            if words[i+1] == "``":
                end_of_sentence.append(i)
        
        # Special case 2: '' followed by capital letters
        if i < len(words)-1 and words[i] == "''" and words[i+1][0].isupper():
            end_of_sentence.append(i)
    
    return end_of_sentence

if __name__ == "__main__":
    hyp = sentence_segment("editorial.txt")
    run_evaluation("editorial.txt", hyp, "editorial-eos.txt")
    print()
    hyp = sentence_segment("adventure.txt")
    run_evaluation("adventure.txt", hyp, "adventure-eos.txt")
    print()
    hyp = sentence_segment("humor.txt")
    run_evaluation("humor.txt", hyp, "humor-eos.txt")