import numpy as np #hint: np.log
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    lableIndex = 0
    count=0.0
    cc = defaultdict(lambda: 0)
    for word in x:
        if(label==y[lableIndex]):
            for w in word:
                count = float(word[w])
                if(cc.get(w)==None):
                    cc[w]=count
                else:
                    cc[w]+=count
        lableIndex+=1
    return cc

def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    cc = defaultdict(float)
    labelCount = 0
    corpusCount = get_corpus_counts(x,y,label)
    totalWordCount = 0
    vvWordCount = 0
    for ww in corpusCount:
        totalWordCount += corpusCount[ww]
        
    #print "totalWordCount", totalWordCount
    
    totalWordCountWithSmooth = totalWordCount+(len(vocab)*smoothing)
    couting = 0
    
    #print "vocab_length: ", len(vocab)
    #print "len_cc: ", len(corpusCount)
    for ys in y:
        if(ys==label):
            labelCount+=1
            
    for vocabs in vocab:
        couting+=1
        corpusC = corpusCount[vocabs]+smoothing
        #corpusC = corpusCount[vocabs]
        #print vocabs
        cc[vocabs] = np.log(corpusC/totalWordCountWithSmooth)
        #cc[vocabs]+=smoothing
    
    #print "couting", couting
    return cc

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    vocab = set()
    labels = set()
    for label in y:
        labels.add(label)
    weights = defaultdict(float)
    for bag_of_words in x:
        for vocabs in bag_of_words:
            vocab.add(vocabs)
    #print "vocabs", vocab
    count = 0
    for label in labels:
        #print count
        pxy = estimate_pxy(x,y,label,smoothing,vocab)
        for pxy_iter in pxy:
            #print pxy_iter, pxy[pxy_iter]
            weights[label,pxy_iter] = pxy[pxy_iter]
        numDoc_label = 0
        for labels_y in y:
            if(labels_y == label):
                numDoc_label+=1
        logOffSet = np.log(float(numDoc_label)/float(len(y)))
        #len(y)
        weights[(label,OFFSET)] = logOffSet
        count+=1
    return weights

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    tagger_nb = defaultdict(float)
    counter_items = np.array(counters.items())
    y = counter_items[:,0]   ### List of labels
    x = counter_items[:,1]   ### List of Counters
    # print y
    theta_word_tag = estimate_nb(x,y,smoothing)
    tag_count = {}
    for tags, counters in counters.items():
        tag_count[tags] = np.sum(counters.values())

    denominator = np.sum(tag_count.values())
    for tags, counters in tag_count.items():
        if(counters!=0):
            theta_word_tag[tags,OFFSET] = np.log(counters)-np.log(denominator)

    return theta_word_tag
