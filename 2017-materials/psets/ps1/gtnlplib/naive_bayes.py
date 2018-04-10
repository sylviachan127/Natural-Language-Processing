from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    lableIndex = 0;
    count=0.0;
    cc = defaultdict(lambda: 0);
    for word in x:
        if(label==y[lableIndex]):
            for w in word:
                count = float(word[w])
                if(cc.get(w)==None):
                    cc[w]=count;
                else:
                    cc[w]+=count;
        lableIndex+=1;
    return cc;
    
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
    cc = defaultdict(float);
    labelCount = 0;
    corpusCount = get_corpus_counts(x,y,label);
    totalWordCount = 0;
    vvWordCount = 0;
    for ww in corpusCount:
        totalWordCount += corpusCount[ww];
        
    #print "totalWordCount", totalWordCount;
    
    totalWordCountWithSmooth = totalWordCount+(len(vocab)*smoothing);
    couting = 0;
    
    #print "vocab_length: ", len(vocab)
    #print "len_cc: ", len(corpusCount)
    for ys in y:
        if(ys==label):
            labelCount+=1;
            
    for vocabs in vocab:
        couting+=1
        corpusC = corpusCount[vocabs]+smoothing;
        #corpusC = corpusCount[vocabs];
        #print vocabs
        cc[vocabs] = np.log(corpusC/totalWordCountWithSmooth);
        #cc[vocabs]+=smoothing
    
    #print "couting", couting;
    return cc;
        
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    #print "len(x): ", len(x)
    #print "len(y): ", len(y)
    vocab = set();
    labels = set();
    for label in y:
        labels.add(label);
    weights = defaultdict(float)
    for bag_of_words in x:
        for vocabs in bag_of_words:
            vocab.add(vocabs);
    #print "vocabs", vocab
    count = 0;
    for label in labels:
        #print count;
        pxy = estimate_pxy(x,y,label,smoothing,vocab);
        for pxy_iter in pxy:
            #print pxy_iter, pxy[pxy_iter];
            weights[label,pxy_iter] = pxy[pxy_iter];
        numDoc_label = 0
        for labels_y in y:
            if(labels_y == label):
                numDoc_label+=1;
        logOffSet = np.log(float(numDoc_label)/float(len(y)))
        #len(y)
        weights[(label,OFFSET)] = logOffSet;
        count+=1;
    return weights
    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    labels = set([u'worldnews', u'science', u'askreddit', u'iama', u'todayilearned']);
    bestAcc = 0;
    returnDict = {}
    for smoothing in smoothers:
        #estimate_nb(x_tr,y_tr,smoothing);
        theta_nb = estimate_nb(x_tr,y_tr,smoothing);
        #dev_predict = clf_base.predict(x_dv,theta_nb,labels);
        #train_predict = clf_base.predict(x_tr,theta_nb,labels);
        y_hat = clf_base.predict_all(x_dv,theta_nb,labels);
        accuracy = evaluation.acc(y_hat,y_dv);
        print "accuracy: ", accuracy
        if(accuracy>bestAcc):
            bestAcc = accuracy;
        returnDict[smoothing] = accuracy;
    return bestAcc,returnDict
    #raise NotImplementedError
