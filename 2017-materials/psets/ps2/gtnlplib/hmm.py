from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param i: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    returnDict = {}
    if(curr_tag!=END_TAG):
        returnDict[(curr_tag,tokens[m],EMIT)]=1
        returnDict[(curr_tag,prev_tag,TRANS)]=1
    else:
        returnDict[(curr_tag,prev_tag,TRANS)]=1
    return returnDict;
    

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    # hint: these are your first two lines
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    # hint: call compute_transition_weights
    weights = compute_transition_weights(tag_trans_counts,smoothing)

    # hint: set weights for illegal transitions to -np.inf
    for prev_tag in all_tags:
        weights[(prev_tag,END_TAG,TRANS)]=-np.inf
        weights[(END_TAG,END_TAG,TRANS)]=-np.inf
        weights[(START_TAG,prev_tag,TRANS)]=-np.inf
        weights[(START_TAG,START_TAG,TRANS)]=-np.inf
        weights[(START_TAG,END_TAG,TRANS)]=-np.inf
        weights[(END_TAG,START_TAG,TRANS)]=-np.inf

    # hint: call get_tag_word_counts and estimate_nb_tagger
    tag_word_counts = most_common.get_tag_word_counts(trainfile);
    # print tag_word_counts
    counter_items = np.array(tag_word_counts.items())
    counters = counter_items[:,1] 
    update_nb_tagger = defaultdict(float)
    nb_tagger = naive_bayes.estimate_nb_tagger(tag_word_counts,smoothing)
    for key in nb_tagger:
        value = nb_tagger[key];
        # print "key: ", key
        # print type(key)
        # print "value: ", value
        if(key[0]!=OFFSET and key[1]!=OFFSET):
            new_key = (key[0],key[1],EMIT)
            update_nb_tagger[new_key]=value
    # print nb_tagger
    # print counters
    # print "counters: ", counters
    # for count in counters:
    # nb_tagger = naive_bayes.estimate_nb_tagger(counters,smoothing)
    # print "nb_tagger: ", nb_tagger
    # hint: Counter.update() combines two Counters
    newDict = defaultdict(float)
    newDict.update(update_nb_tagger)
    newDict.update(weights)

    # hint: return weights, all_tags
    return newDict, all_tags
    # raise NotImplementedError


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """
    weights = defaultdict(float)
    counter_items = np.array(trans_counts.items())
    y = counter_items[:,0]   ### List of labels
    x = counter_items[:,1]   ### List of Counters
    for prev_tag in y:
        denominator = trans_counts[prev_tag]
        denominator_value = np.sum(denominator.values())+(len(y)*smoothing)
        for current_tag in y:
            nominator = trans_counts[prev_tag][current_tag]
            nominator+=smoothing
            denominator = trans_counts[prev_tag]
            weights[(current_tag,prev_tag,TRANS)] = np.log(nominator/denominator_value)
        # weights[(prev_tag,END_TAG,TRANS)]=0
        weights[(START_TAG,prev_tag,TRANS)]=-np.inf
    for prev_tag in y:
        denominator = trans_counts[prev_tag]
        denominator_value = np.sum(denominator.values())+(len(y)*smoothing)
        nominator = trans_counts[prev_tag][END_TAG] + smoothing
        weights[(END_TAG,prev_tag,TRANS)] = np.log(nominator/denominator_value)
    return weights
    

