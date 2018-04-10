import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG,TRANS, EMIT, OFFSET
# from gtnlplib import hmm
import numpy as np

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

def viterbi_step(tag, m, words, feat_func, weights, prev_scores):
    """
    Calculate the best path score and back pointer for a given node in the trellis

    :param tag: The tag for which we want to calculate the best path
    :param m: index of the token for which we want to calculate the best tag
    :param words: the list of tokens to tag
    :param feat_func: A function of (words, curr_tag, prev_tag, curr_index) that produces features 
    :param weights: A defaultdict that maps features to numeric score. Should not key error for indexing into keys that do not exist.
    :param prev_scores: a dict, in which keys are tags for token m-1 and values are viterbi scores
    :returns: tuple of (best_score, best_feature), where
        best_score   -- The highest score of any sequence of tags
        best_feature -- The feature in the previous layer of the trellis corresponding
            to the best score

    :rtype: tuple

    """
    scores = {}
    for pv in prev_scores:
        prev_tag = pv
        prev_score = prev_scores[prev_tag]
        feat = feat_func(words,tag,prev_tag,m)
        feat_keys = feat.keys();
        total = prev_score
        # print "prev_tage_score: ", total
        for key in feat_keys:
            w = weights[key];
            # print "key: ", key
            # print "emit or tran: ", w
            total += w
            scores[prev_tag]=total

    # print "scores: ", scores
    best_score = max(scores.values())
    best_tag = argmax(scores)
    
    return best_score, best_tag

def build_trellis(tokens,feat_func,weights,all_tags):
    """Construct a trellis for the hidden Markov model. Output is a list of dicts.

    :param tokens: list of word tokens to be tagged
    :param feat_func: feature function (words, tag, prev_tag, index)
    :param weights: defaultdict of weights
    :param all_tags: list/set of all possible tags
    :returns: list of dicts, length = len(words)
    first dict should represent score from start to token 1, 
    then score from token 1 to token 2,
    etc until token M
    :rtype: list of dicts

    """
    
    trellis = [None]*(len(tokens))

    # build the first column separately
    prev_scores={}
    vsList = []
    prev_scores[START_TAG]=0
    first = {}
    currentDict ={};

    total_prev_score = []
    for tag in all_tags:
        vs = viterbi_step(tag, 0, tokens, feat_func, weights, prev_scores);
        currentDict[tag]=vs[0]
        first[tag] = vs
    trellis[0] = first # your code here
    total_prev_score.append(currentDict)

    # iterate over the remaining columns
    for m in range(1,len(tokens)):
        currentDict = {};
        current_level = {}
        for tag in all_tags:
            prev_scores = total_prev_score[m-1];
            vs = viterbi_step(tag, m, tokens, feat_func, weights, prev_scores);
            currentDict[tag]=vs[0]
            current_level[tag] = vs
        trellis[m] = current_level
        total_prev_score.append(currentDict)
        
    return trellis


def viterbi_tagger(tokens,feat_func,weights,all_tags):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words     -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights   -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags  -- A set of all possible tags

        Returns:
        tags       -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
    """
    trellis = build_trellis(tokens,feat_func,weights,all_tags)
    last_trellis = trellis[-1]
    best_score={}
    last_tag = None
    best_score = dict()
    for item in last_trellis:
        last_trellis_key = last_trellis[item]
        best_score[item] = last_trellis_key[0]
    last_tag = argmax(best_score)
    best_score = max(best_score.values())
    reverse_output = [last_tag]
    for m,v_m in enumerate(reversed(trellis[1:])):
        last_tag = v_m[last_tag][1]
        reverse_output.insert(0,last_tag)
    return reverse_output,best_score

