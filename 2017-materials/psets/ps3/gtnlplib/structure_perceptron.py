from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag 
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    M = len(tokens)
    updateDict = defaultdict(float)
    assume_tag = (tagger(tokens,feat_func,weights,all_tags))[0]
    for index in range(len(assume_tag)):
        if(assume_tag[index]!=tags[index]):
            wrong_wf = feat_func(tokens,assume_tag[index],assume_tag[index-1],index)
            for keys in wrong_wf:
                updateDict[keys]-=wrong_wf[keys]
            right_wf = feat_func(tokens,tags[index],tags[index-1],index)
            for keys in right_wf:
                updateDict[keys]+=right_wf[keys]
    return updateDict
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    # compute all_tags if it's not provided
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    # this initialization should make sure there isn't a tie for the first prediction
    # this makes it easier to test your code
    weights = defaultdict(float,
                          {('NOUN',constants.OFFSET):1e-3})

    weight_history = []

    w_sum = defaultdict(float) 
    avg_weights = defaultdict(float)
    
    t=0 #hint
    for it in xrange(N_its):
        for index in range(len(labeled_instances)):
            words = labeled_instances[index][0]
            tags = labeled_instances[index][1]   
            pu = sp_update(words,tags,weights,feat_func,tagger,all_tags)
            # t+=1
            for pu_key in pu:
                weights[pu_key]+=pu[pu_key];
                # weights[pu_key]+=pu[pu_key]/t
                w_sum[pu_key]+=(t*pu[pu_key]);
            t+=1;
        avg_weights = defaultdict(float)
        for w in weights:
            avg_weights[w]= weights[w]-((w_sum[w])/t);
        weight_history.append(avg_weights.copy())
    # print "avg_weights: ", len(avg_weights)

    # set to correct 3.1
    # key = (constants.END_TAG,constants.OFFSET)
    # if key in avg_weights:
    #     del avg_weights[key]

    return avg_weights, weight_history



