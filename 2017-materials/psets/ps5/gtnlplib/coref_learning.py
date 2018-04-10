import numpy as np
from collections import defaultdict
import coref

# deliverable 3.2
def mention_rank(markables,i,feats,weights):
    """ return top scoring antecedent for markable i

    :param markables: list of markables
    :param i: index of current markable to resolve
    :param feats: feature function
    :param weights: weight defaultdict
    :returns: index of best scoring candidate (can be i)
    :rtype: int

    """
    highestAnt = 0
    highestScore = 0
    for index in range(i+1):
        feat_score = feats(markables,index,i)
        currentScore = 0
        for key, value in feat_score.iteritems():
            key_weight = weights[key]
            currentScore += (key_weight*value)
        if(currentScore>highestScore):
            highestScore = currentScore
            highestAnt = index
    return highestAnt
    ## hide
    # raise NotImplementedError
    
# deliverable 3.3
def compute_instance_update(markables,i,true_antecedent,feats,weights):
    """Compute a perceptron update for markable i.
    fp + fn should be update

    true!= pred - >fn

    i + true - > +
    predict + i -> - 
    This function should call mention_rank to determine the predicted antecedent,
    and should make an update if the true antecedent and predicted antecedent *refer to different entities*

    Note that if the true and predicted antecedents refer to the same entity, you should not
    make an update, even if they are different.

    :param markables: list of markables
    :param i: current markable
    :param true_antecedent: ground truth antecedent
    :param feats: feature function
    :param weights: defaultdict of weights
    :returns: dict of updates
    :rtype: dict

    """
    # # keep
    pred_antecedent = mention_rank(markables,i,feats,weights)
    returnDict = dict()
    fn = False
    fp = False
    i_ent = markables[i]['entity']
    pred_ent = markables[pred_antecedent]['entity']
    true_ent = markables[true_antecedent]['entity']
    if(true_ent!=pred_ent) or (pred_ent!=i_ent):
        fp = True
    if(true_antecedent<i):
        if(pred_ent!=i_ent) or (pred_antecedent==i):
            fn = True
    if(fp or fn):
        fv = feats(markables, true_antecedent,i)
        fv_hat = feats(markables, pred_antecedent,i)
        returnDict = fv
        for key, value in fv_hat.iteritems():
            if(key in returnDict):
                returnDict[key] = returnDict[key]-value
            else:
                returnDict[key] = -value

    return returnDict
    
# deliverable 3.4
def train_avg_perceptron(markables,features,N_its=20):
    # the data and features are small enough that you can
    # probably get away with naive feature averaging

    weights = defaultdict(float)
    tot_weights = defaultdict(float)
    weight_hist = []
    T = 0

    # print "at 3.4_sylvia"
    for it in xrange(N_its):
        num_wrong = 0 #helpful but not required to keep and print a running total of errors
        index = 0
        for document in markables:
            # YOUR CODE HERE
            true_antecedent = coref.get_true_antecedents(document)
            for i in range(len(document)):
                pu = compute_instance_update(document, i, true_antecedent[i], features, weights)
                if(len(pu)!=0):
                    num_wrong+=1
                for pu_key in pu:
                    weights[pu_key]+=pu[pu_key]
                for w_key in weights:
                    tot_weights[w_key]+=weights[w_key]
                T+=1
        print num_wrong,

        # update the weight history
        weight_hist.append(defaultdict(float))
        for feature in tot_weights.keys():
            weight_hist[it][feature] = tot_weights[feature]/T

    return weight_hist

# helpers
def make_resolver(features,weights):
    return lambda markables : [mention_rank(markables,i,features,weights) for i in range(len(markables))]
        
def eval_weight_hist(markables,weight_history,features):
    scores = []
    for weights in weight_history:
        score = coref.eval_on_dataset(make_resolver(features,weights),markables)
        scores.append(score)
    return scores
