from gtnlplib.constants import OFFSET

import operator
# use this to find the highest-scoring label
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    fv = {}
    fv[(label,OFFSET)]=1
    for bf in base_features:
        fv[(label,bf)]=base_features[bf]
        # fv[(label,bf)]+=1
        # if(fv[(label,bf)]==None):
        #     fv[(label,bf)]=1
        # else:
        #     fv[(label,bf)]+=1
    # print "fv", fv
    return fv
    
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    # print "base_features: ", base_features
    # print "weights: ", weights
    # for w in weights:
    #     print w

    scores = {}
    for label in labels:
        fv = make_feature_vector(base_features,label)
        for fv_key in fv:
            if(weights[fv_key]!=0 and weights[fv_key]!=None):
            # if(weights[fv_key]!=None):
                currentWeight = (fv[fv_key]*weights[fv_key])
                # print "current Label: ", label, " done"
                if(scores.get(label)==None):
                    scores[label]=currentWeight
                else:
                    # print "label: ", label
                    # print "scores: " , scores
                    # print "scores[label]: ", scores[label]
                    # print "currentWeight: ", currentWeight
                    scores[label]+=currentWeight
    # print argmax(scores)     
    return argmax(scores),scores
