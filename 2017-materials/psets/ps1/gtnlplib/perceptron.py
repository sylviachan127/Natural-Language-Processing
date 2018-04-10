from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string 
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels 
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    y_max = predict(x,weights,labels);
    y_max_label = y_max[0];
    fv = make_feature_vector(x,y);
    fv_new = make_feature_vector(x,y_max_label);
    new_thetha = defaultdict(float);
    if(y_max_label!=y):
        for f in fv:
            #new_thetha[f] = weights[f] +fv[f];
            new_thetha[f]+=fv[f];
        for f in fv_new:
            #new_thetha[f] = weights[f] - fv_new[f];
            new_thetha[f]-=fv_new[f];
    return new_thetha


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            pu = perceptron_update(x_i,y_i,weights,labels)
            for v_p in pu:
                weights[v_p] += pu[v_p]
        weight_history.append(weights.copy())
    return weights, weight_history

def estimate_avg_perceptron(x,y,N_its):
    """estimate averaged perceptron classifier

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    w_sum = defaultdict(float) #hint
    weights = defaultdict(float)
    weight_history = []
    avg_weights = defaultdict(float)
    
    t=1.0 #hint
    for it in xrange(N_its):
        #t=1;
        for x_i,y_i in zip(x,y):
            pu = perceptron_update(x_i,y_i,weights,labels);
            #t=1;
            for pu_key in pu:
                weights[pu_key]+=pu[pu_key];
                w_sum[pu_key]+=(t*pu[pu_key]);
            t+=1;
        for w in weights:
            avg_weights[w]= weights[w]-((w_sum[w])/t);
           # avg_weights[w]=((w_sum[w]-(weights[w])/t)
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history
