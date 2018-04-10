import itertools
import coref_rules
from nltk import wordnet

# useful?
pronoun_list=['it','he','she','they','this','that']
poss_pronoun_list=['its','his','her','their']
oblique_pronoun_list=['him','her','them']
def_list=['the','this','that','these','those']
indef_list=['a','an','another']

# d3.1
def minimal_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    # count = 0
    if(a==i):
        f['new-entity']=1.0
    else:
        em = coref_rules.exact_match_no_overlap(markables[a],markables[i])
        if(em):
            f['exact-match'] = 1
        lt = coref_rules.match_last_token_no_overlap(markables[a],markables[i])
        if(lt):
            f['last-token-match']=1
        emc = coref_rules.match_on_content(markables[a],markables[i])
        if (emc):
            f['content-match']=1
        mo = coref_rules.mention_overlap(markables[a],markables[i])
        if(mo):
            f['crossover']=1
    return f

# deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=10,
                      max_token_distance=10):
    """compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: feature dict
    :rtype: dict

    """
    f = dict()
    if(a!=i):
        # print x
        a_end= x[a]['end_token']
        i_start = x[i]['start_token']
        if((i_start-a_end)<max_token_distance):
            f['token-distance-'+str(i_start-a_end)]=1
        else:
            f['token-distance-'+str(10)]=1

        if((i-a)<max_mention_distance):
            f['mention-distance-'+str(i-a)]=1
        else:
            f['mention-distance-'+str(10)]=1
    ## your code here
    return f
    
###### Feature combiners

# deliverable 3.6
def make_feature_union(feat_func_list):
    """return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function

    """
    def f_out(x,a,i):
        # your code here
        xx = minimal_features(x,a,i)
        yy = distance_features(x,a,i)
        zz = dict(xx.items() + yy.items())
        return zz
        # return None
    return f_out

# deliverable 3.7
def make_feature_cross_product(feat_func1,feat_func2):
    """return a feature function that is the cross-product of the two feature functions

    :param feat_func1: a feature function
    :param feat_func2: a feature function
    :returns: another feature function
    :rtype: function

    """
    def f_out(x,a,i):
        min_feat = feat_func1(x,a,i)
        dis_feat = feat_func2(x,a,i)
        returnDict = dict()
        for key1, value1 in min_feat.iteritems():
            for key2, value2 in dis_feat.iteritems():
                returnDict[key1+'-'+key2]=value1*value2
        # your code here
        return returnDict
    return f_out

# deliverable 3.9
def make_bakeoff_features():
    def f_out(x,a,i):
        xx = minimal_features(x,a,i)
        yy = distance_features(x,a,i)
        zz = dict(xx.items() + yy.items())

        # return zz
        # min_feat = minimal_features(x,a,i)
        # dis_feat = distance_features(x,a,i,10,10)
        # returnDict = dict()
        # for key1, value1 in min_feat.iteritems():
        #     for key2, value2 in dis_feat.iteritems():
        #         returnDict[key1+'-'+key2]=value1*value2

        bakeoff_min_feat = minimal_features_bakeoff(x,a,i)
        return dict(zz.items() + bakeoff_min_feat.items())
        # return dict(zz.items() + returnDict.items())
        # currentCombine = dict(zz.items() + returnDict.items())
        # return dict(currentCombine.items() + bakeoff_min_feat.items())
        # return None
    return f_out

def minimal_features_bakeoff(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    fm = match_first_token_no_overlap(markables[a],markables[i])
    if(fm):
        f['first-token-match']=1
    numMatch = number_match_no_overlap(markables[a],markables[i])
    if(numMatch):
        f['numMatch']=1
    # genderMatch = gender_match(markables[a],markables[i]) 
    ex_noPronoun = coref_rules.exact_match_no_pronouns(markables[a],markables[i])
    if(ex_noPronoun):
        f['exacti-no-pronoun']=1
    return f

def match_first_token_no_overlap(m_a,m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if final tokens match and strings do not overlap
    :rtype: boolean

    """
    maString = m_a['string']
    maStart = m_a['start_token']
    maEnd = m_a['end_token']

    miString = m_i['string']
    miStart = m_i['start_token']
    miEnd = m_i['end_token']
 
    if(maStart<miStart):
        x1 = maStart
        x2 = miStart
    else:
        x1 = miStart
        x2 = maStart
    if(maEnd<miEnd):
        y1 = maEnd
        y2 = miEnd
    else:
        y1 = miEnd
        y2 = maEnd


    if(maString[0].replace(' ', '').lower()==miString[0].replace(' ', '').lower()):
        if (x1 <= y2 and y1 <= x2):
            return True
        else:
            return False
    else:
        return False

def number_match_no_overlap(m_a,m_i):
    singular = ['a', 'an', 'this', 'that', 'is']
    plural = ['those', 'these', 'some','are']

    maString = m_a['string']
    maStart = m_a['start_token']
    maEnd = m_a['end_token']

    miString = m_i['string']
    miStart = m_i['start_token']
    miEnd = m_i['end_token']
 
    if(maStart<miStart):
        x1 = maStart
        x2 = miStart
    else:
        x1 = miStart
        x2 = maStart
    if(maEnd<miEnd):
        y1 = maEnd
        y2 = miEnd
    else:
        y1 = miEnd
        y2 = maEnd
    maSing = False
    maPlural = False
    for i in range(len(maString)):
        ma_s = maString[i]
        if ma_s in singular:
            maSing = True
        if ma_s in plural:
            maPlural = True
    miSing = False
    miPlural = False
    for i in range(len(miString)):
        mi_s = miString[i]
        if mi_s in singular:
            miSing = True
        if mi_s in plural:
            miPlural = True
    if (x1 <= y2 and y1 <= x2):
        if(miSing and maSing) or ( not miSing and not maSing):
            if(miPlural and maPlural) or (not miPlural and not maPlural):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def gender_match(m_a,m_i):

    maString = m_a['string']
    miString = m_i['string']
    maTag = m_a['tags']
    miTag = m_i['tags']
    contentWord = ['NN','NNS','NNP','NNPS']
    for i in range(len(maTag)):
        if maTag[i] in contentWord:
            print wordnet.synsets(maString[i])
    return False



