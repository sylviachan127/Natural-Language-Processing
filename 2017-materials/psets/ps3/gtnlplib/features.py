from gtnlplib import constants

# Deliverable 1.1
def word_feats(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    Note! You need to handle the case where $m >= len(words)$. In this case, you should only output the offset feature. 

    :param words: list of word tokens
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    fv = dict() 
    if(m<len(words)):
        fv[(y,constants.CURR_WORD_FEAT,words[m])] = 1
    fv[(y,constants.OFFSET)] = 1
    return fv

# Deliverable 2.1
def word_suff_feats(words,y,y_prev,m):
    """This function should return all the features returned by word_feats,
    plus an additional feature for each token, indicating the final two characters.

    You may call word_feats in this function.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    fv = word_feats(words,y,y_prev,m)
    if(m<len(words)):
        word = words[m]
        if(len(word)>=2):
            subfix = word[-2:]
            fv[(y,constants.SUFFIX_FEAT,subfix)]=1
        else:
            fv[(y,constants.SUFFIX_FEAT,word)]=1
    return fv;
    
def word_neighbor_feats(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """

    returnDict = dict()
    if(m<len(words)):
        currentKey = (y,constants.CURR_WORD_FEAT,words[m])
        returnDict[currentKey]=1
    offsetKey = (y,constants.OFFSET)
    returnDict[offsetKey]=1
    if(m<=len(words)):
        if(m!=0):
            prevKey = (y,constants.PREV_WORD_FEAT,words[m-1])
        else:
            prevKey = (y,constants.PREV_WORD_FEAT,constants.PRE_START_TOKEN)
        returnDict[prevKey]=1
    if(m<len(words)):
        if(m!=(len(words)-1)):
            nextKey = (y,constants.NEXT_WORD_FEAT,words[m+1])
        else:
            nextKey = (y,constants.NEXT_WORD_FEAT,constants.POST_END_TOKEN)
        returnDict[nextKey]=1
    return returnDict

    
def word_feats_competitive_en(words,y,y_prev,m):
    # 0.893685051958
    rDict = word_suff_feats(words,y,y_prev,m)
    # add first two word
    if(m<len(words)):
        word = words[m]
        if(len(word)>=2):
            subfix = word[:2]
            rDict[(y,constants.SUFFIX_FEAT,subfix)]=1

    y = word_neighbor_feats(words,y,y_prev,m)
    z = rDict.copy()
    z.update(y)
    return z
    
def word_feats_competitive_ja(words,y,y_prev,m):
    # 0.917909610856
    rDict = word_suff_feats(words,y,y_prev,m)
    # add first two word
    if(m<len(words)):
        word = words[m]
        if(len(word)>=2):
            subfix = word[:2]
            rDict[(y,constants.SUFFIX_FEAT,subfix)]=1


    y = word_neighbor_feats(words,y,y_prev,m)
    z = rDict.copy()
    z.update(y)
    return z

def hmm_feats(words,y,y_prev,m):
    # 0.81934452438
    returnDict = dict()
    if(m==0):
        y_prev = constants.START_TAG
    if(y==constants.END_TAG):
        returnDict[(constants.END_TAG,constants.PREV_TAG_FEAT,y_prev)]=1
    else:
        returnDict[(y,constants.CURR_WORD_FEAT,words[m])]=1
        returnDict[(y,constants.PREV_TAG_FEAT,y_prev)]=1
    return returnDict;

def hmm_feats_competitive_en(words,y,y_prev,m):
    returnDict = word_feats_competitive_en(words,y,y_prev,m)
    if(m==0):
        y_prev = constants.START_TAG
    if(y==constants.END_TAG):
        returnDict[(constants.END_TAG,constants.PREV_TAG_FEAT,y_prev)]=1
    else:
        returnDict[(y,constants.CURR_WORD_FEAT,words[m])]=1
        returnDict[(y,constants.PREV_TAG_FEAT,y_prev)]=1

    return returnDict;

def hmm_feats_competitive_ja(words,y,y_prev,m):
    returnDict = word_suff_feats(words,y,y_prev,m)
    if(m==0):
        y_prev = constants.START_TAG
    if(y==constants.END_TAG):
        returnDict[(constants.END_TAG,constants.PREV_TAG_FEAT,y_prev)]=1
    else:
        returnDict[(y,constants.CURR_WORD_FEAT,words[m])]=1
        returnDict[(y,constants.PREV_TAG_FEAT,y_prev)]=1
    return returnDict;


