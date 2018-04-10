### Rule-based coreference resolution  ###########
# Lightly inspired by Stanford's "Multi-pass sieve"
# http://www.surdeanu.info/mihai/papers/emnlp10.pdf
# http://nlp.stanford.edu/pubs/conllst2011-coref.pdf

import nltk

# this may help
pronouns = ['I','me','mine','you','your','yours','she','her','hers','he','him','his','it','its','they','them','their','theirs','this','those','these','that','we','our','us','ours']
downcase_list = lambda toks : [tok.lower() for tok in toks]

############## Pairwise matchers #######################

def exact_match(m_a,m_i):
    """return True if the strings are identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical
    :rtype: boolean

    """
    return downcase_list(m_a['string'])==downcase_list(m_i['string'])

# deliverable 2.2
def exact_match_no_pronouns(m_a,m_i):
    """return True if strings are identical and are not pronouns

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical and are not pronouns
    :rtype: boolean

    """
    maString = m_a['string']
    miString = m_i['string']
    if(' '.join(m_a['string']).replace(' ', '').lower()==' '.join(m_i['string']).replace(' ', '').lower()):
        if(len(maString)==1):
            if(maString=="i"):
                return True
            if(maString[0].replace(' ', '').lower() in pronouns):
                return False
        return True
    else:
        return False

# deliverable 2.3
def match_last_token(m_a,m_i):
    """return True if final token of each markable is identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :rtype: boolean

    """
    maString = m_a['string']
    miString = m_i['string']
    if(maString[-1].replace(' ', '').lower()==miString[-1].replace(' ', '').lower()):
        return True
    else:
        return False

# deliverable 2.4
def match_last_token_no_overlap(m_a,m_i):
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


    if(maString[-1].replace(' ', '').lower()==miString[-1].replace(' ', '').lower()):
        if (x1 <= y2 and y1 <= x2):
            return True
        else:
            return False
    else:
        return False

def mention_overlap(m_a,m_i):
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


    if (x1 <= y2 and y1 <= x2):
        return False
    else:
        return True

def exact_match_no_overlap(m_a,m_i):
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


    if(len(maString)!=len(miString)):
        return False
    else:
        for i in range(len(maString)):
            if(maString[i].replace(' ', '').lower()!=miString[i].replace(' ', '').lower()):
                return False
        if (x1 <= y2 and y1 <= x2):
            return True
        else:
            return False
# deliverable 2.5
def match_on_content(m_a, m_i):
    """

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if all match on all "content words" (defined by POS tag) and markables do not overlap
    :rtype: boolean

    """

    contentWord = ['NN','NNS','NNP','NNPS','PRP','PRP$','WP','WP$','CD']
    maTag = m_a['tags']
    maStart = m_a['start_token']
    maEnd = m_a['end_token']

    miTag = m_i['tags']
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

    maContent = []
    maString = []
    maCount = 0
    for i in range(len(maTag)):
        if(maTag[i].replace(' ', '') in contentWord):
            maContent.insert(maCount,maTag[i])
            maString.insert(maCount,m_a['string'][i])
            maCount+=1
    miContent = []
    miString = []
    miCount = 0
    for i in range(len(miTag)):
        if(miTag[i].replace(' ', '') in contentWord):
            miContent.insert(miCount,miTag[i])
            miString.insert(miCount,m_i['string'][i])
            miCount+=1
    if(len(maContent)!=len(miContent)):
        return False
    else:
        for i in range(len(maContent)):
            if(maString[i].replace(' ', '').lower()!=miString[i].replace(' ', '').lower()):
                return False           
        if (x1 <= y2 and y1 <= x2):
            return True
        else:
            return False

    

########## helper code

def most_recent_match(markables,matcher):
    """given a list of markables and a pairwise matcher, return an antecedent list
    assumes markables are sorted

    :param markables: list of markables
    :param matcher: function that takes two markables, returns boolean if they are compatible
    :returns: list of antecedent indices
    :rtype: list

    """
    antecedents = range(len(markables))
    for i,m_i in enumerate(markables):
        for a,m_a in enumerate(markables[:i]):
            if matcher(m_a,m_i):
                antecedents[i] = a
    return antecedents

def make_resolver(pairwise_matcher):
    """convert a pairwise markable matching function into a coreference resolution system, which generates antecedent lists

    :param pairwise_matcher: function from markable pairs to boolean
    :returns: function from markable list and word list to antecedent list
    :rtype: function

    The returned lambda expression takes a list of words and a list of markables. 
    The words are ignored here. However, this function signature is needed because
    in other cases, we want to do some NLP on the words.

    """
    return lambda markables : most_recent_match(markables,pairwise_matcher)
