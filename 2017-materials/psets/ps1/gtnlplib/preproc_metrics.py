def get_token_type_ratio(vocabulary):
    """compute the ratio of tokens to types

    :param vocabulary: a Counter of words and their frequencies
    :returns: ratio of tokens to types
    :rtype: float

    """
    
    typeSum = 0.0;
    sumWord = 0.0;
    vocabSum = 0.0;
    tokenSum = 0.0;
    for v in vocabulary:
        tokenSum +=1;
        for key in v:
            sumWord+=vocabulary[key];
            typeSum+=1;
    #print "tokenSum",tokenSum; #27707.0
    #print "typeSum",typeSum; #245535.0
    #print "sumWord", sumWord; #457112657.0
    #print "sum(vocab.iter())", sum(vocabulary.itervalues()); #545027
    return float(sum(vocabulary.itervalues()))/float(tokenSum);
    
def type_frequency(vocabulary, k):
    """compute the number of words that occur exactly k times

    :param vocabulary: a Counter of words and their frequencies
    :param k: desired frequency
    :returns: number of words appearing k times
    :rtype: int

    """
    #raise NotImplementedError
    exactlyK = 0;
    for v in vocabulary:
        if(vocabulary[v]==k):
            exactlyK+=1;
    return exactlyK;
    
def unseen_types(first_vocab, second_vocab):
    """compute the number of words that appear in the second vocab but not in the first vocab

    :param first_vocab: a Counter of words and their frequencies in one dataset
    :param second_vocab: a Counter of words and their frequencies in another dataset
    :returns: number of words that appear in the second dataset but not  in the first dataset
    :rtype: int

    """
    #raise NotImplementedError
    notAppear = 0;
    for v in second_vocab:
        if(first_vocab[v]==0 or first_vocab[v]==None):
            notAppear+=1;
    return notAppear;
