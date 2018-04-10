import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import OFFSET, START_TAG, END_TAG

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tag_word_counts(filename):
    """build a dict of counters, one per tag, counting the words that go with each tag

    :param trainfile: training data
    :returns: dict of counters
    :rtype: dict

    """
    all_counters = defaultdict(lambda : Counter())
    for words, tags in enumerate(conll_seq_generator(filename)):
        word_list = tags[0]
        tag_list = tags[1]
        for x in range(len(word_list)):
            currentWord = word_list[x]
            currentTag = tag_list[x]
            currentTagSet = all_counters[currentTag]
            currentTagSet[currentWord] += 1
            all_counters[currentTag] = currentTagSet

    return all_counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun

    :returns: simple weight dictionary

    """
    weights = defaultdict(float)
    weights[('NOUN'),OFFSET] = 1.
    print weights
    return weights

def get_most_common_word_weights(trainfile):
    """Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training file, the weights should be set so that the output tag is Noun.

    :param trainfile: training file
    :returns: classification weights
    :rtype: defaultdict

    """
    # # trainfile = trainfile[:3]
    # # weights = defaultdict(float)
    weights = get_noun_weights()
    print weights
    dict_word = get_tag_word_counts(trainfile)
    for types in dict_word:
        # print types
        for x in dict_word[types]:
            weights[types,x] = dict_word[types][x]
            # print x
            # print "count: ", dict_word[types][x]
    # print dict_word

    # wordweight = defaultdict(lambda: defaultdict(float))
    # # nested = wordweight["hi"]
    # # nested["Noun"]=2
    # # nested["verb"]=5
    # # nested["verb2"]=4
    # # if(len(nested)!=0):
    # #     print "max", argmax(nested)
    # # print "hi"
    # # print wordweight
    # # for words, tags in enumerate(conll_seq_generator(trainfile)):
    # #     word_list = tags[0]
    # #     tag_list = tags[1]
    # #     for x in range(len(word_list)):
    # #         currentWord = word_list[x]
    # #         currentTag = tag_list[x]

    # #         nestedWeight = wordweight[currentWord]
    # #         nestedWeight[currentTag]+=1
    # #         mostFreqentTag = argmax(nestedWeight)
    # #         weights[currentWord] = mostFreqentTag
    # for words, tags in enumerate(conll_seq_generator(trainfile)):
    #     word_list = tags[0]
    #     tag_list = tags[1]
    #     for x in range(len(word_list)):
    #         currentWord = word_list[x]
    #         currentTag = tag_list[x]

    #         nestedWeight = wordweight[currentWord]
    #         nestedWeight[currentTag]+=1
    #         mostFreqentTag = argmax(nestedWeight)
    #         weights[(currentTag,currentWord)] = nestedWeight[currentTag]
    # # print weights
    # weights = get_noun_weights()
    return weights

def get_tag_trans_counts(trainfile):
    """compute a dict of counters for tag transitions

    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict

    """
    tot_counts = defaultdict(lambda : Counter())
    for _,tags in conll_seq_generator(trainfile):
        tags = [START_TAG] + tags + [END_TAG]
        tag_trans = zip(tags[:-1],tags[1:])
        for prev_tag, curr_tag in tag_trans:
            tot_counts[prev_tag][curr_tag] += 1

    return dict(tot_counts)
