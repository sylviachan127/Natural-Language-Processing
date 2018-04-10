# Yuen Han Chan
# 3.1 (0.5 points)

Fill in the rest of the table below:

|      | they | can | can | fish | END |
|------|------|-----|-----|------|-----|
| Noun | -2   | -10 | -10 | -15  | n/a |
| Verb | -13  | -6  | -11 | -16  | n/a |
| End  | n/a  | n/a | n/a | n/a  | -17 |


# 4.3 (0.5 points)

Do you think the predicted tags "PRON AUX AUX NOUN" for the sentence "They can can fish" are correct? Use your understanding of parts-of-speech from the notes.

I think the predicted tags is not correct, and the correct tag should be "PRON AUX VERB NOUN".  Since “They can can fish" mean the ability to put fish into can, the fish is the subject can the second can is the main action.  Since the first can is a verb that provoides supported fuction for the main verb "the second can", it can remains as an AUX.  The word "can" can essentilly takes two forms, if it is a verb, then it must be followed by a Noun, and if it is a AUX, it must be follow by a verb.

# 4.4 (0.5 points)

The HMM weights include a weight of zero for the emission of unseen words. Please explain:

- why this is a violation of the HMM probability model explained in the notes;
- How, if at all, this will affect the overall tagging.

If we give a weight of 0 as emission value for unseen words, then HMM will not works as intended since all state sequences will have 0 probabilities for the input sequences.  That where the smoothing value come from, so unseen words can still get a small non-zero probabilities, and avoid the zero probabilites problem.

# 5.1 (1 point 4650; 0.5 points 7650)

Please list the top three tags that follow verbs and nouns in English and Japanese.

English_VERB: DET, ADP, PRON
JAPANESE_VERB: NOUN, --END--, PUNCT

ENGLISH_NOUN: PUNCT, ADP, NOUN
JAPANESE_NOUN: NOUN,VERB, PUNCT

Try to explain some of the differences that you observe, making at least two distinct points about differences between Japanese and English.

In a simpler word, the English sentence structure is arrange as Subject-Verb-Object (SVO), while Japanese sentence structure is arrange as Subject-Object-Verb (SOV).  So in above, we can see --END-- and PUNCT is a common tag following Japanese Verb.  As an example: if "Sylvia eats a burger" is a sentence in English, then it will restructure as "Sylvia a burger eat" in Japanese.  Since we often has determinator and adposition comes before a noun in English, we see that DET and ADP are common tag following verb.

While English has pre-position such as "went to China", Japanese has post-position, which they are switch.  So it will become "China to went" in Japnese.  That that why we don't see DET and ADP as common tag following Japense_Verb. 

In addition, Japanaese is a head final language, while English is a head-inital language.  So in English, a setnece such as "The yummy ice-cream that sells at every summer", can translate to "every summer sells the yummy ice-cream", resulting we don't know what is being refer to until the very end at the setnence.  That explains why PUNCT and NOUN are common tag that follows Japnese and English NOUN, but in different place of the setnence.  One is place in the inital of setnence while flipped for Japanese/

