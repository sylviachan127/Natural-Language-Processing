# Deliverable 1.3

Why do you think the type-token ratio is lower for the dev data as compared to the training data?

(Yes the dev set is smaller; why does this impact the type-token ratio?)

Because the dev data contains less words than words in training data.  Thus this result in smaller numinator, which impact the type-token ratio.  It also make sense that there's more word in the training data, thus the result would be more "well-rounded" to include most words that actually occurs in real world setting.


# Deliverable 3.5

Explain what you see in the scatter plot of weights across different smoothing values.

The smoothing value is there to serve as a "safe-bound", so all unknown words (word that is not encounter in training, or not exist in particular type) would have a small non-zero value as probilities to occur later instead of giving them just a value of zero.  From the plot, we can see as the smoothing values increase, the weight for that word also increase.  We do not want a smoothing values too high because it would assign high weight to words that we never encounter.  In another word, as the smoothing values increase, the importance of that words decreases, and that is not desirable.  

# Deliverable 6.2

Now compare the top 5 features for logistic regression under the largest regularizer and the smallest regularizer.
Paste the output into ```text_answers.md```, and explain the difference. (.4/.2 points)


# Deliverable 7.2

Explain the new preprocessing that you designed: why you thought it would help, and whether it did.

# Deliverable 8

Describe the research paper that you have chosen.

- What are the labels, and how were they obtained?
- Why is it interesting/useful to predict these labels?  
- What classifier(s) do they use, and the reasons behind their choice? Do they use linear classifiers like the ones in this problem set?
- What features do they use? Explain any features outside the bag-of-words model, and why they used them.
- What is the conclusion of the paper? Do they compare between classifiers, between feature sets, or on some other dimension? 
- Give a one-sentence summary of the message that they are trying to leave for the reader.
