################### NLTK Imports #########################################
from nltk.corpus import twitter_samples, stopwords
#importing nltk's POS tagger
from nltk.tag import pos_tag
#importing WordNetLemmatizer (lematization - analyzed the word structure and its context to convert into normalize form)
from nltk.stem.wordnet import WordNetLemmatizer
#importing nltk's freq_dist to find the most common words and classifiers for the training model
from nltk import FreqDist, classify, NaiveBayesClassifier
#importing word tokenizer for tokenizinf custom sentences
from nltk.tokenize import word_tokenize
#importing confusion matrix from nltk
from nltk.metrics import ConfusionMatrix



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

import re
import string
import random

#importing nltk sklearn library for naive bayes
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pickle
from nltk.metrics.scores import (precision, recall)
import collections
################### NLTK Imports #########################################

#####################  Remove Noise  ######################################
def remove_noise(tweet_tokens, stopwords = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|''(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stopwords:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
#####################  Remove Noise  ######################################

#################### Get all Words ####################################
def get_all_words(cleaned_tokens):
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token
#################### Get all Words ####################################

######################### Get Tweets for model #######################
def get_tweets_for_model(cleaned_tokens):
    for tweet_tokens in cleaned_tokens:
        yield dict([token, True] for token in tweet_tokens)
######################### Get Tweets for model #######################


###################### Main Function ###################################

if __name__ == "__main__":
    ##################### Getting Data Samples #######################
    positive_tweets = twitter_samples.strings('positive_tweets.json') #tweets with positive sentiments.
    negative_tweets = twitter_samples.strings('negative_tweets.json') #tweets with negative sentiments.
    text = twitter_samples.strings('tweets.20150430-223406.json') #tweets with no sentiments.
    #print((text[0]))
    ################### Getting Data Samples #########################

    ################### Tokenised Data Samples #############################
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')#[0]
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')#[0]
    #print(positive_tweet_tokens[0])
    ################### Tokenised Data Samples #############################

    ##################### Normalising Data ################################
    stopwords = stopwords.words("english")
    #print(stopwords)
    positive_cleaned_tokens = []
    negative_cleaned_tokens = []
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens.append(remove_noise(tokens,stopwords))
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens.append(remove_noise(tokens, stopwords))
    ##################### Normalising Data ################################

    ################### Getting Words Frequency #######################
    all_positive_words =    get_all_words(positive_cleaned_tokens)  
    frequently_distributed_positive_words = FreqDist(all_positive_words)
    freq_positive_words =  frequently_distributed_positive_words.most_common(25)
    #print(freq_positive_words[0])
    all_negative_words = get_all_words(negative_cleaned_tokens)
    frequently_distributed_negative_words = FreqDist(all_negative_words)
    freq_negative_words =  frequently_distributed_negative_words.most_common(25)
    #print(freq_negative_words[0])
    ################### Getting Words Frequency #######################

    ################### Plotting Graphs #######################
    ############# Positive WordCLoud ########################
    pos_tok_words = ' '.join(list(zip(*frequently_distributed_positive_words.most_common(100)))[0])
    wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=110).generate(pos_tok_words)

    plt.figure('Positive WordCloud', figsize=(10,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    ############# Positive WordCLoud ########################

    ############# Negative WordCLoud ########################
    neg_tok_words = ' '.join(list(zip(*frequently_distributed_negative_words.most_common(100)))[0])
    wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=110).generate(neg_tok_words)

    plt.figure('Negative WordCloud', figsize=(10,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    ############# Negative WordCLoud ########################
    
    ############# Positive bar graph ########################
    p_words = list(zip(*freq_positive_words))[0]
    p_count = list(zip(*freq_positive_words))[1]
    p_x_pos = np.arange(len(p_words))

    plt.figure('Positive BarGraph')
    plt.bar(p_x_pos,p_count,align='center')
    plt.xticks(p_x_pos,p_words)
    plt.ylabel('Count')
    ############# Positive bar graph ######################

    ############ Negative bar graph #######################
    n_words = list(zip(*freq_negative_words))[0]
    n_count = list(zip(*freq_negative_words))[1]
    n_x_pos = np.arange(len(n_words))

    plt.figure('Negative BarGraph')
    plt.bar(n_x_pos,n_count,align='center')
    plt.xticks(n_x_pos,n_words)
    plt.ylabel('Count')
    plt.show()
    ############ Negative bar graph #######################
    ################### Plotting Graphs #######################

    #######################   Making the TRAINING AND TESTING SET  ##################
    positive_model_tokens = get_tweets_for_model(positive_cleaned_tokens)
    negative_model_tokens = get_tweets_for_model(negative_cleaned_tokens)

    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_model_tokens]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_model_tokens]

    dataset = positive_dataset + negative_dataset
    print(len(dataset))
    random.shuffle(dataset)
    training_dataset = dataset[:7000]
    testing_dataset = dataset[7000:]
    print(training_dataset[0])
    #######################   Making the TRAINING AND TESTING SET  ##################

    ############################ Building Our classifier Class ####################
    class VoteClassifier(ClassifierI):
        def __init__(self,*classifiers):
            self.__classifiers = classifiers
        
        def classify(self, features):
            votes = []
            for c in self.__classifiers:
                v = c.classify(features)
                votes.append(v)
            return mode(votes)
        
        def confidence_measure(self,features):
            votes = []
            for c in self.__classifiers:
                v = c.classify(features)
                votes.append(v)
            
            choice_votes = votes.count(mode(votes))
            confidence = choice_votes / len(votes)
            return confidence
    ############################ Building Our classifier Class ####################
    
    #######################   TRAINING the Model  ################## 
    NB_classifier = NaiveBayesClassifier.train(training_dataset)
    save_classifier = open("original_NB.pickle","wb")
    pickle.dump(NB_classifier,save_classifier)
    save_classifier.close()

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_dataset)
    save_classifier = open("MNB.pickle","wb")
    pickle.dump(MNB_classifier,save_classifier)
    save_classifier.close()

    LR_classifier = SklearnClassifier(LogisticRegression())
    LR_classifier.train(training_dataset)
    save_classifier = open("LR.pickle","wb")
    pickle.dump(LR_classifier,save_classifier)
    save_classifier.close()

    LSVC_classifier = SklearnClassifier(LinearSVC())
    LSVC_classifier.train(training_dataset) 
    save_classifier = open("LSVCB.pickle","wb")
    pickle.dump(LSVC_classifier,save_classifier)
    save_classifier.close() 
    #######################   TRAINING the Model  ##################

    #######################   TESTING the Model  ##################
    #conf_matrix = ConfusionMatrix(training_dataset,testing_dataset)
    #print(conf_matrix)
    print("Accuracy of the model using Naive Bayes:", classify.accuracy(NB_classifier, testing_dataset))

    print("Accuracy of the model using Multinomial Naive Bayes:", classify.accuracy(MNB_classifier, testing_dataset))

    print("Accuracy of the model using Logistic Regression:", classify.accuracy(LR_classifier, testing_dataset))

    print("Accuracy of the model using Linear Support Vector Classifer:", classify.accuracy(LSVC_classifier, testing_dataset))


    refsets_NB = collections.defaultdict(set)
    testsets_NB = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testing_dataset):
        refsets_NB[label].add(i)
        observed = NB_classifier.classify(feats)
        testsets_NB[observed].add(i)
    print( 'Precision:', precision(refsets_NB['Positive'], testsets_NB['Positive'])*100)
    print( 'Recall:', recall(refsets_NB['Positive'], testsets_NB['Positive'])*100)

    refsets_MNB = collections.defaultdict(set)
    testsets_MNB = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testing_dataset):
        refsets_MNB[label].add(i)
        observed = MNB_classifier.classify(feats)
        testsets_MNB[observed].add(i)
    print( 'Precision:', precision(refsets_MNB['Positive'], testsets_MNB['Positive'])*100)
    print( 'Recall:', recall(refsets_MNB['Positive'], testsets_MNB['Positive'])*100)

    refsets_LR = collections.defaultdict(set)
    testsets_LR = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testing_dataset):
        refsets_LR[label].add(i)
        observed = LR_classifier.classify(feats)
        testsets_LR[observed].add(i)
    print( 'Precision:', precision(refsets_LR['Positive'], testsets_LR['Positive'])*100)
    print( 'Recall:', recall(refsets_LR['Positive'], testsets_LR['Positive'])*100)

    refsets_LSVC = collections.defaultdict(set)
    testsets_LSVC = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testing_dataset):
        refsets_LSVC[label].add(i)
        observed = LSVC_classifier.classify(feats)
        testsets_LSVC[observed].add(i)
    print( 'Precision:', precision(refsets_LSVC['Positive'], testsets_LSVC['Positive'])*100)
    print( 'Recall:', recall(refsets_LSVC['Positive'], testsets_LSVC['Positive'])*100)
    print(NB_classifier.show_most_informative_features(10))
    #######################   TESTING the Model  ################## 

    #################### Voting Classifier accuracy and Confidence Check ##########################
    voting_classifier = VoteClassifier(NB_classifier,MNB_classifier,LR_classifier,LSVC_classifier)
    print("Voting Classifier accuracy :",(classify.accuracy(voting_classifier, testing_dataset))*100)
    print("Classification:", voting_classifier.classify(testing_dataset[4][0]), "Confidence %:",voting_classifier.confidence_measure(testing_dataset[4][0])*100)
    #################### Voting Classifier accuracy and Confidence Check ##########################

    ####################### Giving our own input ################################
    my_tweet = "very good customer services is provided by your agency!"
    custom_token = remove_noise(word_tokenize(my_tweet))
    print("Prediction of myTweet : ", voting_classifier.classify(dict([token, True] for token in custom_token))) 
    print("Confidence % :",voting_classifier.confidence_measure(dict([token, True] for token in custom_token))*100)
    ####################### Giving our own input ################################
###################### Main Function ###################################3