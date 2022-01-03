#
# sentiment.py -lexicon|learn [-data <data-dir>]
#

import sys
import string
import re
import argparse
import data
from nltk.cluster import KMeansClusterer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier 
from ml import *
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

#word vec imports
import gensim
from gensim.models import Word2Vec
from nltk.corpus import brown
vectorizer = CountVectorizer(analyzer='word', min_df=2, ngram_range=(1, 2), stop_words='english')
clf = LogisticRegression(max_iter=3000) 
#clf = MLPClassifier(hidden_layer_sizes=(125,), max_iter=200,activation='relu',solver='adam',random_state=1)



#########WARNING#############
#ONLY PART 1 WORKS RIGHT NOW#
#########WARNING#############

#clf        = MLPClassifier(hidden_layer_sizes=(125,), max_iter=400,activation ='relu',solver='adam',random_state=1)


#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#
def clean_better(text):
    #to remove all non characters
    text = text.encode("ascii","ignore")
    text = text.decode()

    #to remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))
    
    #replace links and numbers
    text = re.sub(r'[A-Za-z]*[//]*tco[//]*[A-Za-z]*', '', text)
    text = re.sub(r'[A-Za-z]*[//]*http[//]*[A-Za-z]*', '', text)
    text = re.sub(r'[0-9]*', '', text)
    
    
    return(text.lower())

def clean(text):
    try:
        #split the text 
        newText = text.split(" ")

        #replace certain chars
        newText = [word.replace("!","").replace(".","").replace("\n", "").replace("@","").lower() for word in newText]

        #remove any empty strings and return 
        newText.remove('') if '' in newText else None

        for word in newText:
            if not isenglish(word):
                newText.remove(word)
    except:
        newText = text
    return newText

def posNeg(tweet, lex_pos, lex_neg, sum=0):

    for word in tweet:
        if word in lex_pos:
            sum += 1.1       
        elif word in lex_neg:
            sum -= .8
        

    return sum

    

def label_with_lexicon(lexicon_positives, lexicon_negatives, testing_tweets):

    '''
    lexicon_positives: a list of strings, positive words
    lexicon_negatives: a list of strings, negative words
    testing_tweets: a List of tweets that this function must predict sentiment
    Returns: a List of predicted sentiment labels ('positive','negative','objective')
             --> equal to the length of testing_tweets
    '''

    #define return string
    ret = list()

    for tweet in testing_tweets:
        
        #clean tweet and label
        tweet = clean(tweet)
        
        finalsentVal = posNeg(tweet, lexicon_positives, lexicon_negatives)

        if finalsentVal > 0:
            ret.append('positive')
        elif finalsentVal < 0:
            
            ret.append('negative')
        else:
            
            ret.append('objective')
    return ret
def classify(tweet):
    oldtwt = tweet
    tweet = clean_better(tweet)
    tweet_2 = clean(tweet)
    sum=0

    #keep and old metric
    if ":)" in oldtwt.split(" "):
        sum+=1
    elif ":(" in oldtwt.split(" "):
        sum-=1

    #create small list of words to compare against
    good = [
        lemma.lemmatize("happy"),lemma.lemmatize("excited"),lemma.lemmatize("love"),lemma.lemmatize("good"),\
        lemma.lemmatize("positive"),lemma.lemmatize("like"),lemma.lemmatize("beautiful"), lemma.lemmatize("pretty"), \
        lemma.lemmatize("nice"),lemma.lemmatize("laugh"), lemma.lemmatize("live"),lemma.lemmatize("positive"),\
        lemma.lemmatize("grace"), lemma.lemmatize("peace"), lemma.lemmatize("hug")
        ]

    bad = [
        lemma.lemmatize("bad"), lemma.lemmatize("ugly"),lemma.lemmatize("kill"),lemma.lemmatize("cry"),\
        lemma.lemmatize("bored"),lemma.lemmatize("hate"),lemma.lemmatize("destroy"),lemma.lemmatize("scream"), \
        lemma.lemmatize("terrible"), lemma.lemmatize("fear"),lemma.lemmatize("gross"),lemma.lemmatize("beat"),\
        lemma.lemmatize("negative"), lemma.lemmatize("fuck"),lemma.lemmatize("disgrace"),lemma.lemmatize("shit")
        ]

    #loop through lemmas
    
    for word in tweet.split(" "):
        word = lemma.lemmatize(word.lower())

        if word in good:
            sum+=1
        elif word in bad:
            sum-=1

    for word in tweet_2:
        if word in good and word not in tweet.split(" "):
            sum+=1
        if word in bad and word not in tweet.split(" "):
            sum-=1

        
    return sum

def label_with_learning(raw_tweets, testing_tweets):
    '''
    raw_tweets: a Tweets object with function next_tweet() to retrieve tweets one at a time
    testing_tweets: a List of tweets that this function must predict sentiment
    Returns: a List of predicted sentiment labels ('positive','negative','objective')
             --> equal to the length of testing_tweets
    '''

    """Train"""
    index = 0       #keep track of size
    labels = list() #define labels
    tweets = list() #define tweets
    next = True     #to end while loop

    #go through the testing data
    while(next):
        rawNext = raw_tweets.get_next_tweet()
        #print(index)

        #indicates that there are no more tweets left
        if rawNext == None:
            next = False
            break

        
        #used for filling x_counts  cleaned tweet
        tweetString = clean_better(rawNext)
        #tweetString = " ".join(tweet)
        
        ###############Classifier Piece###############
        #add classifier based on 
        sum = classify(rawNext)
        if sum>0:
            labels.append('positive')
            tweets.append(tweetString)
        elif sum<0:
            labels.append('negative')
            tweets.append(tweetString)

        
        ###############Classifier Piece###############
        
        index+=1

    X_train_counts = vectorizer.fit_transform(tweets)     # X_train is a list of strings
    clf.fit(X_train_counts, labels)

    
    """End Train"""

    """Test"""
    size=index
    tweets = list()
    index=0
    while(index< len(testing_tweets)):
        
    

        #used for filling x_counts  cleaned tweet
        tweet = clean_better(testing_tweets[index])
        #tweet = " ".join(tweet)
    
        tweets.append(tweet)
        index+=1
        
        
    X_test_counts = vectorizer.transform(tweets)   # NOT fit_transform(), just transform()
    

    """
    for i in guesses:
        print(i)
    """
    guesses = list()
    x = clf.predict_proba(X_test_counts)

    for prob in x:

 
        #determine sentiment based on difference
        if(prob[1]>.85):
            guesses.append('positive')
        elif(prob[0]>.27):
            guesses.append('negative')
        else:
            guesses.append('objective')
        
    """End Test"""


    return guesses


#
# PROVIDED FOR YOU
#

def isenglish(tweet):
    # Python 3.7 - fast!
    return tweet.isascii()

    # Python 3.6 - twice as slow...
    #return all(ord(c) < 128 for c in s)



#
# DO NOT CHANGE ANYTHING BELOW THIS LINE.
#
if __name__ == '__main__':

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="author.py")
    parser.add_argument('-data', action="store", dest="data", type=str, default='data', help='Directory containing the books')
    parser.add_argument('-lexicon', action="store_true", default=False, help='Use the lexicon')
    parser.add_argument('-learn', action="store_true", default=False, help='Train a classifier')    
    args = parser.parse_args()
    
    tweets = data.Tweets(args.data)
    testing_tweets = [x[1] for x in tweets.get_labeled_tweets()]
    testing_labels = [x[0] for x in tweets.get_labeled_tweets()]
    
    if args.lexicon and not args.learn:
        lexicon = tweets.get_lexicon()
        predicted_labels = label_with_lexicon(lexicon[0], lexicon[1], testing_tweets)
    elif not args.lexicon and args.learn:
        predicted_labels = label_with_learning(tweets, testing_tweets)
    else:
        print('Must choose one of -lexicon or -learn')
        sys.exit(0)

        
    # EVALUATE
    accuracy = data.evaluate(predicted_labels, testing_labels)
    print('Final Accuracy: %.2f%%\n\n' % (accuracy))
