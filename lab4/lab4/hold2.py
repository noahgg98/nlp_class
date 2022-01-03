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

#word vec imports
import gensim
from gensim.models import Word2Vec
from nltk.corpus import brown
m = ml()
kc = KMeansClusterer(2, distance=nltk.cluster.util.cosine_distance, repeats=25)

iterWord = list()
iterWordNew = list()
wvModel2 = ""

#create wordvector model
#will download needed corpus if not already downloaded
try:
    wvModel = gensim.models.Word2Vec(brown.sents())
    wvModel.save('brown.embedding')
    wvModel = gensim.models.Word2Vec.load('brown.embedding')

except:
    nltk.download('brown')

#for regression piece
vectorizer = CountVectorizer(analyzer='word', min_df=2, ngram_range=(1, 2))
clf = LogisticRegression(max_iter=2000, dual=False, solver='lbfgs')   

#clf        = MLPClassifier(hidden_layer_sizes=(125,), max_iter=400,activation ='relu',solver='adam',random_state=1)


#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#
def clean(text):
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

def posNeg(tweet, lex_pos, lex_neg, sum=0):

    for word in tweet.split(" "):
        if word in lex_pos:
            sum += 1       
        elif word in lex_neg:
            sum -= 1
    
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
        twet = tweet
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

##########HELPERS FOR LEARNING##########
def wordVecSent(tweetString):
    global wvModel
    global m
    global allWords

    return "positive"

    
def create_iter_wvModel():
    global wvModel
    iterList = list()
    for i, word in enumerate(wvModel.wv.index_to_key):
        iterWord.append(word)
        iterList.append(wvModel.wv[word])
    return iterList

def rebalance(cluster):
    newIter = list()
    i=0
    for word in iterWord:
        if cluster[i] < 4:
            iterWordNew.append(word)
            newIter.append(wvModel.wv[word])
        i+=1
    return newIter
        





        
        



def label_with_learning(raw_tweets, testing_tweets):
    global kc
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
    goods = list()
    bads = list()
    next = True     #to end while loop

    #create clusters
    
    clusters = kc.cluster(create_iter_wvModel(), assign_clusters=True)
    i = 0
    for word in iterWord:
        if clusters[i] == 1:
            goods.append(word)
        else:
            bads.append(word)
        i+=1
    
    

    #go through the testing data
    while(index<10000):
        rawNext = raw_tweets.get_next_tweet()
        tweetString = clean(rawNext)

        #indicates that there are no more tweets left
        if rawNext == None:
            next = False
            break

        finalsentVal = posNeg(tweetString, goods, bads)

        if finalsentVal > 0:
            #add classifier based on 
            labels.append("positive")
            tweets.append(tweetString)
        elif finalsentVal < 0:
            #add classifier based on 
            labels.append("negative")
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
        tweet = clean(testing_tweets[index])
       
    
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
        #print(prob)
        
        #determine sentiment based on difference
        if prob[1] > .5:
            guesses.append("positive")
        elif prob[0] > .4:
            guesses.append("negative")
        else:
            guesses.append("objective")
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
