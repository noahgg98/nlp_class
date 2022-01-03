#
# sentiment.py -lexicon|learn [-data <data-dir>]
#

import sys
import string
import re
import argparse
import data
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
def wordVecSent(tweetString, rawtweet):
    global wvModel
    global m
    sum = 0

    if ":(" in rawtweet:
        sum-=1.5
    if ":)" in rawtweet:
        sum+=1.5

    #get word vectors for good and bad words
    bword_1 = "horrible"
    bword_2 = "terrible"
    bword_3 = "cry"
    bword_4 = "sad"
    bword_5 = "scared"
    bwords = [bword_1,bword_2,bword_3,bword_4,bword_5]

    gword_1 = "happy"
    gword_2 = "excite"
    gword_3 = "laugh"
    gword_4 = "enjoy"
    gword_5 = "good"
    gwords = [gword_1,gword_2,gword_3,gword_4,gword_5]

    #combine both lists
    words = [bword_1,bword_2,bword_3,bword_4,bword_5, gword_1,gword_2,gword_3,gword_4,gword_5]
    
    sum = 0
    bestComp = 0
    bestW = ''
    for word in tweetString.split(" "):
        for w in words:
            try:
                comp = wvModel.wv.similarity(word, w)
                if comp > bestComp:
                    bestComp = comp
                    bestW = w
            except:
                bestW = bestW

  
        if bestW in gwords:
            sum+=bestComp
        elif bestW in bwords:
            sum-=bestComp
        else:
            sum+=0
            
   
    #return based on sum value
    
    if sum>0:
        return 'positive'
    elif sum<0:
        return 'negative'
    else:
        return 'objective'
        
        
        



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
    while(index<10000):
        rawNext = raw_tweets.get_next_tweet()

        #indicates that there are no more tweets left
        if rawNext == None:
            next = False
            break

        
        #used for filling x_counts  cleaned tweet
        tweetString = clean(rawNext)
        
        
        
        ###############Classifier Piece###############
        #add classifier based on 
        labels.append(wordVecSent(tweetString, rawNext))
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
        if prob[2] > .75:
            guesses.append("positive")
        elif prob[0] > .65:
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
