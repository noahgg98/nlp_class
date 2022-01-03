#
# logit-author.py -data <data-dir> [-test]
#

import argparse
import data
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import gensim
import gensim.downloader


# Ignore 'future warnings' from the toolkit.
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#alg models to be filled in train
vectorizer = None
clf        = None
wvmodel = gensim.downloader.load('glove-wiki-gigaword-100')

#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#
def train(passages):
    
    global vectorizer
    global clf

    '''
    Given a list of passages and their known authors, train your Logistic Regression classifier.
    passages: a List of passage pairs (author,text) 
    Returns: void
    '''
    p_fit = list()
    auths = list()
    
    for passage in passages:
        p_fit.append(passage[1])   #turn into list of words
        auths.append(passage[0])   #turn into author list

    # need to define x_train_counts
    # Y_train is just the author
    # Create a "vectorizer" object.	
    vectorizer = CountVectorizer(analyzer='word', min_df=5, ngram_range=(1, 2))   # 1 and 2-grams, min 5 count

    # Convert your text with the vectorizer.
    X_train_counts = vectorizer.fit_transform(p_fit)     # X_train is a list of strings
        

    clf = LogisticRegression(solver='lbfgs', max_iter=10000)            # creates a Logistic Regression object
    clf.fit(X_train_counts, auths)                                      # trains the model using gradient descent
    
def wordAvg(list_of_words, size=300):
    global wvmodel

    #get list of words and add their vectors to list
    allWordVecs = []
    allWordVecs = [wvmodel[word] for word in list_of_words] 

    #get average of words
    avgOfWords = [0]*size

    #define lambda to add
    add = lambda wv, avgOfWords: [(total+num) for total, num in zip(avgOfWords, wv)]

    wvLen = len(allWordVecs)

    for i in range(0, wvLen):

        #update array and get new avg sum
        avgOfWords = add(allWordVecs[i], avgOfWords)

    avgOfWords = [avgOfWords[i]/wvLen for i in range(0, wvLen)]

    print(avgOfWords)

def test(passages):
    global vectorizer
    global clf
    '''
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    '''

    p_fit = list()
    for passage in passages:
        p_fit.append(passage[1])   #turn into list of words
        
    
    
    X_test_counts = vectorizer.transform(p_fit)   # NOT fit_transform(), just transform()

    guesses = clf.predict(X_test_counts)


    return guesses


    

#
# DO NOT CHANGE ANYTHING BELOW THIS LINE.
#
if __name__ == '__main__':

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="author.py")
    parser.add_argument('-data', action="store", dest="data", type=str, default='data', help='Directory containing the books')
    parser.add_argument('-test', action="store", type=bool, default=False, help='Use the test set not dev')
    args = parser.parse_args()
    
    passages = data.Passages(args.data)
    trainset = passages.get_training()
    if args.test:  testset = passages.get_test()
    else:          testset = passages.get_development()

    # TRAIN
    train(trainset)

    # TEST
    predicted_labels = test(testset)
    
    # EVALUATE
    accuracy = data.evaluate(predicted_labels, testset)
    print('Accuracy: %.2f%%\n' % (accuracy))
