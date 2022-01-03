#
# logit-author.py -data <data-dir> [-test]
#

import argparse
import data
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model._ridge import RidgeClassifierCV



# Ignore 'future warnings' from the toolkit.
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#alg models to be filled in train
vectorizer = None
clf        = None
vectorizer = TfidfVectorizer(analyzer='word', min_df=5, ngram_range=(1, 2))
clf        = MLPClassifier(hidden_layer_sizes=(125,), max_iter=300,activation ='relu',solver='adam',random_state=1)

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

    # Convert your text with the vectorizer.
    X_train_counts = vectorizer.fit_transform(p_fit)     # X_train is a list of strings
        
            
    clf.fit(X_train_counts, auths)                                     # trains the model using gradient descent

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
