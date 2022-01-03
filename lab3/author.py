#
# author.py -data <data-dir> [-test]
#Noah Garcia-Galan and Paul Hendron


import argparse
import data
from SmoothedBigram import SmoothedBigram
import math

#define globals
bmModels = list()
authorPassages = list()
allAuths = list()




#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#
def clean(text):
    #split the text 
    newText = text[1].split(" ")

    #replace certain chars
    newText = [word.replace("!","").replace(".","").replace("\n", "").lower() for word in newText]

    #remove any empty strings and return 
    newText.remove('') if '' in newText else None
    
    return newText


    '''
    Given a list of passages and their known authors, train your learning model.
    passages: a List of passage pairs (author,text) 
    Returns: void
    '''
def train(passages):
    global bmModels
    global authorPassages
    global allAuths

    #set current author
    currAuthor = ""
    
    for passage in passages:
        
        #account for changes in authors
        if passage[0] != currAuthor:
            

            #set new bigram model
            bm = SmoothedBigram()
            bmModels.append(bm)

            #set new author
            currAuthor = passage[0]
            allAuths.append(currAuthor)

            #set new author pasage list
            newAuth = list()
            authorPassages.append(newAuth)
            

        cleanP = clean(passage)
        authorPassages[len(authorPassages)-1].append(cleanP)
    
    #train each model one specific author
    for i in range(0, len(authorPassages)):
        bmModels[i].train(authorPassages[i])
        
   
    
  


def test(passages):
    global bmModels
    global allAuths
    '''
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    '''

    #go through each passage in list
    bestProb = 0
    finalList = list()
    for passage in passages:

        #setup variable to be used in comparisons
        author = ''
        cleanP = clean(passage)
        index = 0
        bestProb = -99*99*99    #set to something that will get beaten 

        for bm in bmModels:
            
            #add up the probabilities
            overallProb = 0 
            
            for i in range(0, len(cleanP)):
                overallProb += math.log2(bm.get_word_probability(cleanP, i))
            
                
            #replace highest prob 
            if overallProb > bestProb:
                bestProb = overallProb
                author = allAuths[index]

            index+=1
        finalList.append(author)
   
   
    return finalList




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

    # TRAIN
    train(passages.get_training())

    # TEST
    if args.test:  testset = passages.get_test()
    else:          testset = passages.get_development()
    predicted_labels = test(testset)
    
    # EVALUATE
    accuracy = data.evaluate(predicted_labels, testset)
    print('Final Accuracy: %.2f%%\n\n' % (accuracy))
