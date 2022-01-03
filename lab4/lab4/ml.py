from collections import Counter
import re
import itertools
import statistics
import contractions
import ngram
from nltk.util import pr
from scipy import spatial
import numpy
import nltk
from nltk.stem import WordNetLemmatizer
from ngram import NGram
from nltk.corpus import stopwords, wordnet
import gensim.downloader
from gensim.models import KeyedVectors, Word2Vec

#need to download 
#nltk.download('wordnet')


class ml:

    def __init__(self):
        self.WVMODEL = gensim.downloader.load('glove-wiki-gigaword-100')

    #############################
    ##########_HELPERS_##########
    #############################
    """
    This function set provides all the helper functions
    These functions are used to clean and extract/manipulate data
    No actual analysis occurs here
    Includes the following:
        - Cleaner
        - Taggers
        - Tokenizers
    """

    """
    cleans the text to implement easier control
    -scrubs all stop words
    -tokenization
    -removes special chars
    -replaces cotractions
    -makes all text lowercase
    """
    def cTxt(self, text):


        #change all words to lower case
        text = text.lower()

        #split text by word and create empty list for cleaned text
        txt = text.split()
        

        #changes contractions to full words
        expTxt = []

        #expand all contractions in sentence
        [expTxt.append(contractions.fix(word)) for word in txt]

        #gets stop words to be removed
        #downloads stop words from nltk
        stopW = set(stopwords.words("english"))
        cleanTxt = []
       
        #removes stop words from the text
        for word in expTxt:
            if word not in stopW:
                cleanTxt.append(word)
      
        #turn back into sentence
        cleanTxt = ' '.join(cleanTxt)
       
        #tag the sentence
        cleanTxtTags = self.tagger(cleanTxt,0,1)
                
        #lemmatization piece
        #creates sentence of lemmatized words in clean text
        #uses wordnet return tags
        cleanTxt = ' '.join(WordNetLemmatizer().lemmatize(\
                      word[0], self.wordNetRet(word[1])) \
                      for word in cleanTxtTags)

        
        return cleanTxt

    """
    Used to return the word net lemma tag needed
    @params:
        - tag: pos_tag given to word
    @return: proper wordnet tag
    """
    def wordNetRet(self, posTag):
        #series of if else based on posTag
        if posTag == "NN":
            return wordnet.NOUN
        elif posTag[0] == "V":
            return wordnet.VERB
        elif posTag == "JJ":
            return wordnet.ADJ 
        elif posTag[0] == "R":
            return wordnet.ADV
        else:
            return wordnet.NOUN
    

    """
    takes a text input and tokenizes it by word
    can take a list and handle each value in list

    @params: 
        - text: list or phrase to be tokenized
    @ return: tokenized phrase by word
    """
    def wordTokenizer(self, text):
        txt = ""
        #convert to a string if needed
        if type(text) != type("string"):
            for i in text:
                txt = txt + i 
        else:
            txt = text

        #use nltk library to tokenize word
        wordTokens = nltk.word_tokenize(txt) 

        return wordTokens
    
    """
    takes a text input and tokenizes it by word
    can take a list and handle each value in list

    @params: 
        - text: phrase to be tokenized
    @ return: tokenized phrase by sentence
    """
    def sentenceTokenizer(self, sentence):

        #tokenize by sentence
        sentTokens = nltk.sent_tokenize(sentence)

        return sentTokens


    """
    Tags text in a given sentence
    @params: 
        - Text      : text to be tokenized (string)
        - tokenized : if sentence has been tokenized (bool) / optional
        - word      : if word tokenized (bool) / optional
    @return: tagged text
    """
    def tagger(self, text, tokenized=0, word=0):

        if not tokenized:
            if word:
                toBeTagged = self.wordTokenizer(text)
            else:
                toBeTagged = self.sentenceTokenizer(text) 
        else:
            toBeTagged     = text
        
        #tag the words
        tagged = nltk.pos_tag(toBeTagged)
        
        return tagged

    """
    Gets the average of a list of word vectors
    Used for sentance word vector average
    @params: 
        - wordVecs  : list of word vectors from a sentence
        - size      : size of the arrays, preset to 100 (optional)

    @return: single average word vector
    """
    def sent_avg_iter(self, wordVecs, size=300):

        #init array to all zeros
        avgOfWordVecs = [0] * size

        #anon function to aid in addition later
        add = lambda wv, avgOfWordVecs: [(total + num) for total, num in zip(avgOfWordVecs, wv)] 

        #get length of wordVecs list
        wvLen = len(wordVecs)

        #loop through each word vector
        while(wvLen > 0):
            
            #update array and decrement counter
            avgOfWordVecs = add(wordVecs[wvLen-1], avgOfWordVecs)
            wvLen -= 1
        
        #get average 
        for i in range(0,wvLen):
            avgOfWordVecs[i] = (avgOfWordVecs[i] / len(wordVecs)) 

        

        return avgOfWordVecs
        

        
        

    #########################################################
    #####################HELPERS_END#########################
    #########################################################

    #########################################################
    ################NGRAM_FUNCTIONS##########################
    #########################################################
    """
    These functions are used for ngram analysis
    Include:
        - ngram creations
        - ngram comparisons
        - complete ngram analysis function
    """

    """
    Funtion to create ngrams from given phrase
    @params:
        - text: text to be coverted to ngram
        - n: number of division, pre-set to 3 (optional)
    @return: ngram list
    """
    def nGram(self, text, n=3):
        #to grab functions
       

        #lower text and remove non essential chars
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        #tokenize
        tokens = self.wordTokenizer(text)

        #return ngrams as a list
        return list(nltk.ngrams(tokens, n))
    """
    Searches for ngram similarities based on a supplied ngram
    threshold autoset to 0 but can be changed
    """
    def ngramSearch(self, ngram, toSearch, thresh=.75):
        #create NGram object for searchItem
        ngram = NGram(ngram, key = lambda x:x[1])

        #two lists 
        #simVals -> values on similarities
        #found -> found ngrams
        simVals = []
        found = []
        
        for n in toSearch:
            #actual searching
            f = sorted(ngram.searchitem(n, thresh))

            #ensures that no empty finds are saved
            if len(f) > 0:
                found.append(f)
                #print(f)
           

        for item in found:
            simVals.append(item[0][1])

       
        if len(simVals) > 0:
            #take the mean value of points collected
            return statistics.mean(simVals)
        else:
            return 0



    """ 
    looks at two sets of ngrams defined by num
    compares both sets based on length of shortest
    takes into account three warp types for compare function
    each ngram must have at least one data point

    @params: 
        - s1, s2  : sentences to be analyzed
        - warp    : similarity leniency, preset to 1 (optional)
        - num     : ngram number, preset to 3 (optional)

    @return: single average word vector
    """
    def compareNgram(self, s1, s2, warp=1, num=3):
        #defines how long the for loop will be going
        if len(s1) > len(s2):
            string2 = s1
            string1 = s2
        elif len(s2) > len(s1):
            string2 = s2
            string1 = s1
        else:
            string2 = s1
            string1 = s2

        #set counters
        cnt  = 0 
        indx = 0
        
        #going to be used to store the values
        valContainer = []
        compVals     = []
        avgVals      = []
        firstVal     = []
        secondVal    = []
        thirdVal     = []
        
        for n1 in string1:
            #reset counter and container list
            indx=0
            valContainer.clear()
            
            #set the secon ngram
            n2 = string2[cnt]

            
            while(indx<num):
                #compare the strings in a ngram with warp 1, 2, 3
                valContainer.append(NGram.compare(n1[indx],n2[indx], warp=warp))
                valContainer.append(NGram.compare(n1[indx],n2[indx], warp=warp+1))
                valContainer.append(NGram.compare(n1[indx],n2[indx], warp=warp+2))
                
                #add container to vals list
                compVals.append(valContainer)
                avgVals.append(statistics.mean(valContainer))

                #add to specific containers
                firstVal.append(valContainer[0])
                secondVal.append(valContainer[1])
                thirdVal.append(valContainer[2])
               
                #increment
                indx+=1

            #increment
            cnt+=1
        

        #algorithm I have decided to use in order to add weights and bias
        final = (statistics.mean(firstVal)*.5) + (statistics.mean(secondVal)*.1) + (statistics.mean(thirdVal)*.3) + \
                (statistics.mean(avgVals)*.1)

        return -(final-1)


    def finalNgramData(self, s1, s2, ngramNum=3, extra=False, extraS1=[], extraS2=[]):
        
        hash=True

        #clean text
        s1 = self.cTxt(s1)
        s2 = self.cTxt(s2) 

        #defines how long the for loop will be going
        if len(s1) > len(s2):
            smallString = s2
            bigString = s1
        elif len(s2) > len(s1):
            smallString = s1
            bigString = s2

        else:
            hash=False

        #creates string of equal length
        #ensures smaller sentence with larger does not equal greater equivalence
        addr=""
        cnt=0
        while cnt<ngramNum:
            addr = addr + "x"
            cnt+=1


        if hash:
            #add padding to the string
            while len(smallString) < len(bigString):
                smallString = smallString + addr + " "
            
            #reset the strings
            s1=bigString
            s2=smallString

        

        

        #put strings into usable nGram
        n1 = self.nGram(s1)
        n2 = self.nGram(s2)
    

        #run search function forward and reverse
        ngramSearch        = self.ngramSearch(n1, n2)
        ngramSearchReverse = self.ngramSearch(n2,n1)

        #take average of forward and reverse search
        searchFinal        = (ngramSearch + ngramSearchReverse)/2

        #compare the ngrams
        ngramCompare       = self.compareNgram(n1,n2)

        #take the averages of both series of tests
        final              = ((1-ngramCompare) * .7) + (searchFinal * .3)

        #return
        return final
    #########################################################
    ################NGRAM_FUNCTIONS##########################
    #########################################################

    """
    Uses word2vec to get a sentence average
    @params:
        - s1: Phrase or sentence to be averaged
    @returns : averaged sentence vector
    """
    def sent_vector_avg(self, s1):

        #create list to hold word vecs
        wordVecs = []
        

        #tokenize the sentence and init the index 
        #index used to move through list of tokenized sentences
        tokenized_sents = self.sentenceTokenizer(s1)
        i = 0
        
        #add word vectors to array after being tokenized and cleaned
        while i < len(tokenized_sents):
            for w in self.cTxt(tokenized_sents[i]).split(" "):
                try:
                    wordVecs.append(self.WVMODEL[w])
                except:
                    None
                
                #increment index
                i+=1
    
        return self.sent_avg_iter(wordVecs)
        

        

    """
    Simply returns cosine similarity of two sentence vector averages
    @params:
        - s1        : Phrase or sentence to be comapred, can be a sent_vect_avg
        - s2        : Phrase or sentence to be compared, can be a sent_vect_avg
        - vectorized: if set to true will alter return slightly by skippin sent_vector_avg call
    @returns : Cosine similarity of two sentence vecotr averages
    """
    def sent_vector_comp(self, s1, s2, vectorized=False):
        
        if vectorized: 
            return 1-spatial.distance.cosine(numpy.array(s1), numpy.array(s2))
        
        else:
            return 1-spatial.distance.cosine(numpy.array(self.sent_vector_avg(s1)), numpy.array(self.sent_vector_avg(s2)))


    """
    Finds the sentence most similar to a base sentence (user given)
    @params:
        - s2compare  : Base sentence to compare to
        - sentList   : List of sentences to compare base too
        - vectorAvg  : if any vector average list is given, preset to false (optional)
        - sentVecList: list of sent vectors if given (optional)
    @returns : Sentence with highest cosine similarity to base sentence
    """
    def similar_sent_vector(self, s2Compare, sentList):

        #now compare each sentence in sentVecs to provided sentence
        compare = lambda s2C, sentVec: self.sent_vector_comp(s2C, sentVec) 

        #set up save counters to determine best sentence
        bestSentMatch = ""
        highestCosineMatch = 0

        #compare each vector avg in sentVecs with s2compare
        for vec in sentList:
            cosineMatch = compare(vec, s2Compare)
            

            #update save counters if new best match found
            if cosineMatch > highestCosineMatch:
                highestCosineMatch = cosineMatch
                bestSentMatch = vec

        #return save counters as a list
        return [bestSentMatch, highestCosineMatch]

    #########################################################
    ################ORG___FUNCTIONS##########################
    #########################################################

    def sent_avg_cluster(self, wordList):

        #create container to hold all clusters
        clusters = []

        #create initial cluster
        firstCluster = {
                        "Cluster_Sents": [wordList[0]], 
                        "Cluster_Avg"  : self.sent_vector_avg(wordList[0]),
                        "Items"        : 1
                        }

        #add firstCluster to clusters list
        clusters.append(firstCluster)

        #print(len(clusters))
        
        
        
        for sent in wordList[1:]:

            newCluster = True

            #get sentence vector avg
            sentAvg = self.sent_vector_avg(sent)

            #counters to keep track of which clusters need to be compared
            highestAvg = 0
            indexOfCluster = 0
            currIndex = 0


            
            for cluster in clusters:

                
            
                #get cluster avg and compare with sentence
                clusterAvg = cluster.get("Cluster_Avg")
                clusterSentComp = self.sent_vector_comp(sentAvg, clusterAvg, True)
                

                

                if clusterSentComp > .88:
                    if clusterSentComp > highestAvg:
                        highestAvg = clusterSentComp
                        indexOfCluster = currIndex
                    #set cluster flag
                    newCluster = False

                currIndex+=1

            #if there is going to be an addition to an existing cluster
            if not newCluster:
                #grab all items from dictionary and update with new values

                #get cluster average of closest cluster within params
                clusterAvg  = clusters[indexOfCluster].get("Cluster_Avg")
                tempAvgs    = [clusterAvg, sentAvg]

                #update sentence list of the chosen cluster
                newSentList = clusters[indexOfCluster].get("Cluster_Sents")
                newSentList.append(sent)

                #update item val and overall avg of chosen cluster
                newSentAvg  = self.sent_avg_iter(tempAvgs)
                newItemsVal = clusters[indexOfCluster].get("Items") + 1

                #pop old values
                clusters[indexOfCluster].clear()

                #re-add new values
                clusters[indexOfCluster]["Cluster_Sents"] = newSentList
                clusters[indexOfCluster]["Cluster_Avg"]   = newSentAvg
                clusters[indexOfCluster]["Items"]         = newItemsVal
                    

            
            #set new cluster if sentence did not fit into any of the other clusters
            else:
                #create next cluster
                nextCluster = {
                        "Cluster_Sents": [sent], 
                        "Cluster_Avg"  : self.sent_vector_avg(sent),
                        "Items"        : 1
                        }
                
                
                #add new cluster to list of clusters
                clusters.append(nextCluster)

        #for testing
        for cluster in clusters:
            print(cluster)
            print("\n\n")
        







                


                    















        


        
        

        
        

        

    #---------------------------------------------------------
    """
    grabs the phrases from a given file
    parameter: file, a string of the file path
    """
    """
    def phrases(self, file):
        f = open(file, "r")
        phrases = f.read()
        phrases = phrases.split(",")
        return phrases

    


    def freqDist(self, tagged):
        x = nltk.FreqDist(tag for (word, tag) in tagged)  
        x.plot(cumulative=True)


        

    def compareToken(self, text, keyW=False):
       
        #if key word comparison is needed
        if keyW:
            #gets the hit phrases from file and tokenize
            test = self.phrases('phrases.txt')
            test = self.tagger(test)
        else:
            text = test
    """
