from LanguageModel import LanguageModel
from UnigramModel import UnigramModel
import random


class BigramModel(UnigramModel):

    def __init__(self):
        super().__init__()

        #set wordcount start tag value
        self.wordcounts[self.START]= 0

        self.wordTotal = list() 

        #bigram dictionaries
        self.bicounts = dict()
        self.bicounts[self.STOP] = 0

        #define start tag and length of word
        self.N = 0
      

    # REQUIRED FUNCTIONS from abstract parent LanguageModel
        
    def train(self, sentences):

        #use the unigram model to update wordcounts based on unigram
        #updates N
        super().train(sentences)
        
        #implement bigram version now
        for s in sentences:

            #add stop tag
            s = s.copy()
            s.append(self.STOP)

            #add start tag to front of sentence
            s = list(reversed(s))
            s.append(self.START)
            s = list(reversed(s))

            #create bigram of sentence
            sentBigram = self.create_bigram(s)

            #update start tag value each sentence
            self.wordcounts[self.START] += 1

            #find each bigram and update count or add to dict
            for bi in sentBigram:
                if not bi in self.bicounts:
                    self.bicounts[bi] = 1
                else:
                    self.bicounts[bi] += 1

        

    def get_word_probability(self, sentence, index):

        #reconstruct sentence to create bigram if under length 2
        if len(sentence) < 2:
            sentence = [self.START, sentence[index]]
            index+=1

        

        #set word to be tested
        word = sentence[index-1] 
        tup = (sentence[index-1],sentence[index]) 
        


        return self.bigram_probability(tup, word)

    def generate_sentence(self):
        
        words = []
        word = self.generate_word()
        while word != self.STOP:

            #update local and gloabl 
            words.append(word)
            self.wordTotal.append(word)
            word = self.generate_word()

        self.wordTotal = list()
        return words
       

    def generate_word(self):
            threshold = random.uniform(0, 1)
            sum = 0.0
            for word in self.wordcounts.keys():

                #define proper bigram to probability function
                lenOverall = len(self.wordTotal)
                lastEle = self.wordTotal[lenOverall - 1] if lenOverall > 0 \
                        else "<S>"

                #get bigram and return word once sum reached thresh
                sum += self.bigram_probability((lastEle, word), lastEle)
                if sum > threshold:
                    return word
    
    def bigram_probability(self, bigram,word):
    
        
        if bigram not in self.bicounts:
            return 0

        #get values with gram counts
        bigramCount = self.bicounts[bigram]
        wordCount   = self.wordcounts[word]

        #MLE
        prob = bigramCount/wordCount

       

        return prob

    
    #HELPER FUNCTIONS   
    def create_bigram(self, sentence):

        grams = []
        

        #go through index creating unigrams
        for i in range(0, len(sentence)-1):
            grams.append((sentence[i], sentence[i+1]))

        return grams

        
#
# TESTING ONLY
#
if __name__ == '__main__':
    print('hi!')

