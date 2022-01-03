from LanguageModel import LanguageModel
from BigramModel import BigramModel 

class SmoothedBigram(BigramModel):

    def __init__(self):
        super().__init__()
        self.MAX_NUM = 0
        self.COUNTS  = dict()

        self.K       = .01111
        self.V       = 0
       

    def train(self, sentences):
        super().train(sentences)      
        self.V = len(list(self.get_vocabulary())) * self.K
    
    def set_K(self, newVal):
        self.K = newVal
    
            
    def bigram_probability(self, bigram, word):

        #laplace smoothing
        bCount = 0.0 if bigram not in self.bicounts else self.bicounts[bigram] 
        wCount = 0.0 if word not in self.wordcounts else self.wordcounts[word]

        #return new probability with k value pre-set
        return (bCount+self.K)/(wCount+self.V)

   

