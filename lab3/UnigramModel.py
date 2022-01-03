#
# A fully functional unigram language model.
#
# @author Nate Chambers
#
from LanguageModel import LanguageModel
import random

class UnigramModel(LanguageModel):

    def __init__(self):
        super().__init__()

        self.wordcounts = dict()
        self.wordcounts[self.STOP] = 0
        self.N = 0

    # REQUIRED FUNCTIONS from abstract parent LanguageModel
        
    def train(self, sentences):
        
        for s in sentences:
            s = s.copy()          # don't want to alter the original given sentences
            s.append(self.STOP)
          
            for w in s:
                if not w in self.wordcounts:
                    self.wordcounts[w] = 1
                else:
                    self.wordcounts[w] += 1
            self.N += len(s)

    def get_word_probability(self, sentence, index):
        if index == len(sentence):
            return self.get_unigram_probability(self.STOP)
        else:
            return self.get_unigram_probability(sentence[index])

    def get_vocabulary(self):
        words = list(self.wordcounts.keys())
        words.append(self.UNK)
        return words

    def generate_sentence(self):
        words = []
        word = self.generate_word()
        while word != self.STOP:
            words.append(word)
            word = self.generate_word()
        return words

    # HELPER FUNCTIONS for this unigram language model
    
    def get_unigram_probability(self, word):
        # Pretends an <UNK> token was seen once in the data.
        if not word in self.wordcounts:
            return 1.0 / (self.N+1.0)
        else:
            return self.wordcounts[word] / (self.N+1.0)

    def generate_word(self):
        threshold = random.uniform(0, 1)
        sum = 0.0
        for word in self.wordcounts.keys():
            sum += self.get_unigram_probability(word)
            if sum > threshold:
                return word
        
#
# TESTING ONLY
#
if __name__ == '__main__':

    u = UnigramModel()
    u.train( [ ['the','dog','is','angry'], ['the','cat','is','surly'] ] )

    print('P(the) =   ', u.get_unigram_probability('the'))
    print('P(cat) =   ', u.get_unigram_probability('cat'))
    print('P(is) =    ', u.get_unigram_probability('is'))
    print('P(surly) = ', u.get_unigram_probability('surly'))
    print('P(new) =   ', u.get_unigram_probability('new'))    
