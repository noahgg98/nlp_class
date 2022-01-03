from abc import ABC, abstractmethod

class LanguageModel(ABC):

    def __init__(self):
        super().__init__()
        self.START = '<S>'
        self.STOP = '</S>'
        self.UNK = '<UNK>'                

    # CHILD CLASSES MUST IMPLEMENT THESE ABSTRACT METHODS
        
    @abstractmethod
    def train(self, sentences):
        ''' Void method. The given sentences is a list of lists of strings. '''
        pass

    @abstractmethod
    def get_word_probability(self, sentence, index):
        ''' 
        Return one float. 
        Compute the probability of the token at position INDEX in the given SENTENCE. 
        '''
        pass

    @abstractmethod
    def get_vocabulary(self):
        ''' Return a list of words! Probably the keys() of a dictionary. '''
        pass

    @abstractmethod
    def generate_sentence(self):
        ''' Return one list of strings (words). '''
        pass

    
    # DON'T CHANGE THESE
    
    def get_sentence_probability(self, sentence):
        '''
        Returns the probability, according to the model, of the specified
        sentence.  This is the product of the probabilities of each word in
        the sentence (including a final stop token).
        '''
        prob = 1.0
        for i in range(0,len(sentence)):
            prob *= self.get_word_probability(sentence,i)
        return prob

    def check_probability(self, context):
        '''
        Given a list of words, sums over the probabilities of every token that
        could follow. If the model implements a valid probability 
        distribution, this should always sum to 1.
        '''
        modelsum = 0.0
        for word in self.get_vocabulary():
            context.append(word)
            prob = self.get_word_probability(context, len(context)-1)
            modelsum += prob
            context.pop()
        return modelsum
