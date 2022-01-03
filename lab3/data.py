from os import listdir
from os.path import isdir, isfile, join
import sys


class Passages:

    def __init__(self, d='data'):
        self.rootdir = d

    def get_training(self):
        return self.read_datums(join(self.rootdir, 'train'))

    def get_development(self):
        return self.read_datums(join(self.rootdir, 'dev'))

    def get_test(self):
        return self.read_datums(join(self.rootdir, 'test'))
    
    def read_datums(self, dirpath):
        '''
        Read all passages from all authors in the given directory.
        Returns a list of pairs: (author,text)
        '''
        # Directory exists?
        if not isdir(dirpath):
            print('ERROR: bad directory path', dirpath)
            sys.exit(1)

        passages = list()
            
        # Read the author files from this directory.
        files = [join(dirpath,f) for f in listdir(dirpath) if isfile(join(dirpath, f))]
        for f in files:
            author = f[f.rindex('-')+1:].upper()
            passages.extend(self.read_passages(f,author))

        # List of tuples (author,text)
        return passages
    
    def read_passages(self, path, author):
        ''' 
        Read passages from a single file.
        Returns a list of pairs: (author,text)
        '''
        fh = open(path)
        passages = list()
        text = ''
        for line in fh:
            if '--**--**--' in line:
                if len(text) > 10:
                    passages.append( [author,text.strip()] )
                text = ''
            else:
                text += line
        fh.close()
        return passages


def evaluate(guesses, passages):
    '''
    The length of the two given lists must be equal.
    guesses: a list of author names
    passages: a list of pairs (author,text)
    '''

    # Sanity check.
    if len(guesses) != len(passages):
        print('ERROR IN EVALUATE: you gave me', len(guesses), 'guessed labels, but', len(passages), 'passages.')
        return 0.0

    # Compare the guesses with the gold labels.
    numRight = 0
    numWrong = 0
    xx = 0;
    for guess in guesses:
      passage = passages[xx]
      if guess == passage[0]:
        numRight += 1
      else:
        numWrong += 1
      xx += 1

    # Compute accuracy.
    print("Correct:   " + str(numRight))
    print("Incorrect: " + str(numWrong))
    accuracy = numRight / (numRight+numWrong)
    return accuracy * 100.0

    
    
if __name__ == '__main__':

    d = Passages()
    passages = d.read_datums('data/dev')
    print(passages)
