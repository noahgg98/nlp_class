from os import listdir
from os.path import isdir, isfile, join
import sys
import gzip


class Tweets:

    def __init__(self, d='data'):
        self.rootdir = d
        tweetdir = join(d,'tweets')
        
        # Raw tweet files.
        self.files = [join(tweetdir,f) for f in listdir(tweetdir) if isfile(join(tweetdir, f))]
        self.files = sorted(self.files)[0:3]  # remove [0:3] for more files
        self.fh = None
        self.reset()

    def reset(self):
        if self.fh:
            self.fh.close()
        self.currenti = -1
        self.fh = None    
        
    def get_next_tweet(self):
        '''
        Returns a single tweet from the currently open file. If at the end of the tweet
        file, it closes the file and opens the next file in the directory.
        '''
        # If first time, or the file handle is invalid, open the next file.
        if self.currenti == -1 or not self.fh:
            self.currenti += 1
            if self.currenti < len(self.files):
                print('Opening', self.files[self.currenti], '...')
                self.fh = gzip.open(self.files[self.currenti], 'r')
            else:
                return None

        # Read one tweet
        asbytes = self.fh.readline()
        try:
            t = asbytes.decode("utf-8")
        except:
            return self.get_next_tweet()

        # Reached end of file: python is dumb
        if t == '':
            self.fh.close()
            self.fh = None
            return self.get_next_tweet()
        
        # "tweet \t user \t timestampe \t timezone"
        t = t.split('\t')[0]
        if t == None:
            print('RETURNING None from',asbytes)
        return t

    def get_labeled_tweets(self):
        '''
        Returns a list of pairs: (label,tweet)
        '''
        tweets = []
        with open(join(self.rootdir,'combined-labeled-tweets.txt')) as fh:
            for line in fh:
                line = line.strip()
                (label,tweet) = line.split('\t')
                tweets.append( (label,tweet) )
        return tweets

    def get_lexicon(self):
        ''' Returns a tuple of two Sets: positive words and negative words '''

        with open(join(self.rootdir,'lexicon','positive-words.txt'), encoding='ISO-8859-1') as fh:
            lines = fh.readlines()
            pos = [ x.strip() for x in lines ]

        with open(join(self.rootdir,'lexicon','negative-words.txt'), encoding='ISO-8859-1') as fh:
            lines = fh.readlines()
            neg = [ x.strip() for x in lines ]

        return (set(pos),set(neg))


def evaluate(guesses, labels):
    '''
    The length of the two given lists must be equal.
    guesses: a list of guessed string labels
    labels: a list of correct string labels
    '''

    # Sanity check.
    if len(guesses) != len(labels):
        print('ERROR IN EVALUATE: you gave me', len(guesses), 'guessed labels, but', len(labels), 'gold ones.')
        return 0.0

    rights = dict()
    wrongs = dict()
    golds = dict()
    
    # Compare the guesses with the gold labels.
    numRight = 0
    numWrong = 0
    xx = 0;
    for guess in guesses:
      gold = labels[xx]
      golds[gold] = golds.get(gold,0) + 1
      if guess == gold:
        numRight += 1
        rights[guess] = rights.get(guess,0) + 1
      else:
        numWrong += 1
        wrongs[guess] = wrongs.get(guess,0) + 1
      xx += 1

    # Compute precision.
    for y in sorted(golds):
        if not y in rights:
            p = 0
            r = 0
        else:
            p = rights.get(y,0) / (rights.get(y,0)+wrongs.get(y,0))
            r = rights.get(y,0) / golds[y]
        print(y.upper())
        print('  Precision = %d/%d = %.2f' % (rights.get(y,0), (rights.get(y,0)+wrongs.get(y,0)), p))
        print('  Recall    = %d/%d = %.2f' % (rights.get(y,0), golds[y], r))

    # Compute accuracy.
    print("Correct:   " + str(numRight))
    print("Incorrect: " + str(numWrong))
    accuracy = numRight / (numRight+numWrong)

            
    return accuracy * 100.0



if __name__ == '__main__':

    d = Tweets()
    for i in range(10):
        print(d.get_next_tweet())
