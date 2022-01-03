from keras import models
from keras.backend import dropout, permute_dimensions
from numpy.lib.npyio import save
from DataHelper import *
import re
import random
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class TextGenerator:

    def __init__(self):
        self.TWEETSTRS = ""
        self.CHARS = list()
        self.C = 0
        self.V = 0


        self.X = 0

        #data to be fed into model
        self.DATA_X_AXIS = []
        self.DATA_Y_AXIS = []

        #set sequential model
        self.MODEL = Sequential()

        #numpy array and one hot encoding
        self.NARR = None
        self.OHE  = None
    
    def clean(self, text):
        words = text.split(" ")
        s = list()
        for w in words:
            w = re.sub(r'http\S+', '', w)
            w = re.sub(r'RT+', '', w)
            w = re.sub(r'RT[A-Za-z0-9]+', '', w)
            w = re.sub(r'@[A-Za-z0-9]+', '', w)
            reg = re.compile('[^a-zA-Z]')
            w = reg.sub('', w)
            if not w  == "":
                s.append(w)
        return " ".join(s)

    
    def get_strings(self):

        #create DH helper
        bert = Bert("")
        datahelper = DataHelper("test.txt")
        datahelper.get_data()
        twtStrs = datahelper.ret_strings()

        """uncomment this section if you want random section during training"""
        """
        #get subsection
        length = len(twtStrs) - 1500
        rand = random.randint(0, length)
        topEnd = rand+1500
        twtStrs = twtStrs[rand : topEnd]
        for string in twtStrs:
            self.TWEETSTRS = self.TWEETSTRS + self.clean(string)
        """
        """uncomment this section if you want random section during training"""


        #clean with bert cleaning and add to large string
        for string in twtStrs[0:2000]:
      
            self.TWEETSTRS = self.TWEETSTRS + self.clean(string)

        #print(self.TWEETSTRS)
        return self.TWEETSTRS

        
    def set_unique(self):

        #get unqie characters
        self.CHARS = sorted(list(set(self.TWEETSTRS.lower())))

        #set vocab and character values
        self.C = len(self.TWEETSTRS)
        self.V = len(self.CHARS)

        
    
    def conversions(self, direction):

        #allow for conversion between int and char
        if direction == 0:
            c2i = dict((c,i) for i,c in enumerate(self.CHARS))
            return c2i
        else:
            i2c = dict((i,c) for i,c in enumerate(self.CHARS))
            return i2c

    def prep_data(self):

        #define sequence length
        slen = 50

        #get conversion returns 
        conv = self.conversions(0)

        #print(self.C - slen) #need o change below for proper calcs
        for i in range(0, self.C - slen):

            sin  = self.TWEETSTRS.lower()[i:i+slen]
            sout = self.TWEETSTRS.lower()[i + slen]

            #grab conversion vals from conv dict
            conv_dict = [conv[c] for c in sin]
            conv_val  = conv[sout]

            #append to x and y axis daya
            self.DATA_X_AXIS.append(conv_dict)
            self.DATA_Y_AXIS.append(conv_val)

    def set_model(self):

        #from documented site --> [sample, timesteps, features]
        # 50 is same as slen from prep_data
        self.NARR = numpy.reshape(self.DATA_X_AXIS, (len(self.DATA_X_AXIS), 50, 1))

        #normalize
        self.NARR = self.NARR / float(self.V)

        #one hot encoding
        self.OHE = np_utils.to_categorical(self.DATA_Y_AXIS)


        #add pieces to model
        self.MODEL.add(LSTM(256, input_shape=(self.NARR.shape[1], self.NARR.shape[2]), return_sequences=True))
        self.MODEL.add(Dropout(.2))
        self.MODEL.add(LSTM(256, return_sequences=True))
        self.MODEL.add(Dropout(.2))
        self.MODEL.add(LSTM(256))
        self.MODEL.add(Dropout(.2))
        self.MODEL.add(Dense(self.OHE.shape[1], activation='softmax'))
 
        #compile the model
        self.MODEL.compile(loss='categorical_crossentropy', optimizer='adam')

    def fit(self):

        #set weights file name
        wfn = "{epoch:02d}-{loss:.4f}.hdf5"

        #define checkpoint
        callback = [ModelCheckpoint(wfn, monitor='loss', verbose=1, save_best_only=True, mode='min')]

        print("fitting")

        #fit the model
        self.MODEL.fit(self.NARR, self.OHE, epochs=50, batch_size=64, callbacks=callback)

    def train(self):
        
        #run process to train the model
        self.get_strings()
        self.set_unique()
        self.prep_data()
        self.set_model()
        #self.fit()     #comment this out if model already exists to load

    def gen_text(self, file2use):

        #get saved weights from training
        self.MODEL.load_weights(file2use)
        self.MODEL.compile(loss='categorical_crossentropy', optimizer='adam')

        #get int to char conversion
        intConv = self.conversions(1)

        #get start based on random seed
        st = numpy.random.randint(0, len(self.DATA_X_AXIS)-1)

        start = self.DATA_X_AXIS[st]

        #gen text
        for i in range(350):
            npyarr = numpy.reshape(start, (1, len(start), 1))
            npyarr = npyarr / float(self.V)
            pred = self.MODEL.predict(npyarr, verbose=0)
            index = numpy.argmax(pred)
            result = intConv[index]
            sin = [intConv[item] for item in start]
            print(result, end='')
            start.append(index)
            start = start[1:len(start)]








    
    


if __name__=="__main__":
    dh = TextGenerator()

    #for i in dh.get_strings():
    #    print(i)
    #dh.get_strings()
    #dh.set_unique()
    #print(dh.clean("🇺🇸RT : LIVE: Trump delivers remarks at the Signing Ceremony for the National Defense Authorization Act for FY 2020.My fellow Americans:"))
    dh.train()
    file = input("which file? ")
    dh.gen_text(file)
	
