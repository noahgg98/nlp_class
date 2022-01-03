
import data
from sklearn.linear_model import LogisticRegression
import numpy as np
from DataHelper import *

class Classifier:

    def __init__(self, filename):
        self.clf  = LogisticRegression(max_iter=3000)
        self.TAGS = list()
        self.EMBS = list()
        self.FN   = filename


    def train(self):
        
        #create DH helper
        datahelper = DataHelper("test.txt")
        datahelper.get_data()

        #get embeddings and tags
        embeds = datahelper.ret_embs()
        tags   = datahelper.ret_tags()


        #TESTING PURPOSES ONLY***********
        tempe = list()
        tempt = list()

        for i in range(0, 2000):
            tempe.append(embeds[i])
            tempt.append(tags[i])

        for i in range(100000, 102000):
            tempe.append(embeds[i])
            tempt.append(tags[i])
 
        #TESTING PURPOSES ONLY***********

        #test 
        myb = Bert('ready_tweet_files.txt')
        x = myb.bert_embedding(myb.clean("I hope that Joe Biden wins the next election")).detach().numpy() 



        self.clf.fit(embeds, tags)

        #test
        guesses = self.clf.predict([embeds[2], x, embeds[1000]])
        print(guesses)
        
        #train model
#we need to add a test portion

        



if __name__=='__main__':
    cl = Classifier("embed_files.txt")

    cl.train()
