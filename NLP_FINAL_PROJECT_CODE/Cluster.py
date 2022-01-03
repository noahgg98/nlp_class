
from transformers import BertTokenizer, BertModel, BertConfig
from data import *
from transformers.models import bert
from bert import *
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt   # we had this one before
from scipy.spatial import distance
from DataHelper import *
from sklearn.cluster import KMeans
import random


class Clusters:

    def __init__(self, k):
        self.CENTROIDS = []
        self.EMBEDS    = []
        self.STRS      = []
        self.TAGS      = []
        self.CLUSTERS  = []
        self.K         = k
        

    def set_data(self):
        #create DH helper
        datahelper = DataHelper("test.txt")
        datahelper.get_data()

        #retrieve embeddings and strings
        self.EMBEDS = np.array(datahelper.ret_embs())
        self.STRS   = datahelper.ret_strings()
        self.TAGS   = datahelper.ret_tags()


    def cluster(self):

        #build cluster
        km = KMeans(n_clusters=self.K)
        km.fit(self.EMBEDS)
        self.CLUSTERS = km.predict(self.EMBEDS)

        #get the centroids from the cluster
        self.CENTROIDS = km.cluster_centers_
    
    def condense_cluster(self, clusternum):

        #define cluster list
        clusterStrs = list()

        #if cluster num does not exist
        if clusternum > self.K:
            print("Bad Cluster Num Exiting...")

        #loop through labels and associated text strings
        #combine all text strings into single list
        stringIndx = 0
        for label in self.CLUSTERS:
            if label == clusternum:
                clusterStrs.append(self.STRS[stringIndx])
            stringIndx += 1

        return clusterStrs

        
    def sample_cluster(self, clusternum, sampleSize = 20):

        #get cluster strings
        clusterStrs = self.condense_cluster(clusternum)


        #check to ensure that sample size is withing cluster size
        if sampleSize > len(clusterStrs):
            print("Your sample size is bigger than the size of the Cluster, Exiting...")

        
        #grab random strings from the cluster
        #picks random index from cluster strings
        finalStrList = list()
        for i in range(0, sampleSize):

            rand = random.randint(0, len(clusterStrs)-1)
            finalStrList.append(clusterStrs[rand])

        return finalStrList
    
    def get_closest_cluster(self, str2check):

        #turn into bert embedding
        bertEmb = Bert('').bert_embedding(str2check)


        #get cosine distance here  
        # and save the closest one   
        closest    = 0      
        clusternum = 0
        for centroids in self.CENTROIDS:
            dist = 1 - distance.cosine(bertEmb.detach().numpy(), centroids)
            if dist > closest:
                closest = clusternum
            clusternum+=1

        return closest



    def populate(self):

        #set input str
        inputStr = ""

        #run into loop user hasnt exited
        while True:
            
            #get input to cluster and turn into embedding
            inputStr = input("Input test string: ")

            #exit once comparisons are done
            if inputStr=="exit" or inputStr=="quit":
                break

            #get a sample form the closest cluster
            strpops = self.sample_cluster(self.get_closest_cluster(inputStr))

            #print the sampling
            print("Printing a sample from the closest cluster")
            for strii in strpops:
                print(strii)
            
            

if __name__=='__main__':

    cl = Clusters(5)

    cl.set_data()
    cl.cluster()
    cl.populate()




