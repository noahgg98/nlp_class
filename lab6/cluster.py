from transformers import BertTokenizer, BertModel, BertConfig
from gensim.parsing.preprocessing import remove_stopwords
from data import *
#from werkzeug.datastructures import T
from mybert import *
import numpy as np
import sklearn
# Don't forget your library imports!
import matplotlib.pyplot as plt   # we had this one before
from wordcloud import WordCloud   # new for WordCloud

N = 100
M = 768


reviews = get_review_sentences(N=100)    # returns 100 review sentences as a List of strings


# Create an empty matrix NxM where M is the length of your embeddings and you have N samples.
your_numpy_matrix = np.empty(shape=(N,M))

# Now fill in all the rows with your BERT embeddings
for i in range(N):	    
  your_numpy_matrix[i] = sent_func(reviews[i]).detach().numpy()


# OPTION 1: Runs k-means cluster with hard-coded k-clusters
labels = kmeans(your_numpy_matrix, k=10)   # labels each row with a cluster ID

for i in range(10):
  print("*****Cluster {}*****".format(str(i)))
  print("content")
  for j in range(N):
    
    if labels[j] == i:
      print(reviews[j])
    
      

# OPTION 2: Runs agglomerative clustering that discovers the \# of clusters
#agg = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0.15, affinity="cosine", linkage="average", compute_distances=True)
#labels = agg.fit(your_numpy_matrix).labels_



while True:
  clustNum = int(input("What cluster to cloud? "))
  # The cloud!
  doc = ""

  for i in range(N):
    if labels[i] == clustNum:
      doc = doc + " " + reviews[i].strip("\n") 
    doc = remove_stopwords(doc)
  cloud = WordCloud(width=480, height=480, margin=0).generate(doc)    # 'doc' is the constructed tweet string
            
  # Now popup the display of our generated cloud image.
  plt.imshow(cloud, interpolation='bilinear')
  plt.axis("off")
  plt.margins(x=0, y=0)
  plt.show()
