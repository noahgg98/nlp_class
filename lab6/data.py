import csv
from nltk.tokenize import sent_tokenize

from scipy.spatial import distance
import numpy as np


def get_review_sentences(filename='reviews.csv', N=1000):
  '''
  Returns a list of strings - sentences from the reviews data.
  '''
  seen = {}
  sentences = []

  with open(filename) as csvfile:
    readit = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in readit:
      if len(row) > 3:
        username = row[0]
        appID = row[-1]
        unique = username + '--' + appID

        # The data has duplicate reviews, don't re-add them.
        if not unique in seen:
          seen[unique] = True
          review = row[2]
          sentences.extend(sent_tokenize(row[2]))

      if len(sentences) >= N:
        break

  if len(sentences) > N:
    sentences = sentences[0:N]
      
  return sentences



def kmeans(X,k=3,max_iterations=100):
    '''
    X: multidimensional data
    k: number of clusters
    max_iterations: number of repetitions before clusters are established
    
    Steps:
    1. Convert data to numpy aray
    2. Pick indices of k random point without replacement
    3. Find class (P) of each data point using euclidean distance
    4. Stop when max_iteration are reached of P matrix doesn't change
    
    Return:
    np.array: containg class of each data point

    FROM: https://gdcoder.com/implementation-of-k-means-from-scratch-in-python-9-lines/
    '''
#    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
        if np.array_equal(P,tmp):break
        P = tmp
    return P
