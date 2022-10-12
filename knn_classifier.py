from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn import metrics
from numpy.linalg import norm
from scipy import spatial



def classify(X_train, X_test, y_train, y_test, k):

  prediction = []
  accuracy_score = 0

  print("There are",X_test.shape[0],"testing points and",X_train.shape[0],"training points")

  for i in range(X_test.shape[0]):
      cosine_distances = {}
      #print("Training with test point", i)

      for j in range(X_train.shape[0]):

          A = X_test.iloc[i,:]
          B = X_train.iloc[j,:]

          cosine = spatial.distance.cosine(A,B)
          cosine_distances[(i,j)] = 1-cosine
        
      cosine_distances_sorted = dict(sorted(cosine_distances.items(), key=lambda item: item[1],reverse=True))
      closest_k_neighbors = dict(list(cosine_distances_sorted.items())[0: k])
    #  print("For test point",i,"k closest distances from training points are", closest_k_neighbors)

      closest_k_neighbors_keys = list(closest_k_neighbors.keys())

      predicted_labels = []

      for pair in closest_k_neighbors_keys:
        training_index = pair[1]
        label = y_train[training_index]
        predicted_labels.append(int(label))

      label_sum = sum(predicted_labels)
    
      if label_sum < (k/2):
        prediction.append(0)
      elif label_sum > (k/2):
        prediction.append(1)
   
  #print(y_test['Occupancy'].values.tolist())

  accuracy_score = metrics.accuracy_score(y_test,prediction)

  return accuracy_score