#coding: utf-8
import numpy as np
from collections import Counter

class Knn(object):
  def __init__(self, n_neighbor):
    self.n_neighbor = n_neighbor

  def fit(self, features, labels):
    self.features = features
    self.labels = labels

  def predict(self, example):
    dist = [(self.labels[i], distance(example, self.features[i]))
                 for i in range(len(self.features))]
    dist.sort(key=lambda x: x[1])
    return Counter([d[0] for d in dist[:self.n_neighbor]]).most_common(1)[0][0]

def distance(x0, x1):
  return np.sum((x0 - x1) ** 2)
