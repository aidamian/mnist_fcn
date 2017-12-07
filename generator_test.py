# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:11:34 2017

@author: Andrei
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata

if __name__=="__main__":

  mnist = fetch_mldata('MNIST original')

  test_size = 100
  new_image = np.zeros((test_size,test_size))
  
  
  nr_images = 10
  
  indices = np.arange(mnist.data.shape[0])

  orig_h, orig_w = (28,28)
  
  sampled = np.random.choice(indices,nr_images, replace=False)
  labels = []
  pos_w = np.random.randint(1,10)
  pos_h = np.random.randint(1,10)
  for i in range(nr_images):
    ind = sampled[i]
    x_cur = mnist.data[ind]
    if (pos_h<=(test_size-orig_h)):
      new_image[pos_h:(pos_h+orig_h),pos_w:(pos_w+orig_w)] = x_cur.reshape((orig_h,orig_w))
      labels.append(mnist.target[ind])
      pos_w += 28 + np.random.randint(-5,10)
    if pos_w >= (test_size - orig_w):
      pos_w = np.random.randint(1,10)
      pos_h += 28 + np.random.randint(-3,8)
  plt.matshow(new_image,cmap="gray")
  plt.show()