#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# rot90 x 3 | input shape (3,3)
def rotations(x1):
  x2 = np.rot90(x1)
  x3 = np.rot90(x2)
  x4 = np.rot90(x3)
  return [x2, x3, x4]

# mirroring | input shape (3,3)
def mirror(x1):
  x2 = x1[::-1]
  x3 = np.array([i[::-1] for i in x1])
  return [x2, x3]

# rotations x mirror | input shape = (9,)
def mutate(arr):
  x1 = arr.reshape(3,3)
  res = [x1]
  res += [i for r in rotations(x1) for i in mirror(r)]
  res += [i for m in mirror(x1) for i in rotations(m)]
  return np.array([i.reshape(9) for i in res])

# mutate dataset (n, 2, 9)
def mutate_data(data):
  # mutate arrays | shapes = (n, 9)
  x1 = np.array([j for i in data[:, 0] for j in mutate(i)])
  y1 = np.array([j for i in data[:, 0] for j in mutate(i)])

  # add axis | shapes = (n, 1, 9)
  x1 = x1[:, np.newaxis]
  y1 = y1[:, np.newaxis]

  # merge arrays | shape (n, 2, 9)
  return np.concatenate((x1, y1), axis=1)
