#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

# get best movement (9x3 arrays)
def bestMove(initial, predict):
  axis = (None, None)
  f = 0

  # check availability
  for yi, row in enumerate(predict):
    for xi, res in enumerate(row):
      if initial[yi][xi] == 0 and res > f:
        axis = (yi, xi)
        f = res
  return axis

# suggest movement
def suggest(nn, arr):
  # check shape
  if arr.shape != (9,):
    raise Exception('invalid input shape')

  # first movement
  if reduce(lambda x,y: x and y, arr == 0):
    return (2,0)

  # return best movement
  i = arr.reshape(1,-1)
  res = nn.predict(i)[0].reshape(3,3)
  return bestMove(arr.reshape(3,3), res)
