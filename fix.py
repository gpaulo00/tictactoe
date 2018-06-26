#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np

# read remaining positions
array = None
with open("results.json", "r") as fp:
  array = json.loads(fp.read())

# data: []item
# item: [20]int (9 board, 9 move, 2 win/lose)
data = np.load("tictactoe.npy")

# rot90 x 3
def rotations(x1):
  x2 = np.rot90(x1)
  x3 = np.rot90(x2)
  x4 = np.rot90(x3)
  return [x2, x3, x4]

# mirroring
def mirror(x1):
  x2 = x1[::-1]
  x3 = np.array([i[::-1] for i in x1])
  return [x2, x3]

# rotations x mirror
def mutations(arr):
  x1 = arr.reshape(3,3)
  res = [x1]
  res += [i for r in rotations(x1) for i in mirror(r)]
  res += [i for m in mirror(x1) for i in rotations(m)]
  return [i.reshape(9).tolist() for i in res]

# handle data
result = []
for item in data:
  position = mutations(item[:9])
  move = mutations(item[9:18])
  win_lose = item[18:].tolist()

  for i in range(9):
    # remove position from remainings
    try:
      array.remove(position[i])
    except:
      pass

    # check if it's already in results
    res = position[i] + move[i] + win_lose
    try:
      result.index(res)
    except:
      # else append
      result.append(res)

# save remaining
with open("results.json", "w") as fp:
  fp.write(json.dumps(array))

np.save("tictactoe2.npy", np.array(result))
