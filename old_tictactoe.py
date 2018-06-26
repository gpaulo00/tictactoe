#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np

import base64
import json
from flask import Flask, send_file

# load neural network from disk
def load(name):
  # read json model
  model = None
  with open("models/%s.json" % name, "r") as fp:
    model = model_from_json(fp.read())

  # load weights
  model.load_weights("models/%s.h5" % name)

  # compile and return model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.predict(np.random.rand(9).reshape(1,-1))
  print("Loaded '%s' model from disk" % name)
  return model

# neural networks
ctx = load("ttt_context")
surv = load("ttt_survival")
reg = load("ttt_strategy")

# get best movement
def bestMove(i, arr):
  one = i.reshape(3,3)
  axis = (None, None)
  f = 0

  # discard used cells
  for yi, row in enumerate(arr.reshape(3,3)):
    for xi, res in enumerate(row):
      if one[yi][xi] == 0 and res > f:
        axis = (yi, xi)
        f = res
  return axis

# suggest a move
def suggest(arr):
  # check shape
  if arr.shape != (3,3):
    raise Exception('invalid input shape')
  inp = arr.reshape(9)

  # first movement
  if reduce(lambda x,y: x and y, inp == 0):
    return (2,0)

  # get context
  c = ctx.predict(inp.reshape(1,-1))[0]
  print("context: ", c)
  c1 = max(c)
  
  # danger!
  if c1 == c[0]:
    # Remove "1"s, Convert -1 to 1
    print("danger!")
    i = np.minimum(inp, 0) * -1
    return bestMove(arr, surv.predict(i.reshape(1,-1))[0].reshape(3,3))

  # go to win!
  elif c1 == c[1]:
    # Remove "-1"s
    print("go to win!")
    i = np.maximum(inp, 0)
    return bestMove(arr, surv.predict(i.reshape(1,-1))[0].reshape(3,3))

  # regular move (not implemented)
  else:
    print("regular")
    return bestMove(arr, reg.predict(inp.reshape(1,-1))[0].reshape(3,3))

# HTTP Server
app = Flask(__name__)

@app.route('/')
def index():
  return send_file('static/index.html', mimetype='text/html')

@app.route("/suggest/<id>")
def resolve(id):
  dec = json.loads(base64.b64decode(id))
  res = suggest(np.array(dec).reshape(3,3))
  return json.dumps(res)

if __name__ == '__main__':
  app.run(host= '0.0.0.0')
