#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
import numpy as np
# from keras.models import Sequential, model_from_json
# from keras.layers import Dense

import base64
import json
from flask import Flask, send_file
from gpaulo.backup import load


# build neural networks
# model = load("tictactoe")
# win = load("winning")

# get best movement
def bestMove(i, arr):
  one = i.reshape(3,3)
  axis = (None, None)
  f = 0

  for yi, row in enumerate(arr.reshape(3,3)):
    for xi, res in enumerate(row):
      if one[yi][xi] == 0 and res > f:
        axis = (yi, xi)
        f = res
  return axis

# suggest movement
def suggest(arr):
  # check shape
  if arr.shape != (9,):
    raise Exception('invalid input shape')

  # first movement
  if reduce(lambda x,y: x and y, arr == 0):
    return (2,0)

  # return best movement
  return bestMove(arr, model.predict(arr.reshape(1,-1))[0].reshape(3,3))

# HTTP Server
app = Flask(__name__)

@app.route('/')
def index():
  return send_file('static/index.html', mimetype='text/html')

# @app.route("/suggest/<id>")
# def resolve(id):
#   matrix = json.loads(base64.b64decode(id))
#   res = suggest(np.array(matrix))
#   return json.dumps(res)

# @app.route("/win/<id>")
# def resolve(id):
#   matrix = np.array(json.loads(base64.b64decode(id)))
#   res = win.predict(matrix.reshape(1,-1))
#   return json.dumps(res[0][0])

boards = json.dumps(np.loadtxt("board_positions.csv").tolist())
@app.route("/data")
def matrixData():
  return boards

# :9 (board), 9:18 (move), 18: (win, lose)
results = np.load("tictactoe.npy").tolist()
@app.route("/results/<data>")
def saveResults(data):
  i = json.loads(base64.b64decode(data))
  results.append(i)
  return json.dumps({ "ok": True })

@app.route("/save")
def checkpoint():
  np.save("tictactoe", np.array(results))
  return json.dumps({ "ok": True })

store = None
@app.route("/storage/<data>")
def storage(data):
  store = json.loads(base64.b64decode(data))
  return json.dumps({ "ok": True })

@app.route("/storage")
def getStorage():
  return json.dumps(store)

if __name__ == '__main__':
  app.run(host= '0.0.0.0')
