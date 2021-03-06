#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import numpy as np
from keras.models import load_model

from flask import Flask, send_file, request
from gpaulo.game import suggest


# build neural networks
nn = load_model("models/tictactoe.hd5")
nn.predict(np.zeros((1, 9)))
nn.summary()

# HTTP Server
app = Flask(__name__)


@app.route("/")
def index():
    return send_file("static/index.html", mimetype="text/html")


@app.route("/suggest/<id>")
def resolve(id):
    matrix = json.loads(base64.b64decode(id))
    res = suggest(nn, np.array(matrix))
    return json.dumps(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
