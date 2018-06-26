#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense

MODELS = "models"
NAME = "winning"

# backup neural network
def backup(model, name):
    # serialize model to JSON
    with open("%s/%s.json" % (MODELS, name), "w") as fp:
        fp.write(model.to_json())

    # serialize weights to HDF5
    model.save_weights("%s/%s.h5" % (MODELS, name))
    print("Saved '%s' model to disk" % name)

# load data
data = np.loadtxt('winning.csv')
X = data[:, :9]
Y = data[:, 9]

# build neural network
model = Sequential()
model.add(Dense(9, activation='relu', input_dim=9))
model.add(Dense(13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=500, batch_size=50)

# test and backup
scores = model.evaluate(X, Y)
print("neural network %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
backup(model, NAME)

print(model.predict(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]).reshape(1,-1)))
