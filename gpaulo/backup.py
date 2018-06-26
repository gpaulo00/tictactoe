
import numpy as np
from keras.models import model_from_json

# load neural network from disk
def load(name, out="models"):
  # read json model
  model = None
  with open("%s/%s.json" % (out, name), "r") as fp:
    model = model_from_json(fp.read())

  # load weights
  model.load_weights("%s/%s.h5" % (out, name))

  # compile and return model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.predict(np.random.rand(9).reshape(1,-1))
  print("Loaded '%s' model from disk" % name)
  return model

# backup neural network
def save(model, name, out="models"):
  # serialize model to JSON
  with open("%s/%s.json" % (out, name), "w") as fp:
    fp.write(model.to_json())

  # serialize weights to HDF5
  model.save_weights("%s/%s.h5" % (out, name))
  print("Saved '%s' model to disk" % name)
