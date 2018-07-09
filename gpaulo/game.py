#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce


def bestMove(initial, predict):
    # variables
    axis = (None, None)
    f = 0

    # check availability
    for idx, res in enumerate(predict):
        if initial[idx] == 0 and res > f:
            axis = (idx // 3, idx % 3)
            f = res
    return axis


def suggest(nn, arr):
    # check shape
    if arr.shape != (9,):
        raise Exception('invalid input shape')

    # first movement
    if np.alltrue(arr == 0):
        return (2, 0)

    # return best movement
    i = arr.reshape(1, -1)
    res = nn.predict(i)[0]
    return bestMove(arr, res)
