#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by H.Turbé, February 2022.
	Scaler for Time Series. 
"""

import math
import numpy as np
import pandas as pd

class TS_Scaler:
    def __init__(self, method="standard", df_stats = None):
        self.method = method
        if self.method not in ["standard", "minmax", "mean"]:
            raise ValueError("Unknown method for scaler")
        self.fitted = False

        if df_stats is not None:
            self.M = df_stats["mean"].values
            self.S = df_stats["s"].values
            self.n = df_stats["n"][0]
            self.min = df_stats["min"]
            self.max = df_stats["max"]
            self.fitted = True
            # self.std = df_stats["std"]

        else:
            self.n = 0
            self.M = 0
            self.S = 0

            self.min = np.array(None)
            self.max = np.array(None)

    def update(self, x):
        """
        self.n += 1

        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)

        self.M = newM
        self.S = newS
        """

        # if np.max(np.abs(x))>3*self.max and (self.max !=0):
        #     return True

        self.n += x.shape[0]

        newM = self.M + np.sum((x - self.M), axis=0) / self.n
        newS = self.S + np.sum((x - self.M) * (x - newM), axis=0)

        self.M = newM
        self.S = newS

        if (np.all(self.max == None)) & (np.all(self.min == None)):
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)

        else:
            self.min = np.min(np.stack((self.min, np.min(x, axis=0))), axis=0)
            self.max = np.max(np.stack((self.max, np.max(x, axis=0))), axis=0)
        
        return True

    def fit(self, reader):
        count =0
        for x in reader:
            if (x[0].shape[0] == 0):
                continue
            count +=1
            self.update(x[0].astype(np.float32))


        self.fitted = True

        dict_stat = {
            "mean": self.M,
            "s": self.S,
            "n": self.n,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }

        return pd.DataFrame(dict_stat)

    def transform(self, X):
        type_input = X.dtype
        X = X.astype(np.float32)
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        if self.method == "standard":
            return ((X - self.M) / self.std).astype(type_input)
        # elif self.method == 'minmax':
        # X_std = (X - self.min) / (self.max - self.min)
        # X_scaled = X_std * (max - min) + min
        # return (X - self.M) / (self.S + 1e-8)
        # elif self.method == 'mean':
        #     return X - self.M
        else:
            raise ValueError("Unknown method")

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.n == 1:
            return 0
        return np.sqrt(self.S / (self.n - 1))
