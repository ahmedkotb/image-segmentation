#!/usr/bin/env python

import numpy as np
import numpy.linalg as la

class PCA:
    def __init__(self,inputData):
        data = inputData.copy()
        #m = no of points
        #n = no of features per point
        self.m = data.shape[0]
        self.n = data.shape[1]
        #mean center the data
        data -= np.mean(data,axis=0)

        # calculate the covariance matrix
        c = np.cov(data, rowvar=0)

        # get the eigenvalues/eigenvectors of c
        eval, evec = la.eig(c)
        # u = eigen vectors (transposed)
        self.u = evec.transpose()

    def getPCA(self,vector,components):
        if components > self.n:
            raise Exception("components must be > 0 and <= n")
        return np.dot(self.u[0:components,:],vector)
