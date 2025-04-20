import numpy as np
import tensorly as tl


class CMTF():

    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def fit(tensors, matrices):
        """ Fit a coupled matrix tensor factorisation using ALS.
        
        Parameters
        ----------
        tensors:
            A list of tensors. This X in the report.
        matrices:
            A list of matrices. This is Y in the report.
        
        Returns
        -------
        TBD
        """
        # criteria:
        # min || X - [[A, B, C]] ||  + || Z - [[A, B, C]] ||  + || Y - A \Sigma V^T||

        # init
        # random or SVD for the martrix ??

        # optim loop
        # 
        pass