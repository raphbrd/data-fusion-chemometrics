import os.path as op
import pickle
import logging
import numpy as np
import tensorly as tl
import tqdm


class CMTF:
    """ Coupled matrix-tensor factorisation using ALS. """

    def __init__(self, tol=1e-8, max_iter=100, verbose=True):
        """ Initialise the CMTF object.

        Parameters
        ----------
        tol: float
            The tolerance for the convergence criterion.
        max_iter: int
            The maximum number of iterations.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.deltas = []  # to track the convergence criteria

    def fit(self, tensors, matrices, rank, lmbda=1):
        """ Fit a coupled matrix tensor factorisation using ALS. It is assumed that the mode-0 is shared between
        the tensor and the matrix

        Denoting n_t the number of tensors and n_m the number of matrices, the optimization criterion is:
            min || X - [[U_1, U_2, U_3]] ||  + || M - U_1 V_^T||

        Parameters
        ----------
        tensors: list of np.ndarray
            A list of third-order tensors of shape (n1, n2_i, n3_i).
        matrices: list of np.ndarray
            A list of matrices of shape (n1, q_i).
        rank: int
            The rank of the decomposition.
        lmbda: int
            The coupling parameter (the lambda in the optim criterion)

        Returns
        -------
        U1: np.ndarray
            The common matrix of the decomposition.
        U2: np.ndarray
            The second matrix of the decomposition.
        U3: np.ndarray
            The third matrix of the decomposition.
        P: np.ndarray
            The coupling matrix of the decomposition.
        """
        self._check_params(tensors, matrices)
        n_tensors = len(tensors)
        n_matrices = len(matrices)
        n1 = tensors[0].shape[0]
        self.logger.info(
            f"Starting CMTF fit for rank = {rank}, lmbda = {lmbda} with {n_tensors} tensors and {n_matrices} matrices"
        )

        # initialize the matrices
        # each tensor X_i is modeled by [[U_1, U_2_i, U_3_i]]
        U1 = np.random.rand(n1, rank)
        U2s = [np.random.rand(tensor.shape[1], rank) for tensor in tensors]
        U3s = [np.random.rand(tensor.shape[2], rank) for tensor in tensors]
        # each matrix has a P matrix factor
        Ps = [np.random.rand(matrix.shape[1], rank) for matrix in matrices]

        X1s = [tl.unfold(tensor, mode=0) for tensor in tensors]
        X2s = [tl.unfold(tensor, mode=1) for tensor in tensors]
        X3s = [tl.unfold(tensor, mode=2) for tensor in tensors]

        # to avoid numerical instability, adding small value to matrices diagonal
        eps = 1e-8 * np.eye(rank)
        prev_loss = np.inf

        pbar = tqdm.tqdm(range(self.max_iter), total=self.max_iter, desc="CMTF fit [delta U1 = 0]")
        for it in pbar:
            # update U1 (common matrix) w.r.t. to each tensor
            # We write the solution of the least squares problem as:
            # U_1^\hat = G^{-1} @ H <=> G @ U^T = H^T
            # thus we use np.linalg.solve for convenience and numerical stability
            G = np.zeros((rank, rank))
            H = np.zeros((n1, rank))
            for i in range(n_tensors):
                U2, U3 = U2s[i], U3s[i]
                kr_prod = tl.tenalg.khatri_rao([U2, U3])
                G += (U3.T @ U3) * (U2.T @ U2)
                H += X1s[i] @ kr_prod
            for j in range(n_matrices):
                P = Ps[j]
                G += lmbda * (P.T @ P)
                H += lmbda * (matrices[j] @ P)
            U1 = np.linalg.solve(G + eps, H.T).T

            # update U2, U3, ... for each tensor
            for i in range(n_tensors):
                U2, U3 = U2s[i], U3s[i]
                X2, X3 = X2s[i], X3s[i]
                kr_prod = tl.tenalg.khatri_rao([U1, U3])
                G = (U3.T @ U3) * (U1.T @ U1) + eps
                U2s[i] = np.linalg.solve(G, (X2 @ kr_prod).T).T

                kr_prod = tl.tenalg.khatri_rao([U1, U2])
                G = (U2.T @ U2) * (U1.T @ U1) + eps
                U3s[i] = np.linalg.solve(G, (X3 @ kr_prod).T).T

            # update all matrix factors
            inv_U1_U1 = np.linalg.inv(U1.T @ U1 + eps)
            for j in range(n_matrices):
                matrix = matrices[j]
                Ps[j] = matrix.T @ U1 @ inv_U1_U1

            # check relative reconstructed error change for early stopping
            current_loss = self.compute_iteration_loss(tensors, matrices, U1, U2s, U3s, Ps)
            if it > 0:
                delta = abs(current_loss - prev_loss) / prev_loss
            else:
                delta = float("inf")
            prev_loss = current_loss
            self.deltas.append(delta)
            if self.verbose and (it + 1) % 100 == 0:
                pbar.set_description(f"CMTF fit [delta Loss = {delta:.8f}]")
            if delta < self.tol:
                pbar.close()
                self.logger.info(f"Early stopping, convergence reached after {it + 1} iterations, delta = {delta:.8f}.")
                break

        return U1, U2s, U3s, Ps

    @staticmethod
    def compute_iteration_loss(tensors, matrices, U1, U2_list, U3_list, Ps):
        """ Compute the convergence criteria for the CMTF algorithm at a given iteration."""
        loss = 0
        for i in range(len(tensors)):
            # tensorly expect a tuple (weights, factors)
            weights = np.ones(U1.shape[1])
            factors = [U1, U2_list[i], U3_list[i]]
            X_reconstructed = tl.cp_to_tensor((weights, factors))
            loss += tl.norm(tensors[i] - X_reconstructed) ** 2
        for j in range(len(matrices)):
            loss += tl.norm(matrices[j] - U1 @ Ps[j].T) ** 2

        return loss

    def save_fit(self, fit_filename, fit, deltas_filename=None):
        """ Save the fit to a file. """
        with open(fit_filename, "wb") as f:
            pickle.dump(fit, f)
        self.logger.info(f"Fit saved to {fit_filename}")
        if deltas_filename is not None:
            with open(deltas_filename, "wb") as f:
                pickle.dump(self.deltas, f)
            self.logger.info(f"Deltas saved to {deltas_filename}")

    def load_fit(self, fit_filename):
        """ Load the fit from a file. """
        if not op.exists(fit_filename):
            raise FileNotFoundError(
                f"File {fit_filename} does not exist. Please check the path or fit the decomposition first"
            )

        with open(fit_filename, "rb") as f:
            fit = pickle.load(f)
        self.logger.info(f"Fit loaded from {fit_filename}")
        return fit

    @staticmethod
    def _check_params(tensors, matrices):
        # checking for correct type
        if not isinstance(tensors, list):
            raise TypeError("The tensors must be a list of tensors")

        if not isinstance(matrices, list):
            raise TypeError("The matrices must be a list of tensors")

        for tensor in tensors:
            if np.isnan(tensor).sum() > 0:
                raise ValueError("The tensors must not contain any NaN values.")
        for matrix in matrices:
            if np.isnan(matrix).sum() > 0:
                raise ValueError("The matrices must not contain any NaN values.")

        # checking for correct and homogeneous dimensions
        n1 = tensors[0].shape[0]
        for tensor in tensors[1:]:
            if tensor.shape[0] != n1:
                raise ValueError("The first dimension of the common mode of all tensors must be equal.")
            if tensor.ndim != 3:
                raise ValueError("The tensors must be 3D arrays (third-order tensors).")

        for matrix in matrices:
            if matrix.shape[0] != n1:
                raise ValueError("The dimension of the common mode of the matrices must match the tensors.")
            if matrix.ndim != 2:
                raise ValueError("The matrices must be 2D arrays.")
