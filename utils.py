import numpy as np
import tensorly as tl
from scipy.optimize import linear_sum_assignment
import os
import os.path as op


def frob(_x):
    """ Compute the Frobenius norm of a tensor or matrix."""
    if len(_x.shape) == 2:
        return np.linalg.norm(_x, ord="fro")
    elif len(_x.shape) == 3:
        return tl.norm(_x)


def replace_nan(tensor, strategy="zero"):
    """ Replace undefined values in a tensor with a specific strategy.

    Parameters
    ----------
    tensor: np.ndarray
        The tensor to process.
    strategy: str
        The strategy to use. Can be "zero" to replace nan with 0 or "mean" to replace by the mean of the tensor.
    """
    nan_mask = np.isnan(tensor)
    if strategy == "zero":
        tensor[nan_mask] = 0.0
    elif strategy == "mean":
        tensor[nan_mask] = np.nanmean(tensor)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Only 'zero' and 'mean' are currently supported.")
    return tensor


def normalize_cp(cp):
    """ Normalize components from CMTF to unit length """
    return tl.cp_normalize(([None] * len(cp), cp))


def normalize_vector(vector):
    """ Normalize a vector to unit length."""
    vector = np.array(vector)
    if vector.ndim > 1:
        raise ValueError("Input must be a vector.")
    return vector / np.linalg.norm(vector)


def normalize_pos_factors(matrix):
    """ Normalize a factor matrix to unit length and take the absolute value.
     This is done by normalizing each column of the matrix."""
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    return np.abs(matrix / np.linalg.norm(matrix, axis=0, keepdims=True))


def match_score(hat_tensor, gt_tensor, return_indices=False):
    """ Compute the match score as defined in Acar et al. (2014). This is the cosine similarity between
    the estimated tensor and the ground truth, after finding the best permutation match.

    Parameters
    ----------
    hat_tensor: np.ndarray
        The estimated tensor.
    gt_tensor: np.ndarray
        The ground truth tensor.
    return_indices: bool
        If True, return the indices of the best permutation match.

    Returns
    -------
    match_score: float
        The match score.
    """
    hat_tensor_norm = np.linalg.norm(hat_tensor, axis=0, keepdims=True)
    gt_tensor_norm = np.linalg.norm(gt_tensor, axis=0, keepdims=True)

    # dot_products = np.sum(hat_tensor * gt_tensor, axis=0)
    cosine_similarity = hat_tensor.T @ gt_tensor / (hat_tensor_norm * gt_tensor_norm)

    row_ind, col_ind = linear_sum_assignment(np.abs(cosine_similarity), maximize=True)

    match_scores = cosine_similarity[row_ind, col_ind]

    if return_indices:
        return match_scores, row_ind, col_ind

    return match_scores


def initialize_paths():
    """ Initialize the paths for the output data and figures directories. """
    output_data_dir = "./output_data"
    output_figures_dir = "./figures"

    if not op.exists(output_data_dir):
        os.makedirs(output_data_dir)
    if not op.exists(output_figures_dir):
        os.makedirs(output_figures_dir)

    return output_data_dir, output_figures_dir
