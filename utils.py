import numpy as np
import tensorly as tl


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
