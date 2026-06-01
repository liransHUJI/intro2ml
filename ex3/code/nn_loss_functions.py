import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    score = np.sum(y_true == y_pred)
    return score / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,) or (n_samples, input_dim)
            True labels. Can be categorical indices or one-hot encoded vectors.
    y_pred: ndarray of shape (n_samples, input_dim)
        Predicted distribution (softmax) of each sample

    Returns
    -------
    output: float
        Cross entropy of given y_true, y_pred
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = np.finfo(float).eps
    clipped = np.clip(y_pred, eps, 1 - eps)

    if y_true.ndim == 1:
        losses = -np.log(clipped[np.arange(y_true.shape[0]), y_true.astype(int)])
    else:
        losses = -np.sum(y_true * np.log(clipped), axis=1)
    return float(np.mean(losses))


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, input_dim)

    Returns:
    --------
    output: ndarray of shape (n_samples, input_dim)
        Softmax(x) for every sample x in the given data X
    """
    X = np.asarray(X)
    shifted = X - np.max(X, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
