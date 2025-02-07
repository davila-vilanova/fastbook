from torch import Tensor, log_softmax, softmax


def normalize_softmax(logits: Tensor) -> Tensor:
    """Normalizes logits applying softmax.

    Returns:
        Tensor: A tensor of predictions with the same dimensions as the input.
        Predictions sum 1, and equal to the probability of each class match.
    """
    return softmax(logits, dim=1)


def normalize_log_softmax(logits: Tensor) -> Tensor:
    """Normalizes logits applying log softmax.

    Returns:
        Tensor: A tensor of predictions with the same dimensions as the input.
        Predictions sum 1, and equal to the probability of each class match.
    """
    return log_softmax(logits, dim=1)
