from typing import Tuple

from torch import Tensor



class EmptyInputError(Exception):
    pass


class TensorShapeError(Exception):
    def __init__(self, message: str, offending_tensor: Tensor):
        self.offending_tensor = offending_tensor
        super().__init__(message)


def ensure_shape(tensor: Tensor, expected: Tuple[int, ...]) -> Tensor:
    """Validates tensor shape matches expected dimensions. Use -1 for 'any'."""
    if len(tensor.shape) != len(expected):
        raise TensorShapeError(
            "Tensor shape and expected shape must have the "
            "same number of dimensions. "
            f"Got {len(tensor.shape)} and {len(expected)}.",
            tensor,
        )
    for actual_dim, expected_dim in zip(tensor.shape, expected):
        if expected_dim == -1:
            continue
        if actual_dim != expected_dim:
            raise TensorShapeError(
                f"Expected shape {expected}, got {tensor.shape}", tensor
            )
    return tensor
