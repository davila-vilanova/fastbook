from enum import Enum
from functools import cache, reduce
from typing import Final, List, Literal, Tuple

import torch
from checks import EmptyInputError, ensure_shape
from fastai.data.external import URLs, untar_data
from fastai.torch_core import tensor
from fastcore.xtras import Path
from image_processing import scale_pixel_values
from PIL import Image
from torch import Tensor, device, stack

Digit = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class DataSplit(Enum):
    TRAINING = "training"
    TESTING = "testing"


IMAGE_SHAPE: Final[Tuple[int, int]] = (28, 28)
NUM_CLASSES: Final[int] = 10


def load_mnist_data() -> Path:
    mnist_path = untar_data(URLs.MNIST)
    print(f"MNIST data downloaded to {mnist_path}")
    return mnist_path


def get_digit_file_paths(
    base_path: Path, digit: Digit, datasplit: DataSplit
) -> List[Path]:
    return list((base_path / datasplit.value / str(digit)).ls().sorted())


@cache
def get_digit_tensors(
    base_path: Path, digit: Digit, datasplit: DataSplit, device: device
) -> List[Tensor]:
    """Gets all tensor images for the given digit in the specified DataSet."""
    return [
        tensor(Image.open(path), device=device)
        for path in get_digit_file_paths(base_path, digit, datasplit)
    ]


def stack_image_tensors(image_tensors: List[Tensor]) -> Tensor:
    """Takes a sequence of MNIST digit image tensors and returns a single tensor
    whose first dimension refers to each of the images."""

    if len(image_tensors) == 0:
        raise EmptyInputError(
            "The input sequence of tensors must have at least one element."
        )

    for t in image_tensors:
        ensure_shape(t, IMAGE_SHAPE)

    # It may be silly to wrap this in a function, but doing so adds some semantics
    # and checks that the input has the expected shape.
    stacked = stack(image_tensors)

    # This should never raise, so it just documents what I'm expecting.
    ensure_shape(stacked, (-1,) + IMAGE_SHAPE)
    return stacked


@cache
def get_stacked_preprocessed_digits(
    base_path: Path,
    digit: Digit,
    datasplit: DataSplit,
    device: device,
) -> Tensor:
    """Gets all images for a given digit in the specified datasplit, stacked in a
    single tensor and normalized."""
    print(
        f"get_stacked_preprocessed_digits is running for digit: {digit}, split: {datasplit}"
    )
    return scale_pixel_values(
        stack_image_tensors(
            get_digit_tensors(
                base_path,
                digit,
                datasplit,
                device,
            )
        )
    )


def labeled_data(
    base_path: Path, datasplit: DataSplit, device: device
) -> Tuple[Tensor, Tensor]:
    """Returns a tuple of tensors:
    - the first contains all the images in the
    dataset for the specified datasplit, stacked, with pixel values scaled, and
    converted to 1D tensors,
    - the second contains the corresponding labels as a vector of integers."""

    _all_digits = range(0, 10)
    _stacked_digits = [
        get_stacked_preprocessed_digits(base_path, digit, datasplit, device)
        for digit in _all_digits
    ]
    _lengths = [
        len(get_digit_file_paths(base_path, digit, datasplit))  # type: ignore
        for digit in _all_digits
    ]

    train_x = torch.cat(_stacked_digits).view(-1, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    train_y = tensor(
        reduce(
            lambda a, b: a + b, [[digit] * _lengths[digit] for digit in _all_digits]
        ),
        dtype=torch.int64,
        device=device,
    )
    return (train_x, train_y)
