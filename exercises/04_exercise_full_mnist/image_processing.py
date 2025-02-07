from torch import Tensor


def scale_pixel_values(input: Tensor) -> Tensor:
    """Takes a tensor of image data with pixel values between 0 and 255
    and returns a tensor with float values between 0 and 1"""
    return input.float() / 255
