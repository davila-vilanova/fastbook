from typing import Optional, Tuple, Union

from torch import Tensor, device, randn, max
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, Sequence
from checks import ensure_shape
from fastai.torch_core import tensor


def init_params(
    shape: Union[int, Tuple[int, int]], device: device, std: float = 1.0
) -> Tensor:
    return (randn(shape, device=device) * std).requires_grad_()


# TODO: I could add a different kind of param initializer here such as Kaiming


class Module(ABC):  # e.g. a model or layer
    @abstractmethod
    def run(self, input: Tensor) -> Tensor:
        pass

    @abstractmethod
    def params(self) -> Sequence[Tensor]:
        pass

    def __call__(self, *args, **kwargs):  # type: ignore
        return self.run(*args, **kwargs)

    def verify_input(self, input: Tensor, in_features: int) -> Tensor:
        return ensure_shape(input, (-1, in_features))

    def verify_output(self, output: Tensor, out_features: int) -> Tensor:
        return ensure_shape(output, (-1, out_features))


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_params_function: Callable[
            [Union[int, Tuple[int, int]], device], Tensor
        ],  # takes shape and device, returns tensor
        device: device,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = init_params_function((in_features, out_features), device)
        self.biases = init_params_function(out_features, device)

    def run(self, input: Tensor) -> Tensor:
        self.verify_input(input, self.in_features)
        # Apply linear transformation: input (e.g. batch) @ weights + biases
        linear = input @ self.weights + self.biases
        return self.verify_output(
            linear, self.out_features
        )  # return unnormalized logits

    def params(self) -> Sequence[Tensor]:
        return [self.weights, self.biases]


class ReLU(Module):
    def __init__(self, device: device):
        self.device = device

    def run(self, input: Tensor) -> Tensor:
        return max(input, tensor(0.0, device=self.device))  # type: ignore

    def params(self) -> Sequence[Tensor]:
        return []


class Sequential(Module):
    def __init__(self, submodules: Sequence[Module]):
        self.submodules = submodules

    def run(self, input: Tensor) -> Tensor:
        # Call all the submodules in order, calling the first with the input,
        # passing the output of each to the input of the next one, and returning
        # the output of the last one
        assert len(self.submodules) > 0
        i = input
        o: Optional[Tensor] = None
        for submodule in self.submodules:
            if o is not None:
                i = o
            o = submodule(i)

        assert o is not None, "Loop should have run at least once"
        return o

    def params(self) -> Sequence[Tensor]:
        return list(chain.from_iterable([sub.params() for sub in self.submodules]))
