from typing import Callable, Generator, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
from checks import ensure_shape
from fastai.data.core import DataLoaders
from model import Module
from torch import Tensor, stack, sum, where, zeros_like


def calculate_accuracy(preds: Tensor, targets: Tensor) -> Tensor:
    """Calculates the accuracy of a matrix of predictions given a matrix of
    hot-encoded targets.

    Args:
        preds (Tensor): A 2D matrix of image index to digit match likelihoods.
        These can be logits or normalized. Just the maximum value of each row is used.
        targets (Tensor): A 2D matrix of targets corresponding to each image

    Returns:
        Tensor: A 0D tensor indicating the average accuracy
    """
    assert preds.ndim == 2
    ensure_shape(targets, (preds.shape[0], preds.shape[1]))

    preds_as_digits = preds.argmax(dim=1)
    targets_as_digits = targets.argmax(dim=1)
    correct = preds_as_digits == targets_as_digits

    return correct.float().mean()


def calculate_loss(preds: Tensor, targets: Tensor) -> Tensor:
    # preds must be normalized
    assert preds.shape == targets.shape

    # TODO is there a more efficient way to do this?
    probability_right_guesses = where(targets == 1.0, preds, zeros_like(preds))
    losses = 1 - probability_right_guesses.sum(dim=1)
    return sum(losses)


class Optimizer:
    def __init__(self, params: Sequence[Tensor], learning_rate: float):
        self.learning_rate = learning_rate
        self.params = params

    def step(self) -> None:
        for p in self.params:
            assert p.grad is not None
            p.data -= p.grad.data * self.learning_rate

    def zero_grad(self) -> None:
        for p in self.params:
            assert p.grad is not None
            p.grad.zero_()  # or set to None


def make_optimizer(params: Sequence[Tensor], learning_rate: float) -> Optimizer:
    return Optimizer(params, learning_rate)


def plot_accuracies(accuracies: Sequence[float]) -> None:
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Training Accuracy Over Epochs")
    plt.show()


class Learner:
    def __init__(
        self,
        dataloaders: DataLoaders,
        model: Module,
        opt_func: Callable[[Iterable[Tensor], float], Optimizer],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        metrics: Callable[[Tensor, Tensor], Tensor],
    ):
        self.dataloaders = dataloaders
        self.model = model
        self.opt_func = opt_func
        self.loss_func = loss_func
        self.metrics = metrics

    def fit(self, epochs: int, lr: float) -> Generator[Tuple[int, float], None, None]:
        train_data, valid_data = self.dataloaders

        optimizer = self.opt_func(self.model.params(), lr)

        for epoch in range(epochs):
            for batch_x, batch_y in train_data:
                logits = self.model(batch_x)
                loss = self.loss_func(logits, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_accuracies = [
                calculate_accuracy(self.model(x), y) for x, y in valid_data
            ]
            # TODO: will this have affected the parameter grads? Significantly?

            yield epoch, stack(epoch_accuracies).mean().item()
