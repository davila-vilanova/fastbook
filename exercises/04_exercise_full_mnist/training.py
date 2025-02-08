from typing import Callable, Generator, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from checks import ensure_shape
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
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
    def __init__(self, learning_rate: float, params: Sequence[Tensor]):
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


def train_model(
    data: Union[
        DataLoaders, Tuple[Tensor, Tensor, Tensor, Tensor]
    ],  # hot encoded targets
    model: Module,
    *,
    normalizer: Callable[[Tensor], Tensor],
    loss_function: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    epochs: int,
) -> Generator[Tuple[int, float], None, None]:
    def data_iterable(
        data: Union[DataLoader, Tuple[Tensor, Tensor]],
    ) -> Sequence[Tuple[Tensor, Tensor]]:
        """Returns a sequence of tuples of tensors, each tuple containing a batch of
        inputs and a batch of targets.
        If passed a DataLoader, it will return the input value. If passed a tuple, it
        will return a list containing the tuple.
        """
        return data if isinstance(data, DataLoader) else [data]

    print(f"isinstance(data, DataLoaders): {isinstance(data, DataLoaders)}")
    train_data, valid_data = (
        data
        if isinstance(data, DataLoaders)
        else ((data[0], data[1]), (data[2], data[3]))
    )

    def train_epoch(x: Tensor, y: Tensor) -> None:
        forward_pass_calc_grad(
            x,
            y,
            model,
            normalizer,
            loss_function,
        )
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(epochs):
        for batch_x, batch_y in data_iterable(train_data):
            train_epoch(batch_x, batch_y)

        accs = [calculate_accuracy(model(x), y) for x, y in data_iterable(valid_data)]
        # TODO: will this have affected the parameter grads? Significantly?

        yield epoch, stack(accs).mean().item()


# Returns average loss
def forward_pass_calc_grad(
    batch_x: Tensor,
    batch_y: Tensor,
    model: Module,
    normalize: Callable[[Tensor], Tensor],
    loss_function: Callable[[Tensor, Tensor], Tensor],
) -> float:
    logits = model(batch_x)
    preds = normalize(logits)  # TODO: is this built into the loss function?
    loss = loss_function(preds, batch_y)
    loss.backward()
    return loss.item()


def plot_accuracies(accuracies: Sequence[float]) -> None:
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Training Accuracy Over Epochs")
    plt.show()
