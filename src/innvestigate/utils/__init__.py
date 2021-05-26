from __future__ import annotations

import math
from typing import Callable, List, Tuple, TypeVar, Union

import keras.backend as K
import keras.utils as kutils
from tensorflow import Tensor

from innvestigate.utils.keras.graph import model_wo_softmax

__all__ = [
    "model_wo_softmax",
    "to_list",
    "unpack_singleton",
    "BatchSequence",
    "TargetAugmentedSequence",
    "preprocess_images",
    "postprocess_images",
]


T = TypeVar("T")  # Generic type, can be anything


def to_list(X: Union[T, List[T]]) -> List[T]:
    """Wraps tensor `X` into a list, if it isn't a list of Tensors yet."""
    if isinstance(X, list):
        return X
    return [X]


def unpack_singleton(x: List[T]) -> Union[T, List[T]]:
    """Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x


###############################################################################


class BatchSequence(kutils.Sequence):
    """Batch sequence generator.

    Take a (list of) input tensors and a batch size
    and creates a generators that creates a sequence of batches.

    :param Xs: One or a list of tensors. First axis needs to have same length.
    :param batch_size: Batch size. Default 32.
    """

    def __init__(self, Xs: Union[Tensor, List[Tensor]], batch_size: int = 32) -> None:
        self.Xs: List[Tensor] = to_list(Xs)
        self.single_tensor: bool = len(Xs) == 1
        self.batch_size: int = batch_size

        if not self.single_tensor:
            for X in self.Xs[1:]:
                assert X.shape[0] == self.Xs[0].shape[0]
        super().__init__()

    def __len__(self) -> int:
        return int(math.ceil(float(len(self.Xs[0])) / self.batch_size))

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor]]:
        ret: List[Tensor] = [
            X[idx * self.batch_size : (idx + 1) * self.batch_size] for X in self.Xs
        ]

        if self.single_tensor:
            return ret[0]
        return tuple(ret)


class TargetAugmentedSequence(kutils.Sequence):
    """Augments a sequence with a target on the fly.

    Takes a sequence/generator and a function that
    creates on the fly for each batch a target.
    The generator takes a batch from that sequence,
    computes the target and returns both.

    :param sequence: A sequence or generator.
    :param augment_f: Takes a batch and returns a target.
    """

    def __init__(self, sequence: List, augment_f: Callable) -> None:
        self.sequence = sequence
        self.augment_f = augment_f

        super().__init__()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx: int):
        inputs = self.sequence[idx]
        if isinstance(inputs, tuple):
            assert len(inputs) == 1
            inputs = inputs[0]

        targets = self.augment_f(to_list(inputs))
        return inputs, targets


###############################################################################


def preprocess_images(images: Tensor, color_coding: str = None) -> Tensor:
    """Image preprocessing

    Takes a batch of images and:
    * Adjust the color axis to the Keras format.
    * Fixes the color coding.

    :param images: Batch of images with 4 axes.
    :param color_coding: Determines the color coding.
      Can be None, 'RGBtoBGR' or 'BGRtoRGB'.
    :return: The preprocessed batch.
    """

    ret: Tensor = images
    image_data_format: str = K.image_data_format()

    # TODO: not very general:
    channels_first: bool = images.shape[1] in [1, 3]
    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 3, 1, 2)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 2, 3, 1)

    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    return ret


def postprocess_images(
    images: Tensor, color_coding: str = None, channels_first: bool = None
) -> Tensor:
    """Image postprocessing

    Takes a batch of images and reverts the preprocessing.

    :param images: A batch of images with 4 axes.
    :param color_coding: The initial color coding,
      see :func:`preprocess_images`.
    :param channels_first: The output channel format.
    :return: The postprocessed images.
    """

    ret: Tensor = images
    image_data_format: str = K.image_data_format()

    assert color_coding in [None, "RGBtoBGR", "BGRtoRGB"]
    if color_coding in ["RGBtoBGR", "BGRtoRGB"]:
        if image_data_format == "channels_first":
            ret = ret[:, ::-1, :, :]
        if image_data_format == "channels_last":
            ret = ret[:, :, :, ::-1]

    if image_data_format == "channels_first" and not channels_first:
        ret = ret.transpose(0, 2, 3, 1)
    if image_data_format == "channels_last" and channels_first:
        ret = ret.transpose(0, 3, 1, 2)

    return ret
