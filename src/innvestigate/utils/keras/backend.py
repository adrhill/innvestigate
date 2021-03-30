from __future__ import annotations

import keras.backend as K

import tensorflow

# TODO: remove this file -A.

__all__ = [
    "to_floatx",
    "gradients",
    "is_not_finite",
    "extract_conv2d_patches",
    "gather",
    "gather_nd",
]


def to_floatx(x):
    return K.cast(x, K.floatx())


def gradients(Xs, Ys, known_Ys):
    """Partial derivatives

    Computes the partial derivatives between Ys and Xs and
    using the gradients for Ys known_Ys.

    :param Xs: List of input tensors.
    :param Ys: List of output tensors that depend on Xs.
    :param known_Ys: Gradients for Ys.
    :return: Gradients for Xs given known_Ys
    """
    return tensorflow.gradients(Ys, Xs, grad_ys=known_Ys, stop_gradients=Xs)


def is_not_finite(x):
    """Checks if tensor x is finite, if not throws an exception."""
    # x = tensorflow.check_numerics(x, "innvestigate - is_finite check")
    return tensorflow.logical_not(tensorflow.is_finite(x))


def extract_conv2d_patches(x, kernel_shape, strides, rates, padding):
    """Extracts conv2d patches like TF function extract_image_patches.

    :param x: Input image.
    :param kernel_shape: Shape of the Keras conv2d kernel.
    :param strides: Strides of the Keras conv2d layer.
    :param rates: Dilation rates of the Keras conv2d layer.
    :param padding: Paddings of the Keras conv2d layer.
    :return: The extracted patches.
    """
    if K.image_data_format() == "channels_first":
        x = K.permute_dimensions(x, (0, 2, 3, 1))
    kernel_shape = [1, kernel_shape[0], kernel_shape[1], 1]
    strides = [1, strides[0], strides[1], 1]
    rates = [1, rates[0], rates[1], 1]
    ret = tensorflow.extract_image_patches(
        x, kernel_shape, strides, rates, padding.upper()
    )

    if K.image_data_format() == "channels_first":
        # TODO: check if we need to permute again.xs
        pass
    return ret


def gather(x, axis, indices):
    """TensorFlow's gather."""
    return tensorflow.gather(x, indices, axis=axis)


def gather_nd(x, indices):
    """TensorFlow's gather_nd."""
    return tensorflow.gather_nd(x, indices)
