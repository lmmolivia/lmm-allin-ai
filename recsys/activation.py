from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf


def sigmoid(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: sigmoid(x) = 1 / (1 + exp(-x)).

    Maps logits to probabilities and is commonly used in binary task towers.
    """

    return tf.nn.sigmoid(inputs)


def tanh(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).

    Zero-centered saturating activation for bounded hidden states.
    """

    return tf.nn.tanh(inputs)


def relu(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: ReLU(x) = max(0, x).

    Default sparse activation for dense hidden layers.
    """

    return tf.nn.relu(inputs)


def leaky_relu(inputs: tf.Tensor, alpha: float = 0.2) -> tf.Tensor:
    """Formula: LeakyReLU(x) = max(x, alpha*x).

    Keeps a small negative slope to reduce dead ReLU units.
    """

    return tf.nn.leaky_relu(inputs, alpha=alpha)


def prelu(inputs: tf.Tensor, alpha: tf.Tensor | float = 0.25) -> tf.Tensor:
    """Formula: PReLU(x) = max(0,x) + alpha*min(0,x).

    Parametric ReLU; pass a trainable alpha tensor if channel-wise slopes are
    needed.
    """

    return tf.maximum(inputs, 0.0) + alpha * tf.minimum(inputs, 0.0)


def elu(inputs: tf.Tensor, alpha: float = 1.0) -> tf.Tensor:
    """Formula: ELU(x) = x if x>0 else alpha*(exp(x)-1).

    Smooth negative branch that can help activations stay closer to zero mean.
    """

    return tf.where(inputs > 0.0, inputs, alpha * tf.math.expm1(inputs))


def selu(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: SELU(x) = scale * ELU(x, alpha).

    Self-normalizing activation used with compatible initialization/dropout.
    """

    return tf.nn.selu(inputs)


def gelu(inputs: tf.Tensor, approximate: bool = True) -> tf.Tensor:
    """Formula: GELU(x) = x * Phi(x).

    Smooth activation widely used in transformer-style networks.
    """

    return tf.nn.gelu(inputs, approximate=approximate)


def swish(inputs: tf.Tensor, beta: float = 1.0) -> tf.Tensor:
    """Formula: Swish(x) = x * sigmoid(beta*x).

    Smooth non-monotonic activation that often works well in deep MLPs.
    """

    return inputs * tf.nn.sigmoid(beta * inputs)


def mish(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: Mish(x) = x * tanh(softplus(x)).

    Smooth non-monotonic activation similar in spirit to Swish.
    """

    return inputs * tf.nn.tanh(tf.nn.softplus(inputs))


def softplus(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: Softplus(x) = log(1 + exp(x)).

    Smooth approximation to ReLU and useful for positive-valued outputs.
    """

    return tf.nn.softplus(inputs)


def softsign(inputs: tf.Tensor) -> tf.Tensor:
    """Formula: Softsign(x) = x / (1 + |x|).

    Bounded activation with polynomial rather than exponential saturation.
    """

    return tf.nn.softsign(inputs)


def softmax(inputs: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Formula: softmax(x_i) = exp(x_i) / sum_j exp(x_j).

    Converts a vector of logits into a categorical distribution.
    """

    return tf.nn.softmax(inputs, axis=axis)


def glu(inputs: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Formula: GLU([a,b]) = a * sigmoid(b).

    Gated Linear Unit; the input dimension on the split axis must be even.
    """

    value, gate = tf.split(inputs, 2, axis=axis)
    return value * tf.nn.sigmoid(gate)


class Dice(tf.keras.layers.Layer):
    """Formula: Dice(x) = p(x)*x + (1-p(x))*alpha*x, p(x)=sigmoid(BN(x)).

    Data Adaptive Activation Function commonly used in DIN/DIEN-style
    recommendation models.
    """

    def __init__(self, epsilon: float = 1.0e-8, name: str | None = None) -> None:
        super().__init__(name=name)
        self.epsilon = epsilon
        self.alpha: tf.Variable

    def build(self, input_shape: tf.TensorShape) -> None:
        """Formula: alpha is learned per last-dimension channel.

        Creates the trainable negative-slope parameter used by Dice.
        """

        dim = int(input_shape[-1])
        self.alpha = self.add_weight(
            name="alpha",
            shape=(dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Formula: p(x)=sigmoid((x-mean)/sqrt(var+eps)).

        Computes a batch-normalized gate and blends x with alpha*x.
        """

        del training
        mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape) - 1)), keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        probability = tf.nn.sigmoid(normalized)
        return probability * inputs + (1.0 - probability) * self.alpha * inputs


class PReLU(tf.keras.layers.Layer):
    """Formula: PReLU(x) = max(0,x) + alpha*min(0,x).

    Keras layer form of PReLU with a learned per-channel alpha.
    """

    def __init__(self, alpha_initializer: str = "zeros", name: str | None = None) -> None:
        super().__init__(name=name)
        self.alpha_initializer = alpha_initializer
        self.alpha: tf.Variable

    def build(self, input_shape: tf.TensorShape) -> None:
        """Formula: alpha has one value per last-dimension channel.

        Creates the trainable negative-slope parameter.
        """

        self.alpha = self.add_weight(
            name="alpha",
            shape=(int(input_shape[-1]),),
            initializer=self.alpha_initializer,
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Formula: y = max(0,x) + alpha*min(0,x).

        Applies the learned piecewise-linear activation.
        """

        return prelu(inputs, self.alpha)


def get_activation(name: str) -> Callable[..., tf.Tensor] | tf.keras.layers.Layer:
    """Formula: activation = registry[name].

    Convenience registry for building model configs from strings.
    """

    registry: dict[str, Callable[..., tf.Tensor] | tf.keras.layers.Layer] = {
        "sigmoid": sigmoid,
        "tanh": tanh,
        "relu": relu,
        "leaky_relu": leaky_relu,
        "prelu": PReLU(),
        "elu": elu,
        "selu": selu,
        "gelu": gelu,
        "swish": swish,
        "mish": mish,
        "softplus": softplus,
        "softsign": softsign,
        "softmax": softmax,
        "glu": glu,
        "dice": Dice(),
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown activation: {name}")
    return registry[key]


__all__ = [
    "Dice",
    "PReLU",
    "elu",
    "gelu",
    "get_activation",
    "glu",
    "leaky_relu",
    "mish",
    "prelu",
    "relu",
    "selu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]
