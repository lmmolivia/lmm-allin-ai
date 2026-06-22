from __future__ import annotations

from typing import Literal, Union

import tensorflow as tf

LearningRate = Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]


def constant_lr(learning_rate: float) -> float:
    """Formula: lr_t = lr.

    Fixed learning rate for simple baselines and small experiments.
    """

    return learning_rate


def exponential_decay(
    initial_learning_rate: float,
    decay_steps: int,
    decay_rate: float,
    staircase: bool = False,
) -> tf.keras.optimizers.schedules.ExponentialDecay:
    """Formula: lr_t = lr_0 * decay_rate^(t / decay_steps).

    Smoothly lowers the learning rate as training progresses.
    """

    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )


def cosine_decay(
    initial_learning_rate: float,
    decay_steps: int,
    alpha: float = 0.0,
) -> tf.keras.optimizers.schedules.CosineDecay:
    """Formula: lr_t = lr_0 * ((1-alpha)*0.5*(1+cos(pi*t/T)) + alpha).

    Cosine annealing schedule often used for deep retrieval/ranking models.
    """

    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
    )


def polynomial_decay(
    initial_learning_rate: float,
    decay_steps: int,
    end_learning_rate: float = 1.0e-5,
    power: float = 1.0,
) -> tf.keras.optimizers.schedules.PolynomialDecay:
    """Formula: lr_t = (lr_0-lr_end)*(1-t/T)^power + lr_end.

    Polynomial schedule for gradual decay to a non-zero final learning rate.
    """

    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        end_learning_rate=end_learning_rate,
        power=power,
    )


def piecewise_constant_decay(
    boundaries: list[int],
    values: list[float],
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Formula: lr_t = values[i] for boundaries[i-1] < t <= boundaries[i].

    Step schedule for manually controlled learning-rate drops.
    """

    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)


def sgd(
    learning_rate: LearningRate = 1.0e-2,
    momentum: float = 0.0,
    nesterov: bool = False,
) -> tf.keras.optimizers.SGD:
    """Formula: v_t = momentum*v_{t-1} + grad; w_t = w_{t-1} - lr*v_t.

    Classic stochastic gradient descent, optionally with momentum/Nesterov.
    """

    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )


def momentum_sgd(
    learning_rate: LearningRate = 1.0e-2,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> tf.keras.optimizers.SGD:
    """Formula: v_t = beta*v_{t-1} + grad; w_t = w_{t-1} - lr*v_t.

    Convenience wrapper for SGD with a non-zero momentum default.
    """

    return sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)


def adam(
    learning_rate: LearningRate = 1.0e-3,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.Adam:
    """Formula: m_t=beta1*m+(1-beta1)*g, v_t=beta2*v+(1-beta2)*g^2.

    Adaptive optimizer that is a strong default for ranking models.
    """

    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )


def adamw(
    learning_rate: LearningRate = 1.0e-3,
    weight_decay: float = 1.0e-4,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.Optimizer:
    """Formula: Adam update plus decoupled weight decay w_t = w_t - lr*wd*w_t.

    AdamW is preferred when weight decay should not be mixed into gradients.
    """

    if not hasattr(tf.keras.optimizers, "AdamW"):
        raise RuntimeError("tf.keras.optimizers.AdamW is not available in this TensorFlow version")
    return tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )


def adagrad(
    learning_rate: LearningRate = 1.0e-2,
    initial_accumulator_value: float = 0.1,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.Adagrad:
    """Formula: w_t = w_{t-1} - lr * g_t / sqrt(G_t + eps).

    Good baseline for sparse ID embeddings because frequent features receive
    smaller effective steps.
    """

    return tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        epsilon=epsilon,
    )


def ftrl(
    learning_rate: LearningRate = 1.0e-2,
    learning_rate_power: float = -0.5,
    initial_accumulator_value: float = 0.1,
    l1_regularization_strength: float = 0.0,
    l2_regularization_strength: float = 0.0,
) -> tf.keras.optimizers.Ftrl:
    """Formula: Follow-The-Regularized-Leader with L1/L2-proximal update.

    Common optimizer for sparse linear or wide recommendation components.
    """

    return tf.keras.optimizers.Ftrl(
        learning_rate=learning_rate,
        learning_rate_power=learning_rate_power,
        initial_accumulator_value=initial_accumulator_value,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
    )


def rmsprop(
    learning_rate: LearningRate = 1.0e-3,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.RMSprop:
    """Formula: v_t = rho*v_{t-1} + (1-rho)*g_t^2; w -= lr*g/sqrt(v+eps).

    Adaptive optimizer using an exponential moving average of squared grads.
    """

    return tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
    )


def adadelta(
    learning_rate: LearningRate = 1.0e-3,
    rho: float = 0.95,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.Adadelta:
    """Formula: update uses RMS(delta_w) / RMS(g) without a fixed global scale.

    Adaptive optimizer that can be useful when tuning learning rate is hard.
    """

    return tf.keras.optimizers.Adadelta(
        learning_rate=learning_rate,
        rho=rho,
        epsilon=epsilon,
    )


def nadam(
    learning_rate: LearningRate = 1.0e-3,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1.0e-7,
) -> tf.keras.optimizers.Nadam:
    """Formula: Adam moments with Nesterov-style momentum lookahead.

    NAdam can converge faster than Adam on some dense neural ranking models.
    """

    return tf.keras.optimizers.Nadam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )


def get_optimizer(
    name: Literal["sgd", "momentum", "adam", "adamw", "adagrad", "ftrl", "rmsprop", "adadelta", "nadam"],
    learning_rate: LearningRate = 1.0e-3,
    **kwargs,
) -> tf.keras.optimizers.Optimizer:
    """Formula: optimizer = registry[name](learning_rate, **kwargs).

    Small factory for config-driven training scripts.
    """

    registry = {
        "sgd": sgd,
        "momentum": momentum_sgd,
        "adam": adam,
        "adamw": adamw,
        "adagrad": adagrad,
        "ftrl": ftrl,
        "rmsprop": rmsprop,
        "adadelta": adadelta,
        "nadam": nadam,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown optimizer: {name}")
    return registry[key](learning_rate=learning_rate, **kwargs)


__all__ = [
    "LearningRate",
    "adadelta",
    "adagrad",
    "adam",
    "adamw",
    "constant_lr",
    "cosine_decay",
    "exponential_decay",
    "ftrl",
    "get_optimizer",
    "momentum_sgd",
    "nadam",
    "piecewise_constant_decay",
    "polynomial_decay",
    "rmsprop",
    "sgd",
]
