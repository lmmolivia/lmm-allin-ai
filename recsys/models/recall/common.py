from __future__ import annotations

import tensorflow as tf


def masked_average_pooling(
    sequence_embeddings: tf.Tensor,
    sequence_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """Average valid user behavior embeddings into one history vector."""

    if sequence_mask is None:
        return tf.reduce_mean(sequence_embeddings, axis=1)

    valid = tf.cast(sequence_mask, sequence_embeddings.dtype)
    valid = tf.expand_dims(valid, axis=-1)
    summed = tf.reduce_sum(sequence_embeddings * valid, axis=1)
    count = tf.maximum(tf.reduce_sum(valid, axis=1), 1.0)
    return summed / count


def last_valid_embedding(
    sequence_embeddings: tf.Tensor,
    sequence_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """Return the latest valid behavior embedding for short-term interest."""

    if sequence_mask is None:
        return sequence_embeddings[:, -1, :]

    valid = tf.cast(sequence_mask, tf.int32)
    lengths = tf.reduce_sum(valid, axis=1)
    last_index = tf.maximum(lengths - 1, 0)
    batch_index = tf.range(tf.shape(sequence_embeddings)[0], dtype=tf.int32)
    gathered = tf.gather_nd(sequence_embeddings, tf.stack([batch_index, last_index], axis=1))
    has_history = tf.cast(tf.expand_dims(lengths > 0, axis=-1), sequence_embeddings.dtype)
    return gathered * has_history


def masked_softmax(
    logits: tf.Tensor,
    valid_mask: tf.Tensor | None,
    axis: int,
) -> tf.Tensor:
    """Softmax over valid positions and keep invalid positions at zero weight."""

    if valid_mask is None:
        return tf.nn.softmax(logits, axis=axis)

    mask = tf.cast(valid_mask, tf.bool)
    while len(mask.shape) < len(logits.shape):
        mask = tf.expand_dims(mask, axis=1)
    masked_logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1.0e9))
    weights = tf.nn.softmax(masked_logits, axis=axis)
    weights *= tf.cast(mask, weights.dtype)
    normalizer = tf.maximum(tf.reduce_sum(weights, axis=axis, keepdims=True), 1.0e-12)
    return weights / normalizer


def squash_capsules(capsules: tf.Tensor, epsilon: float = 1.0e-9) -> tf.Tensor:
    """Squash capsule vectors so longer vectors represent stronger interests."""

    squared_norm = tf.reduce_sum(tf.square(capsules), axis=-1, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * capsules / tf.sqrt(squared_norm + epsilon)


def dot_logits(user_embeddings: tf.Tensor, item_embeddings: tf.Tensor) -> tf.Tensor:
    """Compute dot-product logits for recall retrieval."""

    return tf.reduce_sum(user_embeddings * item_embeddings, axis=-1)


class LabelAwareAttention(tf.keras.layers.Layer):
    """Select the user interest vector most relevant to the positive item."""

    def __init__(self, power_factor: float = 10.0, hard_select: bool = False, name: str | None = None) -> None:
        super().__init__(name=name)
        self.power_factor = power_factor
        self.hard_select = hard_select

    def call(self, interest_embeddings: tf.Tensor, target_item_embedding: tf.Tensor) -> tf.Tensor:
        """Return the item-aware user embedding used for sampled-softmax training."""

        scores = tf.einsum("bkd,bd->bk", interest_embeddings, target_item_embedding)
        if self.hard_select:
            best_index = tf.argmax(scores, axis=1, output_type=tf.int32)
            batch_index = tf.range(tf.shape(scores)[0], dtype=tf.int32)
            return tf.gather_nd(interest_embeddings, tf.stack([batch_index, best_index], axis=1))

        weights = tf.nn.softmax(scores * self.power_factor, axis=1)
        return tf.reduce_sum(tf.expand_dims(weights, axis=-1) * interest_embeddings, axis=1)


__all__ = [
    "LabelAwareAttention",
    "dot_logits",
    "last_valid_embedding",
    "masked_average_pooling",
    "masked_softmax",
    "squash_capsules",
]
