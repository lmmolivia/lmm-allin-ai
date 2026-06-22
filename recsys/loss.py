from __future__ import annotations

from collections.abc import Mapping

import tensorflow as tf


def binary_crossentropy_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    from_logits: bool = False,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -y log(p) - (1-y) log(1-p).

    Pointwise binary loss for click, like, finish, negative-feedback, and
    other Bernoulli ranking targets.
    """

    labels = tf.cast(labels, predictions.dtype)
    loss = tf.keras.losses.binary_crossentropy(labels, predictions, from_logits=from_logits)
    return reduce_loss(loss, sample_weight)


def weighted_binary_crossentropy_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    positive_weight: float = 1.0,
    from_logits: bool = False,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -w_pos*y*log(p) - (1-y)*log(1-p).

    BCE variant for imbalanced targets, where rare positives such as share or
    follow can receive a larger positive weight.
    """

    labels = tf.cast(labels, predictions.dtype)
    if from_logits:
        loss = tf.nn.weighted_cross_entropy_with_logits(labels, predictions, positive_weight)
    else:
        predictions = tf.clip_by_value(predictions, 1.0e-7, 1.0 - 1.0e-7)
        loss = -positive_weight * labels * tf.math.log(predictions) - (1.0 - labels) * tf.math.log(1.0 - predictions)
    return reduce_loss(loss, sample_weight)


def focal_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -alpha_t * (1-p_t)^gamma * log(p_t).

    Down-weights easy examples and is useful when positive behavior labels are
    sparse or class imbalance is severe.
    """

    labels = tf.cast(labels, predictions.dtype)
    probabilities = tf.sigmoid(predictions) if from_logits else predictions
    probabilities = tf.clip_by_value(probabilities, 1.0e-7, 1.0 - 1.0e-7)
    pt = tf.where(tf.equal(labels, 1.0), probabilities, 1.0 - probabilities)
    alpha_t = tf.where(tf.equal(labels, 1.0), alpha, 1.0 - alpha)
    loss = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return reduce_loss(loss, sample_weight)


def mean_squared_error_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = (y - y_hat)^2.

    Regression loss for watch time, value score, calibrated score, or other
    continuous recommendation labels.
    """

    loss = tf.math.squared_difference(tf.cast(labels, predictions.dtype), predictions)
    return reduce_loss(loss, sample_weight)


def mean_absolute_error_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = |y - y_hat|.

    Robust regression loss that is less sensitive to outliers than MSE.
    """

    loss = tf.abs(tf.cast(labels, predictions.dtype) - predictions)
    return reduce_loss(loss, sample_weight)


def huber_loss(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    delta: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = 0.5*x^2 if |x|<=delta else delta*(|x|-0.5*delta).

    Smoothly combines MSE near zero and MAE for large residuals.
    """

    error = tf.cast(labels, predictions.dtype) - predictions
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + delta * linear
    return reduce_loss(loss, sample_weight)


def pairwise_logistic_loss(
    positive_scores: tf.Tensor,
    negative_scores: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = log(1 + exp(-(s_pos - s_neg))).

    Pairwise ranking loss that teaches positives to score above negatives.
    """

    loss = tf.nn.softplus(-(positive_scores - negative_scores))
    return reduce_loss(loss, sample_weight)


def bpr_loss(
    positive_scores: tf.Tensor,
    negative_scores: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log(sigmoid(s_pos - s_neg)).

    Bayesian Personalized Ranking loss, commonly used for implicit-feedback
    retrieval and ranking.
    """

    loss = -tf.math.log_sigmoid(positive_scores - negative_scores)
    return reduce_loss(loss, sample_weight)


def pairwise_hinge_loss(
    positive_scores: tf.Tensor,
    negative_scores: tf.Tensor,
    margin: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = max(0, margin - s_pos + s_neg).

    Margin-based pairwise loss for enforcing a minimum positive-negative gap.
    """

    loss = tf.nn.relu(margin - positive_scores + negative_scores)
    return reduce_loss(loss, sample_weight)


def listwise_softmax_cross_entropy_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -sum_i softmax(y)_i * log_softmax(logit)_i.

    Listwise ranking loss for training over an ordered candidate set.
    """

    label_distribution = tf.nn.softmax(tf.cast(labels, logits.dtype), axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_distribution, logits=logits)
    return reduce_loss(loss, sample_weight)


def sampled_softmax_recall_loss(
    positive_logits: tf.Tensor,
    negative_logits: tf.Tensor,
    temperature: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log exp(s_pos/tau) / (exp(s_pos/tau) + sum_j exp(s_neg_j/tau)).

    Sampled-softmax retrieval loss for one positive item and many sampled
    negative items per user/query.
    """

    logits = concat_positive_negative_logits(positive_logits, negative_logits) / temperature
    labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return reduce_loss(loss, sample_weight)


def nce_loss(
    positive_logits: tf.Tensor,
    negative_logits: tf.Tensor,
    num_negative_samples: int | None = None,
    positive_sample_prob: tf.Tensor | None = None,
    negative_sample_prob: tf.Tensor | None = None,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = softplus(-(s_pos-log(k*q_pos))) + sum_j softplus(s_neg_j-log(k*q_neg_j)).

    Noise Contrastive Estimation loss. The optional sampling probabilities
    correct logits when negatives are sampled from a non-uniform distribution.
    """

    if num_negative_samples is None:
        num_negative_samples = int(negative_logits.shape[-1])

    positive_logits = apply_sampling_correction(
        positive_logits,
        positive_sample_prob,
        num_negative_samples,
    )
    negative_logits = apply_sampling_correction(
        negative_logits,
        negative_sample_prob,
        num_negative_samples,
    )
    loss = tf.nn.softplus(-positive_logits) + tf.reduce_sum(tf.nn.softplus(negative_logits), axis=-1)
    return reduce_loss(loss, sample_weight)


def mns_loss(
    query_embeddings: tf.Tensor,
    positive_item_embeddings: tf.Tensor,
    negative_item_embeddings: tf.Tensor,
    temperature: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log exp(q*p/tau) / (exp(q*p/tau) + sum_j exp(q*n_j/tau)).

    Multi-Negative Sampling loss for retrieval. Each query has one positive
    item vector and multiple negative item vectors.
    """

    positive_logits = tf.reduce_sum(query_embeddings * positive_item_embeddings, axis=-1)
    negative_logits = tf.einsum("bd,bnd->bn", query_embeddings, negative_item_embeddings)
    return sampled_softmax_recall_loss(positive_logits, negative_logits, temperature, sample_weight)


def multi_negative_sampling_loss(
    positive_logits: tf.Tensor,
    negative_logits: tf.Tensor,
    temperature: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = CE([1,0,...,0], [s_pos, s_neg_1, ..., s_neg_k] / tau).

    Logit-based MNS alias when the model has already computed positive and
    negative scores outside the loss.
    """

    return sampled_softmax_recall_loss(positive_logits, negative_logits, temperature, sample_weight)


def info_nce_loss(
    query_embeddings: tf.Tensor,
    item_embeddings: tf.Tensor,
    temperature: float = 1.0,
    labels: tf.Tensor | None = None,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log exp(q_i*k_i/tau) / sum_j exp(q_i*k_j/tau).

    In-batch contrastive retrieval loss where other positives in the batch act
    as negatives by default.
    """

    logits = tf.matmul(query_embeddings, item_embeddings, transpose_b=True) / temperature
    if labels is None:
        labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return reduce_loss(loss, sample_weight)


def contrastive_learning_loss(
    view_a_embeddings: tf.Tensor,
    view_b_embeddings: tf.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
    symmetric: bool = True,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log exp(sim(z_i^a,z_i^b)/tau) / sum_j exp(sim(z_i^a,z_j^b)/tau).

    Self-supervised CL loss for recommendation embeddings. The two views can
    be user-history augmentations, item-content augmentations, or sequence
    dropout/cropping outputs for the same user/item.
    """

    logits = pairwise_similarity(view_a_embeddings, view_b_embeddings, normalize) / temperature
    labels = tf.range(tf.shape(logits)[0], dtype=tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    if not symmetric:
        return reduce_loss(loss, sample_weight)

    reverse_logits = tf.transpose(logits)
    reverse_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, reverse_logits, from_logits=True)
    return 0.5 * (reduce_loss(loss, sample_weight) + reduce_loss(reverse_loss, sample_weight))


def cl_loss(
    view_a_embeddings: tf.Tensor,
    view_b_embeddings: tf.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
    symmetric: bool = True,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L_CL = InfoNCE(view_a, view_b).

    Short alias for contrastive_learning_loss, useful when adding an auxiliary
    CL term to a recommendation recall or ranking model.
    """

    return contrastive_learning_loss(
        view_a_embeddings,
        view_b_embeddings,
        temperature=temperature,
        normalize=normalize,
        symmetric=symmetric,
        sample_weight=sample_weight,
    )


def explicit_negative_contrastive_loss(
    anchor_embeddings: tf.Tensor,
    positive_embeddings: tf.Tensor,
    negative_embeddings: tf.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = -log exp(sim(a,p)/tau) / (exp(sim(a,p)/tau)+sum_j exp(sim(a,n_j)/tau)).

    CL loss with explicit negatives. In recall training this is useful when the
    negative set comes from exposure-not-click, same-category hard negatives,
    or ANN hard negatives.
    """

    if normalize:
        anchor_embeddings = tf.math.l2_normalize(anchor_embeddings, axis=-1)
        positive_embeddings = tf.math.l2_normalize(positive_embeddings, axis=-1)
        negative_embeddings = tf.math.l2_normalize(negative_embeddings, axis=-1)

    positive_logits = tf.reduce_sum(anchor_embeddings * positive_embeddings, axis=-1)
    negative_logits = tf.einsum("bd,bnd->bn", anchor_embeddings, negative_embeddings)
    return sampled_softmax_recall_loss(
        positive_logits,
        negative_logits,
        temperature=temperature,
        sample_weight=sample_weight,
    )


def supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L_i = -1/|P(i)| * sum_p log exp(sim(z_i,z_p)/tau) / sum_{a!=i} exp(sim(z_i,z_a)/tau).

    Supervised CL loss for recommendation representation learning. Samples
    with the same label, such as same author bucket, category bucket, or
    positive-interest cluster, are treated as positives within the batch.
    """

    logits = pairwise_similarity(embeddings, embeddings, normalize) / temperature
    batch_size = tf.shape(logits)[0]
    labels = tf.reshape(labels, (-1, 1))
    positive_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), logits.dtype)
    self_mask = tf.eye(batch_size, dtype=logits.dtype)
    logits_mask = 1.0 - self_mask
    positive_mask *= logits_mask

    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1.0e-12)

    positive_count = tf.reduce_sum(positive_mask, axis=1)
    valid = tf.cast(positive_count > 0.0, logits.dtype)
    per_example_loss = -tf.reduce_sum(positive_mask * log_prob, axis=1) / tf.maximum(positive_count, 1.0)

    if sample_weight is not None:
        valid *= tf.cast(sample_weight, logits.dtype)
    denominator = tf.maximum(tf.reduce_sum(valid), 1.0)
    return tf.reduce_sum(per_example_loss * valid) / denominator


def sequence_contrastive_loss(
    sequence_view_a_embeddings: tf.Tensor,
    sequence_view_b_embeddings: tf.Tensor,
    temperature: float = 0.1,
    normalize: bool = True,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L_seq = InfoNCE(seq_view_a, seq_view_b).

    Convenience wrapper for user behavior sequence CL, where two augmented
    sequence encoders of the same user should stay close in embedding space.
    """

    return contrastive_learning_loss(
        sequence_view_a_embeddings,
        sequence_view_b_embeddings,
        temperature=temperature,
        normalize=normalize,
        symmetric=True,
        sample_weight=sample_weight,
    )


def in_batch_softmax_loss(
    query_embeddings: tf.Tensor,
    item_embeddings: tf.Tensor,
    temperature: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = CE(arange(batch), QK^T / tau).

    Two-tower retrieval loss that uses items from other rows in the batch as
    negative examples.
    """

    return info_nce_loss(query_embeddings, item_embeddings, temperature, sample_weight=sample_weight)


def triplet_loss(
    anchor_embeddings: tf.Tensor,
    positive_embeddings: tf.Tensor,
    negative_embeddings: tf.Tensor,
    margin: float = 1.0,
    sample_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    """Formula: L = max(0, d(anchor,pos) - d(anchor,neg) + margin).

    Metric-learning loss for pushing positive items closer than negatives.
    """

    positive_distance = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=-1)
    negative_distance = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=-1)
    loss = tf.nn.relu(positive_distance - negative_distance + margin)
    return reduce_loss(loss, sample_weight)


def multi_task_binary_crossentropy_loss(
    labels: Mapping[str, tf.Tensor],
    predictions: Mapping[str, tf.Tensor],
    task_weights: Mapping[str, float] | None = None,
) -> tf.Tensor:
    """Formula: L = sum_t w_t * BCE(y_t, p_t).

    Multi-task loss for click, finish, long-play, short-play, like, share, and
    other task heads.
    """

    task_weights = task_weights or {}
    losses = [
        binary_crossentropy_loss(labels[task], prediction) * task_weights.get(task, 1.0)
        for task, prediction in predictions.items()
    ]
    return tf.add_n(losses)


def concat_positive_negative_logits(positive_logits: tf.Tensor, negative_logits: tf.Tensor) -> tf.Tensor:
    """Formula: logits = concat([s_pos], [s_neg_1, ..., s_neg_k]).

    Helper that normalizes positive logits to shape [batch, 1] before recall
    softmax losses.
    """

    positive_logits = tf.reshape(positive_logits, (-1, 1))
    return tf.concat([positive_logits, negative_logits], axis=1)


def apply_sampling_correction(
    logits: tf.Tensor,
    sample_prob: tf.Tensor | None,
    num_negative_samples: int,
) -> tf.Tensor:
    """Formula: corrected_s = s - log(k * q).

    Corrects sampled logits when the sampler probability q is known.
    """

    if sample_prob is None:
        return logits
    sample_prob = tf.cast(sample_prob, logits.dtype)
    correction = tf.math.log(tf.maximum(sample_prob * float(num_negative_samples), 1.0e-12))
    return logits - correction


def pairwise_similarity(
    left_embeddings: tf.Tensor,
    right_embeddings: tf.Tensor,
    normalize: bool = True,
) -> tf.Tensor:
    """Formula: sim(a,b) = a dot b, or cosine(a,b) after L2 normalization.

    Shared similarity helper for CL and retrieval losses.
    """

    if normalize:
        left_embeddings = tf.math.l2_normalize(left_embeddings, axis=-1)
        right_embeddings = tf.math.l2_normalize(right_embeddings, axis=-1)
    return tf.matmul(left_embeddings, right_embeddings, transpose_b=True)


def reduce_loss(loss: tf.Tensor, sample_weight: tf.Tensor | None = None) -> tf.Tensor:
    """Formula: L = mean(w_i * loss_i) or mean(loss_i).

    Shared reducer so every loss returns a scalar by default.
    """

    if sample_weight is not None:
        loss = loss * tf.cast(sample_weight, loss.dtype)
    return tf.reduce_mean(loss)


__all__ = [
    "apply_sampling_correction",
    "binary_crossentropy_loss",
    "bpr_loss",
    "cl_loss",
    "concat_positive_negative_logits",
    "contrastive_learning_loss",
    "explicit_negative_contrastive_loss",
    "focal_loss",
    "huber_loss",
    "in_batch_softmax_loss",
    "info_nce_loss",
    "listwise_softmax_cross_entropy_loss",
    "mean_absolute_error_loss",
    "mean_squared_error_loss",
    "mns_loss",
    "multi_negative_sampling_loss",
    "multi_task_binary_crossentropy_loss",
    "nce_loss",
    "pairwise_hinge_loss",
    "pairwise_logistic_loss",
    "pairwise_similarity",
    "reduce_loss",
    "sampled_softmax_recall_loss",
    "sequence_contrastive_loss",
    "supervised_contrastive_loss",
    "triplet_loss",
    "weighted_binary_crossentropy_loss",
]
