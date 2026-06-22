from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from recsys.activation import Dice, gelu, get_activation, glu, mish, swish
from recsys.loss import (
    binary_crossentropy_loss,
    bpr_loss,
    cl_loss,
    contrastive_learning_loss,
    explicit_negative_contrastive_loss,
    focal_loss,
    info_nce_loss,
    mns_loss,
    nce_loss,
    pairwise_hinge_loss,
    sampled_softmax_recall_loss,
    sequence_contrastive_loss,
    supervised_contrastive_loss,
)
from recsys.optimizer import adam, cosine_decay, ftrl, get_optimizer


def test_recall_losses_return_scalars() -> None:
    positive_logits = tf.constant([2.0, 1.5])
    negative_logits = tf.constant([[0.1, -0.2, 0.3], [0.2, -0.1, 0.0]])
    query = tf.ones((2, 4))
    positive = tf.ones((2, 4))
    negative = tf.zeros((2, 3, 4))

    assert sampled_softmax_recall_loss(positive_logits, negative_logits).shape == ()
    assert nce_loss(positive_logits, negative_logits).shape == ()
    assert mns_loss(query, positive, negative).shape == ()
    assert info_nce_loss(query, positive).shape == ()


def test_contrastive_losses_return_scalars() -> None:
    view_a = tf.ones((4, 6))
    view_b = tf.concat([tf.ones((2, 6)), tf.zeros((2, 6))], axis=0)
    labels = tf.constant([1, 1, 2, 2])
    negatives = tf.zeros((4, 3, 6))

    assert contrastive_learning_loss(view_a, view_b).shape == ()
    assert cl_loss(view_a, view_b).shape == ()
    assert explicit_negative_contrastive_loss(view_a, view_b, negatives).shape == ()
    assert supervised_contrastive_loss(view_a, labels).shape == ()
    assert sequence_contrastive_loss(view_a, view_b).shape == ()


def test_ranking_losses_return_scalars() -> None:
    labels = tf.constant([1.0, 0.0, 1.0])
    predictions = tf.constant([0.9, 0.1, 0.7])
    positive_scores = tf.constant([2.0, 1.0])
    negative_scores = tf.constant([0.5, 0.2])

    assert binary_crossentropy_loss(labels, predictions).shape == ()
    assert focal_loss(labels, predictions).shape == ()
    assert bpr_loss(positive_scores, negative_scores).shape == ()
    assert pairwise_hinge_loss(positive_scores, negative_scores).shape == ()


def test_activation_shapes() -> None:
    inputs = tf.ones((2, 4))
    gated_inputs = tf.ones((2, 8))

    assert swish(inputs).shape == inputs.shape
    assert mish(inputs).shape == inputs.shape
    assert gelu(inputs).shape == inputs.shape
    assert glu(gated_inputs).shape == inputs.shape
    assert Dice()(inputs).shape == inputs.shape
    assert get_activation("relu")(inputs).shape == inputs.shape


def test_optimizer_factories() -> None:
    schedule = cosine_decay(0.01, decay_steps=100)
    assert adam(schedule).__class__.__name__ == "Adam"
    assert ftrl(0.01).__class__.__name__ == "Ftrl"
    assert get_optimizer("adam", learning_rate=0.001).__class__.__name__ == "Adam"
