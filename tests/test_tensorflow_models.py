from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from recsys.models.tensorflow import DINAttention, DINRankModel, FineRankModel, TwoTowerModel, multi_task_binary_crossentropy


def test_two_tower_model_forward_shapes() -> None:
    model = TwoTowerModel(
        user_vocab_sizes={"user_id": 10, "city_id": 5},
        item_vocab_sizes={"item_id": 20, "category_id": 6},
        embedding_dim=4,
        tower_units=(8,),
        tower_output_dim=6,
    )

    outputs = model(
        {
            "user": {
                "user_id": tf.constant([1, 2]),
                "city_id": tf.constant([3, 4]),
            },
            "item": {
                "item_id": tf.constant([5, 6]),
                "category_id": tf.constant([1, 2]),
            },
        },
        training=False,
    )

    assert tuple(outputs["logits"].shape) == (2,)
    assert tuple(outputs["user_embedding"].shape) == (2, 6)
    assert tuple(outputs["item_embedding"].shape) == (2, 6)


def test_fine_rank_model_forward_shapes_and_loss() -> None:
    model = FineRankModel(
        base_feature_dim=5,
        scene_feature_dim=3,
        personal_feature_dim=4,
        embedding_dim=6,
        tasks=("click", "finish"),
        model_dim=8,
        expert_units=(8,),
        tower_units=(4,),
    )

    outputs = model(
        {
            "dense_features": tf.ones((2, 5)),
            "scene_features": tf.ones((2, 3)),
            "personal_features": tf.ones((2, 4)),
            "target_embedding": tf.ones((2, 6)),
            "behavior_embeddings": tf.ones((2, 3, 6)),
            "behavior_mask": tf.constant([[True, True, False], [True, False, False]]),
        },
        training=False,
    )
    loss = multi_task_binary_crossentropy(
        labels={
            "click": tf.constant([1.0, 0.0]),
            "finish": tf.constant([0.0, 1.0]),
        },
        predictions=outputs,
    )

    assert set(outputs) == {"click", "finish"}
    assert tuple(outputs["click"].shape) == (2,)
    assert tuple(outputs["finish"].shape) == (2,)
    assert loss.shape == ()


def test_din_model_forward_shapes() -> None:
    attention = DINAttention(hidden_units=(8,))
    interest = attention(
        target_embedding=tf.ones((2, 6)),
        behavior_embeddings=tf.ones((2, 3, 6)),
        behavior_mask=tf.constant([[True, True, False], [True, False, False]]),
        training=False,
    )
    model = DINRankModel(
        dense_feature_dim=5,
        embedding_dim=6,
        attention_units=(8,),
        tower_units=(8,),
    )
    predictions = model(
        {
            "dense_features": tf.ones((2, 5)),
            "target_embedding": tf.ones((2, 6)),
            "behavior_embeddings": tf.ones((2, 3, 6)),
            "behavior_mask": tf.constant([[True, True, False], [True, False, False]]),
        },
        training=False,
    )

    assert tuple(interest.shape) == (2, 6)
    assert tuple(predictions.shape) == (2,)
