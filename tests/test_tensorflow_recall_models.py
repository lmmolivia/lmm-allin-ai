from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from recsys.models.recall import (
    AuxiliaryTwoTowerModel,
    ComiRecRecallModel,
    MINDRecallModel,
    STAMPRecallModel,
    UMARecallModel,
)


ITEM_VOCABS = {"item_id": 20, "category_id": 6}


def item_features():
    return {
        "item_id": tf.constant([1, 2]),
        "category_id": tf.constant([3, 4]),
    }


def behavior_inputs():
    return {
        "behavior_embeddings": tf.ones((2, 4, 6)),
        "behavior_mask": tf.constant([[True, True, True, False], [True, True, False, False]]),
    }


def test_auxiliary_two_tower_forward_shapes() -> None:
    model = AuxiliaryTwoTowerModel(
        user_vocab_sizes={"user_id": 10, "city_id": 5},
        item_vocab_sizes=ITEM_VOCABS,
        embedding_dim=4,
        tower_units=(8,),
        auxiliary_units=(8,),
        output_dim=6,
    )
    outputs = model(
        {
            "user": {
                "user_id": tf.constant([1, 2]),
                "city_id": tf.constant([3, 4]),
            },
            "item": item_features(),
            "behavior_embeddings": tf.ones((2, 4, 6)),
            "behavior_mask": tf.constant([[True, True, False, False], [True, False, False, False]]),
            "auxiliary_dense_features": tf.ones((2, 3)),
        },
        training=False,
    )

    assert tuple(outputs["logits"].shape) == (2,)
    assert tuple(outputs["user_embedding"].shape) == (2, 6)
    assert tuple(outputs["item_embedding"].shape) == (2, 6)


def test_mind_recall_model_forward_shapes() -> None:
    model = MINDRecallModel(
        item_vocab_sizes=ITEM_VOCABS,
        num_interests=3,
        embedding_dim=4,
        interest_dim=6,
        item_tower_units=(8,),
    )
    inputs = {**behavior_inputs(), "item": item_features()}
    outputs = model(inputs, training=False)

    assert tuple(outputs["interest_embeddings"].shape) == (2, 3, 6)
    assert tuple(outputs["interest_logits"].shape) == (2, 3)
    assert tuple(outputs["logits"].shape) == (2,)


@pytest.mark.parametrize("mode", ["sa", "dr"])
def test_comirec_recall_model_forward_shapes(mode: str) -> None:
    model = ComiRecRecallModel(
        item_vocab_sizes=ITEM_VOCABS,
        mode=mode,
        num_interests=3,
        embedding_dim=4,
        interest_dim=6,
        item_tower_units=(8,),
    )
    outputs = model({**behavior_inputs(), "item": item_features()}, training=False)

    assert tuple(outputs["interest_embeddings"].shape) == (2, 3, 6)
    assert tuple(outputs["interest_logits"].shape) == (2, 3)
    assert tuple(outputs["logits"].shape) == (2,)


def test_stamp_recall_model_forward_shapes() -> None:
    model = STAMPRecallModel(
        item_vocab_sizes=ITEM_VOCABS,
        embedding_dim=4,
        hidden_dim=6,
        item_tower_units=(8,),
    )
    outputs = model({**behavior_inputs(), "item": item_features()}, training=False)

    assert tuple(outputs["logits"].shape) == (2,)
    assert tuple(outputs["user_embedding"].shape) == (2, 6)
    assert tuple(outputs["item_embedding"].shape) == (2, 6)


def test_uma_recall_model_forward_shapes() -> None:
    model = UMARecallModel(
        item_vocab_sizes=ITEM_VOCABS,
        num_interests=3,
        embedding_dim=4,
        interest_dim=6,
        item_tower_units=(8,),
    )
    outputs = model(
        {
            **behavior_inputs(),
            "item": item_features(),
            "user_context": tf.ones((2, 5)),
        },
        training=False,
    )

    assert tuple(outputs["interest_embeddings"].shape) == (2, 3, 6)
    assert tuple(outputs["interest_logits"].shape) == (2, 3)
    assert tuple(outputs["logits"].shape) == (2,)
