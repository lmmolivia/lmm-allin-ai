"""Recall model package grouped by algorithm family."""

from recsys.models.recall.common import (
    LabelAwareAttention,
    dot_logits,
    last_valid_embedding,
    masked_average_pooling,
    masked_softmax,
    squash_capsules,
)
from recsys.models.recall.multi_interest import (
    ComiRecRecallModel,
    ComiRecSAInterestLayer,
    MINDCapsuleLayer,
    MINDRecallModel,
)
from recsys.models.recall.stamp import STAMPEncoder, STAMPRecallModel
from recsys.models.recall.two_tower import AuxiliaryTwoTowerModel, TwoTowerWithAuxiliaryModel
from recsys.models.recall.uma import UMAInterestLayer, UMARecallModel


def describe_recall_model_inputs() -> dict[str, object]:
    """Document common input dictionaries for recall models."""

    return {
        "AuxiliaryTwoTowerModel": {
            "user": "dict[str, int Tensor] user sparse features",
            "item": "dict[str, int Tensor] item sparse features",
            "behavior_embeddings": "optional [batch, sequence_length, dim]",
            "behavior_mask": "optional [batch, sequence_length], True means valid behavior",
            "auxiliary_dense_features": "optional [batch, dense_dim]",
        },
        "MIND/ComiRec/UMA": {
            "behavior_embeddings": "[batch, sequence_length, dim]",
            "behavior_mask": "optional [batch, sequence_length], True means valid behavior",
            "item": "optional dict[str, int Tensor] positive item features",
        },
        "STAMPRecallModel": {
            "behavior_embeddings": "[batch, sequence_length, dim]",
            "behavior_mask": "optional [batch, sequence_length], True means valid behavior",
            "item": "dict[str, int Tensor] positive item features",
        },
    }


__all__ = [
    "AuxiliaryTwoTowerModel",
    "ComiRecRecallModel",
    "ComiRecSAInterestLayer",
    "LabelAwareAttention",
    "MINDCapsuleLayer",
    "MINDRecallModel",
    "STAMPEncoder",
    "STAMPRecallModel",
    "TwoTowerWithAuxiliaryModel",
    "UMAInterestLayer",
    "UMARecallModel",
    "describe_recall_model_inputs",
    "dot_logits",
    "last_valid_embedding",
    "masked_average_pooling",
    "masked_softmax",
    "squash_capsules",
]
