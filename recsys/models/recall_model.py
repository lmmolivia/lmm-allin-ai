"""Compatibility imports for recall models.

New code should import from `recsys.models.recall` or one of its submodules.
This file stays as a thin shim so older imports do not break.
"""

from recsys.models.recall import (
    AuxiliaryTwoTowerModel,
    ComiRecRecallModel,
    ComiRecSAInterestLayer,
    LabelAwareAttention,
    MINDCapsuleLayer,
    MINDRecallModel,
    STAMPEncoder,
    STAMPRecallModel,
    TwoTowerWithAuxiliaryModel,
    UMAInterestLayer,
    UMARecallModel,
    describe_recall_model_inputs,
    dot_logits,
    last_valid_embedding,
    masked_average_pooling,
    masked_softmax,
    squash_capsules,
)

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
