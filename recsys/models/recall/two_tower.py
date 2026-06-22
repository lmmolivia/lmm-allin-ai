from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from recsys.models.recall.common import dot_logits, masked_average_pooling
from recsys.models.tensorflow import FeatureEmbeddingEncoder, MLP


class AuxiliaryTwoTowerModel(tf.keras.Model):
    """Two-tower recall model with an auxiliary user-interest tower.

    The main user tower models stable profile features. The auxiliary tower can
    consume behavior sequence pooling and extra dense features, then a gate
    fuses main and auxiliary user embeddings before matching the item tower.
    """

    def __init__(
        self,
        user_vocab_sizes: Mapping[str, int],
        item_vocab_sizes: Mapping[str, int],
        embedding_dim: int = 32,
        tower_units: Sequence[int] = (128, 64),
        auxiliary_units: Sequence[int] = (128, 64),
        output_dim: int = 64,
        l2_normalize: bool = True,
        name: str = "auxiliary_two_tower_model",
    ) -> None:
        super().__init__(name=name)
        self.l2_normalize = l2_normalize
        self.user_encoder = FeatureEmbeddingEncoder(user_vocab_sizes, embedding_dim, name="user_encoder")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.user_tower = MLP(tower_units, output_units=output_dim, name="user_tower")
        self.item_tower = MLP(tower_units, output_units=output_dim, name="item_tower")
        self.auxiliary_tower = MLP(auxiliary_units, output_units=output_dim, name="auxiliary_tower")
        self.fusion_gate = tf.keras.layers.Dense(output_dim, activation="sigmoid", name="fusion_gate")

    def call(self, inputs: Mapping[str, Any], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return fused user embedding, item embedding, and recall logits."""

        user_embedding, auxiliary_embedding = self.encode_user(inputs, training=training)
        item_embedding = self.encode_item(inputs["item"], training=training)
        return {
            "logits": dot_logits(user_embedding, item_embedding),
            "auxiliary_logits": dot_logits(auxiliary_embedding, item_embedding),
            "user_embedding": user_embedding,
            "auxiliary_embedding": auxiliary_embedding,
            "item_embedding": item_embedding,
        }

    def encode_user(self, inputs: Mapping[str, Any], training: bool | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        """Encode user sparse features and optional auxiliary behavior signals."""

        base_embedding = self.user_tower(self.user_encoder(inputs["user"]), training=training)
        auxiliary_inputs = []
        if "behavior_embeddings" in inputs:
            auxiliary_inputs.append(masked_average_pooling(inputs["behavior_embeddings"], inputs.get("behavior_mask")))
        if "auxiliary_dense_features" in inputs:
            auxiliary_inputs.append(inputs["auxiliary_dense_features"])

        if auxiliary_inputs:
            auxiliary_embedding = self.auxiliary_tower(tf.concat(auxiliary_inputs, axis=-1), training=training)
        else:
            auxiliary_embedding = tf.zeros_like(base_embedding)

        gate = self.fusion_gate(tf.concat([base_embedding, auxiliary_embedding], axis=-1))
        fused_embedding = gate * base_embedding + (1.0 - gate) * auxiliary_embedding
        return self._normalize(fused_embedding), self._normalize(auxiliary_embedding)

    def encode_item(self, item_features: Mapping[str, tf.Tensor], training: bool | None = None) -> tf.Tensor:
        """Encode item sparse features into the recall item vector."""

        return self._normalize(self.item_tower(self.item_encoder(item_features), training=training))

    def _normalize(self, embedding: tf.Tensor) -> tf.Tensor:
        """Optionally L2-normalize recall vectors for cosine ANN retrieval."""

        if not self.l2_normalize:
            return embedding
        return tf.math.l2_normalize(embedding, axis=-1)


TwoTowerWithAuxiliaryModel = AuxiliaryTwoTowerModel


__all__ = [
    "AuxiliaryTwoTowerModel",
    "TwoTowerWithAuxiliaryModel",
]
