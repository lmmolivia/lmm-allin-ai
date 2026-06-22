from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from recsys.models.recall.common import LabelAwareAttention, dot_logits, masked_softmax
from recsys.models.tensorflow import FeatureEmbeddingEncoder, MLP


class UMAInterestLayer(tf.keras.layers.Layer):
    """UMA-style user memory attention layer for multi-interest recall."""

    def __init__(
        self,
        num_interests: int = 4,
        interest_dim: int = 64,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_interests = num_interests
        self.interest_dim = interest_dim
        self.behavior_projection = tf.keras.layers.Dense(interest_dim, name="behavior_projection")
        self.user_context_gate = tf.keras.layers.Dense(num_interests, activation="sigmoid", name="user_context_gate")
        self.memory_queries: tf.Variable

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create K trainable user-memory queries."""

        self.memory_queries = self.add_weight(
            name="memory_queries",
            shape=(self.num_interests, self.interest_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(
        self,
        behavior_embeddings: tf.Tensor,
        behavior_mask: tf.Tensor | None = None,
        user_context: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Aggregate user behaviors into multiple memory-attended interests."""

        behavior = self.behavior_projection(behavior_embeddings)
        logits = tf.einsum("kd,bsd->bks", self.memory_queries, behavior)
        valid = tf.expand_dims(behavior_mask, axis=1) if behavior_mask is not None else None
        weights = masked_softmax(logits, valid, axis=-1)
        interests = tf.matmul(weights, behavior)
        if user_context is not None:
            gate = tf.expand_dims(self.user_context_gate(user_context), axis=-1)
            interests *= gate
        return interests


class UMARecallModel(tf.keras.Model):
    """UMA-style multi-interest recall model with user memory aggregation."""

    def __init__(
        self,
        item_vocab_sizes: Mapping[str, int],
        num_interests: int = 4,
        embedding_dim: int = 32,
        interest_dim: int = 64,
        item_tower_units: Sequence[int] = (128, 64),
        name: str = "uma_recall_model",
    ) -> None:
        super().__init__(name=name)
        self.interest_layer = UMAInterestLayer(num_interests, interest_dim, name="uma_interests")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.item_tower = MLP(item_tower_units, output_units=interest_dim, name="item_tower")
        self.label_attention = LabelAwareAttention(name="label_aware_attention")

    def call(self, inputs: Mapping[str, Any], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return UMA interests and optional item-aware recall logits."""

        del training
        interests = self.encode_user(
            inputs["behavior_embeddings"],
            inputs.get("behavior_mask"),
            inputs.get("user_context"),
        )
        outputs = {"interest_embeddings": interests}
        if "item" in inputs:
            item_embedding = self.encode_item(inputs["item"])
            user_embedding = self.label_attention(interests, item_embedding)
            outputs.update(
                {
                    "item_embedding": item_embedding,
                    "user_embedding": user_embedding,
                    "interest_logits": tf.einsum("bkd,bd->bk", interests, item_embedding),
                    "logits": dot_logits(user_embedding, item_embedding),
                }
            )
        return outputs

    def encode_user(
        self,
        behavior_embeddings: tf.Tensor,
        behavior_mask: tf.Tensor | None = None,
        user_context: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Encode user history into multiple normalized UMA interests."""

        return tf.math.l2_normalize(self.interest_layer(behavior_embeddings, behavior_mask, user_context), axis=-1)

    def encode_item(self, item_features: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Encode item sparse features into normalized recall vectors."""

        return tf.math.l2_normalize(self.item_tower(self.item_encoder(item_features)), axis=-1)


__all__ = [
    "UMAInterestLayer",
    "UMARecallModel",
]
