from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from recsys.models.recall.common import dot_logits, last_valid_embedding, masked_average_pooling, masked_softmax
from recsys.models.tensorflow import FeatureEmbeddingEncoder, MLP


class STAMPEncoder(tf.keras.layers.Layer):
    """STAMP short-term attention/memory-priority user encoder."""

    def __init__(self, hidden_dim: int = 64, name: str | None = None) -> None:
        super().__init__(name=name)
        self.sequence_dense = tf.keras.layers.Dense(hidden_dim, name="sequence_dense")
        self.short_dense = tf.keras.layers.Dense(hidden_dim, name="short_dense")
        self.general_dense = tf.keras.layers.Dense(hidden_dim, name="general_dense")
        self.attention_score = tf.keras.layers.Dense(1, name="attention_score")
        self.output_dense = tf.keras.layers.Dense(hidden_dim, activation="tanh", name="stamp_output")

    def call(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Encode session behavior using general memory and latest behavior."""

        general_memory = masked_average_pooling(behavior_embeddings, behavior_mask)
        short_memory = last_valid_embedding(behavior_embeddings, behavior_mask)
        attention_hidden = tf.nn.sigmoid(
            self.sequence_dense(behavior_embeddings)
            + tf.expand_dims(self.short_dense(short_memory), axis=1)
            + tf.expand_dims(self.general_dense(general_memory), axis=1)
        )
        logits = tf.squeeze(self.attention_score(attention_hidden), axis=-1)
        weights = masked_softmax(logits, behavior_mask, axis=-1)
        attentive_memory = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * behavior_embeddings, axis=1)
        return self.output_dense(tf.concat([attentive_memory, short_memory], axis=-1))


class STAMPRecallModel(tf.keras.Model):
    """STAMP recall model for session/short-term video recommendation."""

    def __init__(
        self,
        item_vocab_sizes: Mapping[str, int],
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        item_tower_units: Sequence[int] = (128, 64),
        name: str = "stamp_recall_model",
    ) -> None:
        super().__init__(name=name)
        self.encoder = STAMPEncoder(hidden_dim, name="stamp_encoder")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.item_tower = MLP(item_tower_units, output_units=hidden_dim, name="item_tower")

    def call(self, inputs: Mapping[str, Any], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return STAMP user embedding, item embedding, and recall logit."""

        del training
        user_embedding = self.encode_user(inputs["behavior_embeddings"], inputs.get("behavior_mask"))
        item_embedding = self.encode_item(inputs["item"])
        return {
            "logits": dot_logits(user_embedding, item_embedding),
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
        }

    def encode_user(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Encode sequence into a normalized STAMP user vector."""

        return tf.math.l2_normalize(self.encoder(behavior_embeddings, behavior_mask), axis=-1)

    def encode_item(self, item_features: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Encode item sparse features into normalized recall vectors."""

        return tf.math.l2_normalize(self.item_tower(self.item_encoder(item_features)), axis=-1)


__all__ = [
    "STAMPEncoder",
    "STAMPRecallModel",
]
