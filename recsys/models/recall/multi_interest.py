from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import tensorflow as tf

from recsys.models.recall.common import LabelAwareAttention, dot_logits, masked_softmax, squash_capsules
from recsys.models.tensorflow import FeatureEmbeddingEncoder, MLP


class MINDCapsuleLayer(tf.keras.layers.Layer):
    """MIND dynamic-routing layer that extracts multiple user interests."""

    def __init__(
        self,
        num_interests: int = 4,
        interest_dim: int = 64,
        num_iterations: int = 3,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_interests = num_interests
        self.interest_dim = interest_dim
        self.num_iterations = num_iterations
        self.behavior_projection = tf.keras.layers.Dense(interest_dim, name="behavior_projection")

    def call(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Route behavior embeddings into K capsule-style interest vectors."""

        behavior = self.behavior_projection(behavior_embeddings)
        batch_size = tf.shape(behavior)[0]
        sequence_length = tf.shape(behavior)[1]
        routing_logits = tf.zeros(
            (batch_size, sequence_length, self.num_interests),
            dtype=behavior.dtype,
        )

        if behavior_mask is not None:
            valid = tf.cast(tf.expand_dims(behavior_mask, axis=-1), behavior.dtype)
            behavior *= valid
        else:
            valid = None

        interests = tf.zeros((batch_size, self.num_interests, self.interest_dim), dtype=behavior.dtype)
        for iteration in range(self.num_iterations):
            assignment = tf.nn.softmax(routing_logits, axis=-1)
            if valid is not None:
                assignment *= valid
            interests = squash_capsules(tf.einsum("bsk,bsd->bkd", assignment, behavior))
            if iteration + 1 < self.num_iterations:
                routing_logits += tf.einsum("bsd,bkd->bsk", behavior, interests)
        return interests


class MINDRecallModel(tf.keras.Model):
    """MIND recall model that returns multiple ANN-ready user interest vectors."""

    def __init__(
        self,
        item_vocab_sizes: Mapping[str, int],
        num_interests: int = 4,
        embedding_dim: int = 32,
        interest_dim: int = 64,
        item_tower_units: Sequence[int] = (128, 64),
        name: str = "mind_recall_model",
    ) -> None:
        super().__init__(name=name)
        self.interest_layer = MINDCapsuleLayer(num_interests, interest_dim, name="mind_capsules")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.item_tower = MLP(item_tower_units, output_units=interest_dim, name="item_tower")
        self.label_attention = LabelAwareAttention(name="label_aware_attention")

    def call(self, inputs: Mapping[str, Any], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return MIND interests and optional item-aware logits."""

        del training
        interests = self.encode_user(inputs["behavior_embeddings"], inputs.get("behavior_mask"))
        outputs = {"interest_embeddings": interests}
        if "item" in inputs:
            item_embedding = self.encode_item(inputs["item"])
            selected_user_embedding = self.label_attention(interests, item_embedding)
            outputs.update(
                {
                    "item_embedding": item_embedding,
                    "user_embedding": selected_user_embedding,
                    "interest_logits": tf.einsum("bkd,bd->bk", interests, item_embedding),
                    "logits": dot_logits(selected_user_embedding, item_embedding),
                }
            )
        return outputs

    def encode_user(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Encode user behavior sequence into multiple normalized interests."""

        return tf.math.l2_normalize(self.interest_layer(behavior_embeddings, behavior_mask), axis=-1)

    def encode_item(self, item_features: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Encode item sparse features into normalized recall vectors."""

        return tf.math.l2_normalize(self.item_tower(self.item_encoder(item_features)), axis=-1)


class ComiRecSAInterestLayer(tf.keras.layers.Layer):
    """ComiRec-SA multi-interest extractor based on self-attentive pooling."""

    def __init__(
        self,
        num_interests: int = 4,
        attention_dim: int = 64,
        interest_dim: int = 64,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_interests = num_interests
        self.attention_projection = tf.keras.layers.Dense(attention_dim, activation="tanh", name="attention_projection")
        self.attention_score = tf.keras.layers.Dense(num_interests, name="attention_score")
        self.output_projection = tf.keras.layers.Dense(interest_dim, name="interest_projection")

    def call(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Return K self-attentive interest embeddings from the behavior sequence."""

        attention_hidden = self.attention_projection(behavior_embeddings)
        logits = tf.transpose(self.attention_score(attention_hidden), perm=[0, 2, 1])
        valid = tf.expand_dims(behavior_mask, axis=1) if behavior_mask is not None else None
        weights = masked_softmax(logits, valid, axis=-1)
        interests = tf.matmul(weights, behavior_embeddings)
        return self.output_projection(interests)


class ComiRecRecallModel(tf.keras.Model):
    """ComiRec recall model with SA or dynamic-routing interest extraction."""

    def __init__(
        self,
        item_vocab_sizes: Mapping[str, int],
        mode: Literal["sa", "dr"] = "sa",
        num_interests: int = 4,
        embedding_dim: int = 32,
        interest_dim: int = 64,
        item_tower_units: Sequence[int] = (128, 64),
        name: str = "comirec_recall_model",
    ) -> None:
        super().__init__(name=name)
        self.mode = mode
        if mode == "sa":
            self.interest_layer = ComiRecSAInterestLayer(num_interests, interest_dim=interest_dim, name="comirec_sa")
        elif mode == "dr":
            self.interest_layer = MINDCapsuleLayer(num_interests, interest_dim, name="comirec_dr")
        else:
            raise ValueError(f"Unknown ComiRec mode: {mode}")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.item_tower = MLP(item_tower_units, output_units=interest_dim, name="item_tower")
        self.label_attention = LabelAwareAttention(name="label_aware_attention")

    def call(self, inputs: Mapping[str, Any], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return ComiRec interests and optional positive-item training logits."""

        del training
        interests = self.encode_user(inputs["behavior_embeddings"], inputs.get("behavior_mask"))
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

    def encode_user(self, behavior_embeddings: tf.Tensor, behavior_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Encode behavior sequence into K normalized ComiRec interests."""

        return tf.math.l2_normalize(self.interest_layer(behavior_embeddings, behavior_mask), axis=-1)

    def encode_item(self, item_features: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Encode item sparse features into normalized recall vectors."""

        return tf.math.l2_normalize(self.item_tower(self.item_encoder(item_features)), axis=-1)


__all__ = [
    "ComiRecRecallModel",
    "ComiRecSAInterestLayer",
    "MINDCapsuleLayer",
    "MINDRecallModel",
]
