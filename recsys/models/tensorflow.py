from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    """Small reusable multilayer perceptron block."""

    def __init__(
        self,
        hidden_units: Sequence[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        output_units: int | None = None,
        output_activation: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.net: list[tf.keras.layers.Layer] = []
        for units in hidden_units:
            self.net.append(tf.keras.layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                self.net.append(tf.keras.layers.Dropout(dropout_rate))
        if output_units is not None:
            self.net.append(tf.keras.layers.Dense(output_units, activation=output_activation))

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Run the dense stack on a batch of features."""

        output = inputs
        for layer in self.net:
            if isinstance(layer, tf.keras.layers.Dropout):
                output = layer(output, training=training)
            else:
                output = layer(output)
        return output


class FeatureEmbeddingEncoder(tf.keras.layers.Layer):
    """Embed scalar sparse features and concatenate their embeddings."""

    def __init__(
        self,
        vocab_sizes: Mapping[str, int],
        embedding_dim: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.feature_names = tuple(vocab_sizes)
        self.embeddings = {
            feature: tf.keras.layers.Embedding(vocab_size, embedding_dim, name=f"{feature}_embedding")
            for feature, vocab_size in vocab_sizes.items()
        }

    def call(self, features: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Return one concatenated vector for all configured sparse features."""

        vectors = []
        for feature in self.feature_names:
            values = tf.cast(features[feature], tf.int32)
            embedding = self.embeddings[feature](values)
            vectors.append(tf.reshape(embedding, (tf.shape(embedding)[0], -1)))
        return tf.concat(vectors, axis=-1)


class CrossNetwork(tf.keras.layers.Layer):
    """Deep & Cross Network v1 layer for explicit feature crossing."""

    def __init__(self, num_layers: int = 2, name: str | None = None) -> None:
        super().__init__(name=name)
        self.num_layers = num_layers
        self.kernels: list[tf.Variable] = []
        self.biases: list[tf.Variable] = []

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create one vector kernel and bias per cross layer."""

        dim = int(input_shape[-1])
        for index in range(self.num_layers):
            self.kernels.append(
                self.add_weight(
                    name=f"kernel_{index}",
                    shape=(dim, 1),
                    initializer="glorot_uniform",
                    trainable=True,
                )
            )
            self.biases.append(
                self.add_weight(
                    name=f"bias_{index}",
                    shape=(dim,),
                    initializer="zeros",
                    trainable=True,
                )
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply x_{l+1} = x0 * (x_l w_l) + b_l + x_l."""

        x0 = inputs
        x = inputs
        for kernel, bias in zip(self.kernels, self.biases):
            cross = tf.matmul(x, kernel)
            x = x0 * cross + bias + x
        return x


class EPNet(tf.keras.layers.Layer):
    """Scene-aware gate that reweights ranking features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_units: Sequence[int] = (64, 32),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.gate = MLP(hidden_units, output_units=feature_dim, output_activation="sigmoid")

    def call(self, features: tf.Tensor, scene_features: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Scale features by a gate computed from scene/request features."""

        gate = self.gate(scene_features, training=training)
        return features * (2.0 * gate)


class PPNet(tf.keras.layers.Layer):
    """Personalized gate that reweights ranking features by user context."""

    def __init__(
        self,
        feature_dim: int,
        hidden_units: Sequence[int] = (64, 32),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.gate = MLP(hidden_units, output_units=feature_dim, output_activation="sigmoid")

    def call(self, features: tf.Tensor, personal_features: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Scale features by a gate computed from personalized user features."""

        gate = self.gate(personal_features, training=training)
        return features * (2.0 * gate)


class SIMLayer(tf.keras.layers.Layer):
    """Target-aware attention over long user behavior sequences."""

    def __init__(
        self,
        attention_dim: int,
        value_dim: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.attention_dim = attention_dim
        self.value_dim = value_dim or attention_dim
        self.query = tf.keras.layers.Dense(attention_dim, use_bias=False)
        self.key = tf.keras.layers.Dense(attention_dim, use_bias=False)
        self.value = tf.keras.layers.Dense(self.value_dim, use_bias=False)

    def call(
        self,
        target_embedding: tf.Tensor,
        behavior_embeddings: tf.Tensor,
        behavior_mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Attend to relevant behaviors; behavior_mask True means valid history."""

        query = tf.expand_dims(self.query(target_embedding), axis=1)
        key = self.key(behavior_embeddings)
        value = self.value(behavior_embeddings)
        logits = tf.reduce_sum(query * key, axis=-1) / tf.sqrt(tf.cast(self.attention_dim, tf.float32))

        if behavior_mask is not None:
            valid_mask = tf.cast(behavior_mask, tf.bool)
            logits = tf.where(valid_mask, logits, tf.fill(tf.shape(logits), -1.0e9))

        weights = tf.nn.softmax(logits, axis=-1)
        output = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * value, axis=1)
        if behavior_mask is not None:
            has_history = tf.reduce_any(valid_mask, axis=1, keepdims=True)
            output = tf.where(has_history, output, tf.zeros_like(output))
        return output


class DINAttention(tf.keras.layers.Layer):
    """Deep Interest Network attention over user behavior sequences."""

    def __init__(
        self,
        hidden_units: Sequence[int] = (80, 40),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.attention_mlp = MLP(hidden_units, output_units=1, name="din_attention_mlp")

    def call(
        self,
        target_embedding: tf.Tensor,
        behavior_embeddings: tf.Tensor,
        behavior_mask: tf.Tensor | None = None,
        training: bool | None = None,
    ) -> tf.Tensor:
        """Return target-aware behavior interest; behavior_mask True means valid history."""

        sequence_length = tf.shape(behavior_embeddings)[1]
        target = tf.tile(tf.expand_dims(target_embedding, axis=1), [1, sequence_length, 1])
        attention_input = tf.concat(
            [
                target,
                behavior_embeddings,
                target - behavior_embeddings,
                target * behavior_embeddings,
            ],
            axis=-1,
        )
        logits = tf.squeeze(self.attention_mlp(attention_input, training=training), axis=-1)

        if behavior_mask is not None:
            valid_mask = tf.cast(behavior_mask, tf.bool)
            logits = tf.where(valid_mask, logits, tf.fill(tf.shape(logits), -1.0e9))

        weights = tf.nn.softmax(logits, axis=-1)
        output = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * behavior_embeddings, axis=1)
        if behavior_mask is not None:
            has_history = tf.reduce_any(valid_mask, axis=1, keepdims=True)
            output = tf.where(has_history, output, tf.zeros_like(output))
        return output


class MMoE(tf.keras.layers.Layer):
    """Multi-gate mixture-of-experts layer for multi-task learning."""

    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
        expert_units: Sequence[int],
        expert_output_dim: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.experts = [
            MLP(expert_units, output_units=expert_output_dim, name=f"expert_{index}")
            for index in range(num_experts)
        ]
        self.gates = [
            tf.keras.layers.Dense(num_experts, activation="softmax", name=f"task_gate_{index}")
            for index in range(num_tasks)
        ]

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> list[tf.Tensor]:
        """Return one expert mixture representation per task."""

        expert_outputs = tf.stack([expert(inputs, training=training) for expert in self.experts], axis=1)
        task_outputs = []
        for gate in self.gates:
            weights = gate(inputs)
            task_outputs.append(tf.reduce_sum(expert_outputs * tf.expand_dims(weights, axis=-1), axis=1))
        return task_outputs


class CGCLayer(tf.keras.layers.Layer):
    """Customized Gate Control layer from PLE/CGC multi-task modeling."""

    def __init__(
        self,
        num_tasks: int,
        num_shared_experts: int,
        num_task_experts: int,
        expert_units: Sequence[int],
        expert_output_dim: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_tasks = num_tasks
        self.shared_experts = [
            MLP(expert_units, output_units=expert_output_dim, name=f"shared_expert_{index}")
            for index in range(num_shared_experts)
        ]
        self.task_experts = [
            [
                MLP(expert_units, output_units=expert_output_dim, name=f"task_{task}_expert_{index}")
                for index in range(num_task_experts)
            ]
            for task in range(num_tasks)
        ]
        gate_width = num_shared_experts + num_task_experts
        self.task_gates = [
            tf.keras.layers.Dense(gate_width, activation="softmax", name=f"cgc_gate_{task}")
            for task in range(num_tasks)
        ]

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> list[tf.Tensor]:
        """Mix shared experts with task-specific experts for each task."""

        shared_outputs = [expert(inputs, training=training) for expert in self.shared_experts]
        task_representations = []
        for task_index in range(self.num_tasks):
            task_outputs = [
                expert(inputs, training=training)
                for expert in self.task_experts[task_index]
            ]
            all_outputs = tf.stack(task_outputs + shared_outputs, axis=1)
            weights = self.task_gates[task_index](inputs)
            task_representations.append(tf.reduce_sum(all_outputs * tf.expand_dims(weights, axis=-1), axis=1))
        return task_representations


class TwoTowerModel(tf.keras.Model):
    """U2I recall model with separate user and item towers."""

    def __init__(
        self,
        user_vocab_sizes: Mapping[str, int],
        item_vocab_sizes: Mapping[str, int],
        embedding_dim: int = 32,
        tower_units: Sequence[int] = (128, 64),
        tower_output_dim: int = 64,
        l2_normalize: bool = True,
        name: str = "two_tower_model",
    ) -> None:
        super().__init__(name=name)
        self.l2_normalize = l2_normalize
        self.user_encoder = FeatureEmbeddingEncoder(user_vocab_sizes, embedding_dim, name="user_encoder")
        self.item_encoder = FeatureEmbeddingEncoder(item_vocab_sizes, embedding_dim, name="item_encoder")
        self.user_tower = MLP(tower_units, output_units=tower_output_dim, name="user_tower")
        self.item_tower = MLP(tower_units, output_units=tower_output_dim, name="item_tower")

    def call(self, inputs: Mapping[str, Mapping[str, tf.Tensor]], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return user/item embeddings and their dot-product logits."""

        user_embedding = self.encode_user(inputs["user"], training=training)
        item_embedding = self.encode_item(inputs["item"], training=training)
        logits = tf.reduce_sum(user_embedding * item_embedding, axis=-1)
        return {
            "logits": logits,
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
        }

    def encode_user(self, user_features: Mapping[str, tf.Tensor], training: bool | None = None) -> tf.Tensor:
        """Encode sparse user features into an ANN-ready user vector."""

        embedding = self.user_tower(self.user_encoder(user_features), training=training)
        return self._normalize(embedding)

    def encode_item(self, item_features: Mapping[str, tf.Tensor], training: bool | None = None) -> tf.Tensor:
        """Encode sparse item features into an ANN-indexed item vector."""

        embedding = self.item_tower(self.item_encoder(item_features), training=training)
        return self._normalize(embedding)

    def _normalize(self, embedding: tf.Tensor) -> tf.Tensor:
        """Optionally L2-normalize tower outputs for cosine retrieval."""

        if not self.l2_normalize:
            return embedding
        return tf.math.l2_normalize(embedding, axis=-1)


class DINRankModel(tf.keras.Model):
    """DIN rank model for target-aware user behavior interest modeling."""

    def __init__(
        self,
        dense_feature_dim: int,
        embedding_dim: int,
        attention_units: Sequence[int] = (80, 40),
        tower_units: Sequence[int] = (128, 64),
        name: str = "din_rank_model",
    ) -> None:
        super().__init__(name=name)
        self.attention = DINAttention(attention_units, name="din_attention")
        self.tower = MLP(tower_units, output_units=1, output_activation="sigmoid", name="din_tower")
        self._dense_feature_dim = dense_feature_dim
        self._embedding_dim = embedding_dim

    def call(self, inputs: Mapping[str, tf.Tensor], training: bool | None = None) -> tf.Tensor:
        """Return click/rank probability from dense features and DIN interest."""

        dense_features = inputs["dense_features"]
        target_embedding = inputs["target_embedding"]
        behavior_embeddings = inputs["behavior_embeddings"]
        behavior_mask = inputs.get("behavior_mask")

        interest = self.attention(
            target_embedding,
            behavior_embeddings,
            behavior_mask=behavior_mask,
            training=training,
        )
        features = tf.concat([dense_features, target_embedding, interest], axis=-1)
        return tf.squeeze(self.tower(features, training=training), axis=-1)


class FineRankModel(tf.keras.Model):
    """Multi-task fine-rank model: DCN + SIM + EPNet + PPNet + CGC + towers."""

    def __init__(
        self,
        base_feature_dim: int,
        scene_feature_dim: int,
        personal_feature_dim: int,
        embedding_dim: int,
        tasks: Sequence[str] = ("click", "finish", "long_play", "short_play", "like", "share", "follow", "negative"),
        model_dim: int = 128,
        cross_layers: int = 2,
        expert_units: Sequence[int] = (128, 64),
        tower_units: Sequence[int] = (64, 32),
        name: str = "fine_rank_model",
    ) -> None:
        super().__init__(name=name)
        self.tasks = tuple(tasks)
        self.input_projection = tf.keras.layers.Dense(model_dim, activation="relu", input_shape=(base_feature_dim + embedding_dim * 2,))
        self.cross = CrossNetwork(cross_layers)
        self.epnet = EPNet(model_dim, name="epnet")
        self.ppnet = PPNet(model_dim, name="ppnet")
        self.sim = SIMLayer(attention_dim=embedding_dim, value_dim=embedding_dim, name="sim")
        self.cgc = CGCLayer(
            num_tasks=len(self.tasks),
            num_shared_experts=2,
            num_task_experts=1,
            expert_units=expert_units,
            expert_output_dim=model_dim,
            name="cgc",
        )
        self.task_towers = {
            task: MLP(tower_units, output_units=1, output_activation="sigmoid", name=f"{task}_tower")
            for task in self.tasks
        }
        self._base_feature_dim = base_feature_dim
        self._scene_feature_dim = scene_feature_dim
        self._personal_feature_dim = personal_feature_dim

    def call(self, inputs: Mapping[str, tf.Tensor], training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return one probability tensor per ranking task."""

        dense_features = inputs["dense_features"]
        scene_features = inputs["scene_features"]
        personal_features = inputs["personal_features"]
        target_embedding = inputs["target_embedding"]
        behavior_embeddings = inputs["behavior_embeddings"]
        behavior_mask = inputs.get("behavior_mask")

        sim_output = self.sim(target_embedding, behavior_embeddings, behavior_mask)
        features = tf.concat([dense_features, target_embedding, sim_output], axis=-1)
        features = self.input_projection(features)
        features = self.cross(features)
        features = self.epnet(features, scene_features, training=training)
        features = self.ppnet(features, personal_features, training=training)

        task_representations = self.cgc(features, training=training)
        outputs = {}
        for task, representation in zip(self.tasks, task_representations):
            outputs[task] = tf.squeeze(self.task_towers[task](representation, training=training), axis=-1)
        return outputs


def multi_task_binary_crossentropy(
    labels: Mapping[str, tf.Tensor],
    predictions: Mapping[str, tf.Tensor],
    task_weights: Mapping[str, float] | None = None,
) -> tf.Tensor:
    """Compute weighted binary cross-entropy across task heads."""

    task_weights = task_weights or {}
    losses = []
    for task, prediction in predictions.items():
        label = tf.cast(labels[task], prediction.dtype)
        loss = tf.keras.losses.binary_crossentropy(label, prediction)
        losses.append(tf.reduce_mean(loss) * task_weights.get(task, 1.0))
    return tf.add_n(losses)


def sampled_softmax_recall_loss(
    positive_logits: tf.Tensor,
    negative_logits: tf.Tensor,
) -> tf.Tensor:
    """Contrast one positive recall logit against sampled negative logits."""

    logits = tf.concat([tf.expand_dims(positive_logits, axis=1), negative_logits], axis=1)
    labels = tf.zeros(tf.shape(positive_logits)[0], dtype=tf.int32)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


def describe_model_inputs() -> dict[str, Any]:
    """Document the expected input dictionaries for the TensorFlow models."""

    return {
        "TwoTowerModel": {
            "user": "dict[str, int Tensor] with scalar sparse user features",
            "item": "dict[str, int Tensor] with scalar sparse item features",
        },
        "FineRankModel": {
            "dense_features": "[batch, base_feature_dim]",
            "scene_features": "[batch, scene_feature_dim]",
            "personal_features": "[batch, personal_feature_dim]",
            "target_embedding": "[batch, embedding_dim]",
            "behavior_embeddings": "[batch, sequence_length, embedding_dim]",
            "behavior_mask": "optional [batch, sequence_length], True means valid behavior and False means padding",
        },
        "DINRankModel": {
            "dense_features": "[batch, dense_feature_dim]",
            "target_embedding": "[batch, embedding_dim]",
            "behavior_embeddings": "[batch, sequence_length, embedding_dim]",
            "behavior_mask": "optional [batch, sequence_length], True means valid behavior and False means padding",
        },
    }
