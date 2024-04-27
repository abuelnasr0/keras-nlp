# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

from keras_nlp.backend import ops
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding


class Phi3SuScaledRotaryEmbedding(RotaryEmbedding):
    """SuRotary positional encoding layer.

    Args:
        max_sequence_length: int. The maximum sequence length that this
            model might ever be used with.
        original_max_position_embeddings: int. The maximum sequence length that
            this model was trained with.
        rope_scaling_short_factor List[float]: List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is smaller than `original_max_sequence_length`.
            Defaults to `None`.
        rope_scaling_long_factor List[float]: List of factors used to adjust
            rope frequencies when the `rope_scaling_type` is `"su"`. List must
            be of length `hidden_dim//num_query_heads//2`. It is used when
            `sequence_length` is larger than `original_max_sequence_length`.
            Defaults to `None`.
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves.

    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.

    References:
     - [Phi-3-mini-128k-instruct original implementation](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/0693e0b867d29e7318280ddd3ff9d5e66698f488/modeling_phi3.py#L142)
    """

    def __init__(
        self,
        max_sequence_length=4096,
        original_max_sequence_length=4096,
        inverese_freq_short_factor=None,
        inverese_freq_long_factor=None,
        max_wavelength=10000,
        **kwargs
    ):
        super().__init__(max_wavelength=max_wavelength, **kwargs)
        self.max_sequence_length = max_sequence_length
        self.original_max_sequence_length = original_max_sequence_length

        scaling_factor = (
            self.max_sequence_length / self.original_max_sequence_length
        )
        if scaling_factor <= 1.0:
            self.embedding_scaling_factor = 1.0
        else:
            self.embedding_scaling_factor = math.sqrt(
                1
                + math.log(scaling_factor)
                / math.log(self.original_max_sequence_length)
            )

        if inverese_freq_short_factor is not None:
            self.inverese_freq_short_factor = ops.convert_to_tensor(
                inverese_freq_short_factor,
                dtype="float32",
            )
        else:
            self.inverese_freq_short_factor = None

        if inverese_freq_long_factor is not None:
            self.inverese_freq_long_factor = ops.convert_to_tensor(
                inverese_freq_long_factor,
                dtype="float32",
            )
        else:
            self.inverese_freq_long_factor = None

    def _get_inverse_freq(self, rotary_dim):
        freq_range = ops.divide(
            ops.arange(0, rotary_dim, 2, dtype="float32"),
            ops.cast(rotary_dim, "float32"),
        )
        inverse_freq = 1.0 / (self.max_wavelength**freq_range)
        return inverse_freq

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        feature_axis = len(inputs.shape) - 1
        sequence_axis = 1

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        # Multiply inverse_freq by a factor.
        if ops.shape(inputs)[sequence_axis] > self.original_max_sequence_length:
            inverse_freq = ops.divide(
                inverse_freq, self.inverese_freq_long_factor
            )
        else:
            inverse_freq = ops.divide(
                inverse_freq, self.inverese_freq_short_factor
            )

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
        else:
            positions = ops.cast(positions, "float32")

        freq = ops.einsum("i,j->ij", positions, inverse_freq)
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(
            embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2)
        )

        # Reshape the embedding to be broadcastable with input shape.
        if feature_axis < sequence_axis:
            embedding = ops.transpose(embedding)
        for axis in range(len(inputs.shape)):
            if axis != sequence_axis and axis != feature_axis:
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(
            ops.cos(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        sin_emb = ops.cast(
            ops.sin(embedding) * self.embedding_scaling_factor,
            self.compute_dtype,
        )
        return cos_emb, sin_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_sequence_length": self.max_sequence_length,
                "original_max_sequence_length": self.original_max_sequence_length,
                "inverese_freq_short_factor": self.inverese_freq_short_factor,
                "inverese_freq_long_factor": self.inverese_freq_long_factor,
            }
        )
        return config