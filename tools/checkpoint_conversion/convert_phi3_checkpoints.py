# Copyright 2023 The KerasNLP Authors
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

import argparse
import json
import os

import huggingface_hub
import numpy as np
import transformers

import keras_nlp
from keras_nlp.models import Phi3Backbone

PRESET_MAP = {
    "phi3_mini_4k_instruct_en": "microsoft/Phi-3-mini-4k-instruct",
    "phi3_mini_128k_instruct_en": "microsoft/Phi-3-mini-128k-instruct",
}

def download_hf_model(hf_model_name, extract_dir):
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.model"],
        ignore_patterns=["*/*"],
        local_dir=extract_dir,
    )

    return hf_model_dir


def convert_model(hf_model):
    # get huggingface model configuration.
    hf_config = hf_model.config.to_dict()

    kwargs = {}
    kwargs["vocabulary_size"] = hf_config["vocab_size"]
    kwargs["num_layers"] = hf_config["n_layer"]
    kwargs["num_heads"] = hf_config["n_head"]
    kwargs["hidden_dim"] = hf_config["hidden_size"]
    kwargs["intermediate_dim"] = hf_config["hidden_size"] * 4
    kwargs["dropout"] = hf_config["hidden_dropout"]
    kwargs["layer_norm_epsilon"] = hf_config["layer_norm_epsilon"]

    return Phi3Backbone(**kwargs)


def convert_tokenizer(hf_model_dir):
    pass


def convert_weights(keras_model, hf_model):
    hidden_dim = keras_model.hidden_dim
    num_heads = keras_model.num_heads
    head_dim = hidden_dim // num_heads
    num_layers = keras_model.num_layers

    # get huggingface model weights.
    hf_wts = hf_model.state_dict()

    # assign huggingface weights to the keras model.
    # Embedding layer.
    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["word_embeddings.weight"].detach().numpy()
    )
    # LayerNorm.
    keras_model.get_layer("token_embedding_layernorm").gamma.assign(
        hf_wts["word_embeddings_layernorm.weight"].detach().numpy()
    )
    keras_model.get_layer("token_embedding_layernorm").beta.assign(
        hf_wts["word_embeddings_layernorm.bias"].detach().numpy()
    )

    keras_model.get_layer("final_layernorm").gamma.assign(
        hf_wts["ln_f.weight"].detach().numpy()
    )
    keras_model.get_layer("final_layernorm").beta.assign(
        hf_wts["ln_f.bias"].detach().numpy()
    )

    # Decoder layers.
    for i in range(num_layers):
        decoder_layer = keras_model.get_layer(f"transformer_layer_{i}")
        # LayrNorm.
        decoder_layer._pre_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.input_layernorm.weight"].detach().numpy()
        )
        decoder_layer._pre_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.input_layernorm.bias"].detach().numpy()
        )
        decoder_layer._post_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.weight"].detach().numpy()
        )
        decoder_layer._post_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.bias"].detach().numpy()
        )

        # Attention layer.
        attention_layer = decoder_layer._self_attention_layer

        fused_qkv_kernal = (
            hf_wts[f"h.{i}.self_attention.query_key_value.weight"]
            .T.detach()
            .numpy()
        )
        fused_qkv_kernal = fused_qkv_kernal.reshape(
            hidden_dim, num_heads, 3, head_dim
        )
        query_kernal = fused_qkv_kernal[..., 0, :]
        key_kernal = fused_qkv_kernal[..., 1, :]
        value_kernl = fused_qkv_kernal[..., 2, :]

        fused_qkv_bais = (
            hf_wts[f"h.{i}.self_attention.query_key_value.bias"]
            .detach()
            .numpy()
        )
        fused_qkv_bais = fused_qkv_bais.reshape(num_heads, 3, head_dim)
        query_bais = fused_qkv_bais[:, 0, :]
        key_bais = fused_qkv_bais[:, 1, :]
        value_bais = fused_qkv_bais[:, 2, :]

        attention_layer._query_dense.kernel.assign(query_kernal)
        attention_layer._query_dense.bias.assign(query_bais)
        attention_layer._key_dense.kernel.assign(key_kernal)
        attention_layer._key_dense.bias.assign(key_bais)
        attention_layer._value_dense.kernel.assign(value_kernl)
        attention_layer._value_dense.bias.assign(value_bais)

        attention_layer._output_dense.kernel.assign(
            hf_wts[f"h.{i}.self_attention.dense.weight"].T.detach().numpy()
        )
        attention_layer._output_dense.bias.assign(
            hf_wts[f"h.{i}.self_attention.dense.bias"].detach().numpy()
        )

        # mlp.
        decoder_layer._mlp_intermediate_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.weight"].T.detach().numpy()
        )
        decoder_layer._mlp_intermediate_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.bias"].detach().numpy()
        )
        decoder_layer._mlp_output_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.weight"].T.detach().numpy()
        )
        decoder_layer._mlp_output_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.bias"].detach().numpy()
        )


def validate_output(
    hf_model,
    keras_model,
):
    input_ids = np.ones((4, 10))
    padding_mask = np.ones((4, 10))

    # Huggingface
    hf_model_input = {
        "input_ids": input_ids,
        "attention_mask": padding_mask,
    }
    hf_model_outputs = hf_model.forward(**hf_model_input)

    # KerasNLP
    keras_model_input = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
    }
    keras_model_outputs = keras_model.predict(keras_model_input)

    # Comparing the outputs.
    print("🔶 KerasNLP output:", keras_model_outputs[0, 0, :10])
    print("🔶 HF output:", hf_model_outputs[0, 0, :10])
    print("🔶 Difference:", np.mean(keras_model_outputs - hf_model_outputs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=PRESET_MAP.keys(),
        help=f"Preset must be one of {", ".join(PRESET_MAP.keys()) }"
    )
    args = parser.parse_args()
    preset = args.preset
    print(f"✅ Coverting {preset}")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name, f"./{preset}_hf_model")
    print("✅ Huggingface model downloaded from hub")

    # Load the causal model to convert lm_head weights.
    hf_causal_model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    hf_model = hf_causal_model.model
    print(hf_causal_model.dtype)
    print("✅ Huggingface model loaded")

    keras_model = convert_model(hf_causal_model)
    print("✅ Keras model loaded")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted")

    validate_output(
        hf_model,
        keras_model,
    )
    print("✅ Numerics validated")

    # Save float32 keras preset
    keras_nlp.src.utils.preset_utils.save_to_preset(keras_model, preset)

    # Delete float32 Keras model and hf model
    del keras_model
    del hf_model

    # Load The model in float16 percision
    preset_path = os.path.join(os.getcwd(), preset)
    keras_model = Phi3Backbone.from_preset(preset_path, dtype="bfloat16")

    # Save float16 keras model
    keras_nlp.src.utils.preset_utils.save_to_preset(keras_model, preset)
    print("✅ Preset saved")

if __name__ == "__main__":
    main()
