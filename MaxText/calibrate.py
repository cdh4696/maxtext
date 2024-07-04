# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI Utility for Running Inference on a Single Stream"""

import jax

import maxengine
from jetstream.engine import token_utils
from jetstream.engine.token_utils import load_vocab

import os
import pyconfig
import sys
import pandas
import orbax
from flax.training import orbax_utils

def load_openorca_dataset_pkl(dataset_fn) -> list[tuple[str, str, int]]:
  # read pickle file
  samples = pandas.read_pickle(dataset_fn)

  prompts = []
  outputs = []
  for _, row in samples.iterrows():
    prompts.append(row["input"])
    outputs.append(row["output"])

  return [(prompt, output, idx) for idx, (prompt, output) in enumerate(zip(prompts, outputs))]

def main(config):
  # 1. Start Engine.
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer

  # 2. Load & Tokenize dataset.
  dataset = load_openorca_dataset_pkl(config.calibrate_dataset)
  tokenized_dataset = []
  for prompt, output, idx in dataset:
    prompt_tokens, true_length = token_utils.tokenize_and_pad(
        prompt, vocab, is_bos=True, prefill_lengths=[config.max_prefill_predict_length]
    )
    if true_length > config.max_prefill_predict_length:  # Filter out the dataset.
      continue
    tokenized_dataset.append([prompt, prompt_tokens, output, true_length, idx])

  print("# of Loaded dataset: ", len(dataset))
  print("# of Tokenized & filtered dataset: ", len(tokenized_dataset))

  # 3. Prepare batches.
  device_count = len(jax.devices())
  batch_size = int(config.per_device_batch_size * device_count)
  #batch_num = len(tokenized_dataset) // batch_size
  batch_num = 1
  print(f"Global batch size: {batch_size}, calibration batch num: {batch_num}")

  input_token_batches, true_length_batches = [], []
  for batch_idx in range(batch_num):
    batch_dataset = tokenized_dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    vec_prompt_tokens = [v[1] for v in batch_dataset]
    vec_true_lengths = [v[3] for v in batch_dataset]
    input_token_batches.append(jax.numpy.array(vec_prompt_tokens))
    true_length_batches.append(jax.numpy.array(vec_true_lengths))

  # 4. Run layerwise calibration.
  quantized_params = engine.execute_layerwise_calibration(params, input_token_batches, true_length_batches)

  # 5. Store the quantized parameters.
  if config.save_quantized_checkpoint:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target({"params":quantized_params})
    orbax_checkpointer.save(
        config.save_quantized_checkpoint, {"params":quantized_params}, save_args=save_args, force=True
    )
    print("QUANTIZED CHECKPOINT SAVED AT: ", config.save_quantized_checkpoint)

  

def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  main(cfg)
