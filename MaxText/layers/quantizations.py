#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools
import json
from typing import Optional

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2.flax import aqt_flax
import common_types
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
import re
from jax.tree_util import tree_flatten_with_path, tree_unflatten
from typing import Tuple, Sequence

MAX_INT8 = 127.5
MAX_INT4 = 7.5

Array = common_types.Array
Config = common_types.Config
AxisIdxes = common_types.AxisIdxes
AxisNames = common_types.AxisNames
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    pass


def _tiling_fn(lhs, rhs, dimension_numbers, tile_size):
  del lhs, rhs
  
  (lhs_ca, rhs_ca), _ = dimension_numbers
  ret = tiled_dot_general.Cfg(
      lhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
      rhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
  )
  
  for lhs_idx, rhs_idx in zip(lhs_ca, rhs_ca):
    ret.lhs.contraction_axes.append(
        tiled_dot_general.AxisTiling(axis=lhs_idx, tile_size=tile_size, tile_count=None)
    )
    ret.rhs.contraction_axes.append(
        tiled_dot_general.AxisTiling(
            axis=rhs_idx, tile_size=tile_size, tile_count=None
        )
    )

  return ret


def _rhs_axis_metadata_wrapper(x: jnp.ndarray, tile_map, no_sharding_axis: Sequence[int], mesh_axes: Tuple[str, ...], is_tiled: bool):
  mesh_axes = list(mesh_axes)
  if is_tiled:
    # tile_map is a mapping between original rank and a list of new, tiled rank.
    if len(mesh_axes) < len(tile_map):
      mesh_axes = [None] * (len(tile_map) - len(mesh_axes)) + mesh_axes
    new_mesh_axes = [None] * len(x.shape)
    for orig_rank, new_rank in tile_map.items():
      assert new_rank
      assert len(new_rank) <= 2
      new_mesh_axes[new_rank[-1]] = mesh_axes[orig_rank]
    mesh_axes = new_mesh_axes
          
  if mesh_axes is not None and len(mesh_axes) > 0:
    for no_shard_idx in no_sharding_axis:
      mesh_axes[no_shard_idx] = None

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()
      

@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral | dict
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    quant_dg, tile_size, tiling_fn = None, -1, None

    if isinstance(self.quant_dg, dict):
      # Mixed precision.
      module_path = '/'.join(nn.module._context.module_stack[-1].path)
      for layer_name_re, layer_quant_dg in self.quant_dg.items():
        if re.fullmatch(layer_name_re, module_path):
          quant_dg, tile_size = layer_quant_dg
      if quant_dg is None:
        quant_dg, tile_size = self.quant_dg['default']
      if tile_size != -1:
        tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    else:
      quant_dg = self.quant_dg
    
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      # Sharding during convert leads to strange behavior when saving
      # the quantized checkpoint.
      rhs_axis_metadata_wrapper = None
    else:
      rhs_axis_metadata_wrapper = functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=tile_size != -1
      )

    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      # Sharding during convert leads to strange behavior when saving
      # the quantized checkpoint.
      rhs_axis_metadata_wrapper = None
    else:
      rhs_axis_metadata_wrapper = functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=False
      )

    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=self.quant_dg,
            lhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
        )
    )
    return aqt_einsum


@dataclass
class Fp8Quantization(Quantization):
  """Configures Fp8 quantization for NVIDIA GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.Fp8DotGeneralOp


def _get_weight_only_quant_config(lhs_bits=None, rhs_bits=None):
  return aqt_config.dot_general_make(lhs_bits=lhs_bits, rhs_bits=rhs_bits)

def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  # This should be able to return a "dictionary", which maps regular expression key to value.
  if not config.quantization or config.quantization == "":
    return None
  elif config.quantization == "int8":
    if config.mixed_precision_config != "":
      with open(config.mixed_precision_config, "r") as fin:
        mixed_precision_config = json.load(fin)
      
      ret_config = {}
      for layer_name_re, layer_quantization_config in mixed_precision_config.items():
        rhs_num_bits = layer_quantization_config.get("bits", 8)
        tile_size = layer_quantization_config.get("tile_size", -1)
        scale = layer_quantization_config.get("scale", 1.0)
        ret_config[layer_name_re] = [aqt_config.dot_general_make(lhs_bits=None, rhs_bits=rhs_num_bits), tile_size]
        if scale < 1.0:
          aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(calibration.AbsMaxCalibration, scale=scale)

      ret_config["default"] = [aqt_config.dot_general_make(lhs_bits=None, rhs_bits=8), -1]
      return ret_config

    # Weight-only for the whole network.
    return aqt_config.dot_general_make(lhs_bits=None, rhs_bits=8)

    # DRQ.
    """
    if config.quantization_local_shard_count == 0:
      drhs_bits = None
      drhs_accumulator_dtype = None
      drhs_local_aqt = None
    else:
      drhs_bits = 8
      drhs_accumulator_dtype = jnp.int32
      drhs_local_aqt = aqt_config.LocalAqt(
          contraction_axis_shard_count=config.quantization_local_shard_count
      )
    

    
    return aqt_config.config_v3(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=drhs_bits,
        rng_type="jax.uniform",
        dlhs_local_aqt=None,
        drhs_local_aqt=drhs_local_aqt,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=drhs_accumulator_dtype,
    )
    """
  elif config.quantization == "int4":
    # Weight-only for the whole network.
    return aqt_config.dot_general_make(lhs_bits=None, rhs_bits=4)
  elif config.quantization == "fp8":
    return "fp8"
  else:
    raise ValueError(f"Invalid value configured for quantization {config.quantization}.")


def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)


def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


def get_quant_mode(quant_mode_str: str = "train"):
  """Set quant mode."""
  if quant_mode_str == "train":
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == "serve":
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == "convert":
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f"Invalid quantization mode {quant_mode_str}.")
  return None


def configure_quantization(config: Config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
      return Fp8Quantization()
    quant_mode = get_quant_mode(quant_mode_str)
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode)
  return None


def _get_aqt_key_paths(aqt_vars):
  """Generate a list of paths which have aqt state"""
  aqt_tree_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_vars)
  aqt_key_paths = []
  for k, _ in aqt_tree_flat:
    pruned_keys = []
    for d in list(k):
      if "AqtDotGeneral" in d.key:
        pruned_keys.append(jax.tree_util.DictKey(key="kernel"))
        break
      else:
        assert "Aqt" not in d.key, f"Unexpected Aqt op {d.key} in {k}."
        pruned_keys.append(d)
    aqt_key_paths.append(tuple(pruned_keys))
  return aqt_key_paths


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  aqt_paths = _get_aqt_key_paths(aqt_vars)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in aqt_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)


def configure_kv_quantization(config: Config):
  """Configure kv quantization based on user config."""
  return config.quantize_kvcache


def get_kvcache_dtype(quantize_kvcache: str):
  if quantize_kvcache == "int8":
    return jnp.int8
  elif quantize_kvcache == "int4":
    return jnp.int4
  elif quantize_kvcache == "":
    return jnp.bfloat16

  print("====== QUANTIZE_KVCACHE_VALUE: [" + quantize_kvcache + "]")
  raise f"Unknown KVCache Quantization Option: {quantize_kvcache}. Should be either 'int8', 'int4' or ''."


def quantize_kv(kv: Array, kv_quant_axis: str, axis_names: AxisNames, quantize_kvcache: str):
  """Quantize key/values stored in kvcache."""
  if kv_quant_axis == "dkv":
    max_axis_over = axis_names.index(CACHE_KV)
  elif kv_quant_axis == "heads_and_dkv":
    max_axis_over = (
      axis_names.index(CACHE_HEADS),
      axis_names.index(CACHE_KV)
    )
  scale = jnp.max(jnp.abs(kv), axis=max_axis_over, keepdims=True)
  if quantize_kvcache == "int8":
    value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
  elif quantize_kvcache == "int4":
    print("========= QUANTIZING KVCACHE to INT4")
    value = jnp.int4(jnp.rint(kv * (MAX_INT4 / scale)))
  else:
    raise f"Unknown KVCache Quantization Option: {quantize_kvcache}. Should be either 'int8', 'int4' or ''."
  return value, scale

def unquantize_kv(value: Array, scale: Array, dtype: jnp.dtype):
  """Unquantize key/values stored in kvcache."""
  # I do not know what 'dtype' here means, but its value is quite unpredictable...
  if value.dtype == jnp.int8:
    return value.astype(dtype) * scale / MAX_INT8
  elif value.dtype == jnp.int4:
    # int4 must be explicitly casted to float before multiplication.
    return value.astype(scale.dtype) * scale / MAX_INT4
  
  raise f"Bad quantized dtype: {value.dtype}"

# Replacing the above code with QTensor *significantly* slows down the running speed - Sharding issue possibly.
"""
def quantize_kv(kv: Array, quantize_kvcache: str):
  kvcache_dtype = get_kvcache_dtype(quantize_kvcache)
  num_bits = 8 if kvcache_dtype == jnp.int8 else 4

  # Only preserve max val for 8-bit quantization; for 4-bit quant, we want to
  # use the Q=7.5 value specifically, hence preserve_max_val=False if k4v4.
  quantizer = aqt_config.quantizer_make(num_bits,
                                        preserve_max_val=num_bits == 8)
  qt, _ = quantizer.quant(kv, calibration_axes=[-1])
  assert len(qt.scale) == 1
  qvalue_dtype = quantizer.numerics.get_dtype()
  qt = qt.qvalue_astype(qvalue_dtype)

  print("======= KVCache quantized to: ", qt.qvalue.dtype, quantize_kvcache)

  return qt.qvalue, qt.scale[0]

def unquantize_kv(value: Array, scale: Array, dtype: jnp.dtype):
  qt = aqt_tensor.QTensor(qvalue=value, scale=[scale], scale_t=None, dequant_dtype=dtype)

  return qt.dequant()
"""
