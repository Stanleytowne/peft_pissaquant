# Copyright 2023-present the HuggingFace Inc. team.
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

# Reference code: https://github.com/yxli2123/LoftQ/blob/main/utils.py
# Reference paper: https://arxiv.org/abs/2310.08659

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from safetensors import SafetensorError, safe_open
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

_normal_map = {}

def _get_normal_map(num_bits=2):
    global _normal_map
    if num_bits not in _normal_map:
        _normal_map[num_bits] = create_normal_map(num_bits)
    return _normal_map[num_bits]

def create_normal_map(num_bits=2, offset=0.9677083):
    try:
        from scipy.stats import norm
    except ImportError:
        raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")

    variations = 2**num_bits
    # one more positive value, this is an asymmetric type
    v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
    v2 = [0]
    v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    return values

def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    U = U[:, : reduced_rank]
    Vh = Vh[: reduced_rank, :]
    S = S[: reduced_rank]

    B = U @ torch.sqrt(torch.diag(S))
    A = torch.sqrt(torch.diag(S)) @ Vh

    return B, A

def pissaquant_init_from_absmax(weight, rank):
    block_size = weight.shape[1] // rank
    while weight.shape[1] % block_size:
        block_size -= 1
    assert block_size > 0, f"block_size must be greater than 0, got {block_size}"

    weight_blocks = weight.view(weight.shape[0], -1, block_size)
    absmax = weight_blocks.abs().max(dim=-1)[0]
    absmax_expanded = absmax.repeat_interleave(block_size, dim=-1)
    absmax_expanded = absmax_expanded.reshape((weight.shape[0], weight.shape[1]))

    return _low_rank_decomposition(absmax_expanded, rank)

def get_qweight_with_AB(weight, B, A, norm_lookup_table):
    scale = B @ A
    scale = torch.clamp(scale, min=1e-8)
    assert weight.shape == scale.shape, f"Weight and scale shapes do not match: {weight.shape} != {scale.shape}"
    
    weight_divabs = weight / scale  # (L, B)
    weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
    L_reshaped = norm_lookup_table.reshape(1, -1)  # (1, 2**K)

    abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
    qweight_idx = torch.argmin(abs_diff, dim=-1)  # (L, B)
    qweight = norm_lookup_table[qweight_idx]  # (L, B)

    return qweight

def refine_ab(
    w_fp32: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
    bits: int,
    eps: float = 1e-8, 
    steps: int = 3000,
    lr: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Iteratively refine A/B to reduce quantization error.
    """
    device = w_fp32.device
    B = B.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    A = A.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([B, A], lr=lr)

    norm_lookup_table = _get_normal_map(bits).to(device)

    for step in range(steps):
        with torch.no_grad():
            qweight = get_qweight_with_AB(w_fp32, B, A, norm_lookup_table)

        optimizer.zero_grad(set_to_none=True)
        scale = torch.clamp(B @ A, min=eps)
        w_hat = qweight * scale
        loss = torch.linalg.norm(w_hat - w_fp32)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logging.info(f"Refine AB: Step {step} loss: {loss.item()}")

    return B.detach(), A.detach(), qweight



def pissaquant_init(weight: Union[torch.Tensor, torch.nn.Parameter], num_bits: int, reduced_rank: int, apply_quantization: bool):

    if num_bits not in [2, 4, 8]:
        raise ValueError("Only support 2, 4, 8 bits quantization")

    out_feature, in_feature = weight.size()
    device = weight.device
    dtype = weight.dtype

    logging.info(
        f"Weight: ({out_feature}, {in_feature}) | Rank: {reduced_rank} | Num Bits: {num_bits}"
    )

    w_fp32 = weight.to(dtype=torch.float32)
    
    B, A = pissaquant_init_from_absmax(w_fp32, reduced_rank)

    B, A, qweight = refine_ab(w_fp32, B, A, num_bits)

    if not apply_quantization:
        new_weight = weight / (B @ A)
    else:
        new_weight = qweight

    return new_weight.to(device=device, dtype=dtype), B, A
