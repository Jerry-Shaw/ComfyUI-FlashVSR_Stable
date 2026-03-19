"""
https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import os

# 从环境变量读取注意力模式，默认使用 sdpa 以避免 ROCm/Triton 问题
ATTENTION_MODE = os.environ.get("FLASHVSR_ATTENTION_MODE", "sdpa").lower()
print(f"[SparseSage] Attention mode: {ATTENTION_MODE}")

def sparse_sageattn(q, k, v, mask_id=None, is_causal=False, tensor_layout="HND"):
    """
    Unified attention function that respects the attention_mode setting.
    When mode is 'sdpa', uses standard PyTorch attention.
    When mode is 'sparse', uses the original sparse implementation.
    """
    
    # SDPA mode - use standard PyTorch attention
    if ATTENTION_MODE == "sdpa":
        return sdpa_attention(q, k, v, is_causal, tensor_layout, attention_mask=mask_id)
    
    # Original sparse mode - use the Triton-based implementation
    else:
        return original_sparse_attention(q, k, v, mask_id, is_causal, tensor_layout)


def sdpa_attention(q, k, v, is_causal=False, tensor_layout="HND", attention_mask=None):
    """
    Standard PyTorch Scaled Dot-Product Attention implementation
    Works with both 3D and 4D tensors
    """
    output_dtype = q.dtype
    
    # Convert to float16 for better performance
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    
    original_shape = q.shape
    
    # 忽略传入的 attention_mask，因为 SDPA 不需要它
    
    # Handle 3D tensors [batch, seq_len, dim]
    if len(original_shape) == 3:
        batch_size, seq_len, total_dim = original_shape
        
        # Try to determine number of heads
        possible_heads = [8, 12, 16, 24, 32, 40, 48]
        num_heads = 24  # Default for Wan models
        
        # Find a suitable number of heads that divides total_dim
        for heads in possible_heads:
            if total_dim % heads == 0:
                num_heads = heads
                break
        
        # If none of the common values work, find any divisor
        if total_dim % num_heads != 0:
            for heads in range(min(64, total_dim), 0, -1):
                if total_dim % heads == 0:
                    num_heads = heads
                    break
        
        head_dim = total_dim // num_heads
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply attention without mask
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal
        )
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    # Handle 4D tensors
    elif len(original_shape) == 4:
        if tensor_layout == "NHD":
            # Already in [batch, num_heads, seq_len, head_dim] format
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal
            )
        else:
            # Assume [batch, seq_len, num_heads, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal
            )
            output = output.transpose(1, 2)
    else:
        raise ValueError(f"Unsupported input dimension: {len(original_shape)}")
    
    return output.to(output_dtype)


def original_sparse_attention(q, k, v, mask_id=None, is_causal=False, tensor_layout="HND"):
    """
    Original sparse attention implementation using Triton kernels
    Only used when ATTENTION_MODE is not 'sdpa'
    """
    try:
        from .quant_per_block import per_block_int8
        from .sparse_int8_attn import forward as sparse_sageattn_fwd
        
        if mask_id is None:
            mask_id = torch.ones(
                (q.shape[0], q.shape[1], 
                 (q.shape[2] + 128 - 1)//128, 
                 (q.shape[3] + 64 - 1)//64), 
                dtype=torch.int8, device=q.device
            )

        output_dtype = q.dtype
        if output_dtype == torch.bfloat16 or output_dtype == torch.float32:
            v = v.to(torch.float16)
        
        seq_dim = 1 if tensor_layout == "NHD" else 2
        km = k.mean(dim=seq_dim, keepdim=True)

        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)
        
        o = sparse_sageattn_fwd(
            q_int8, k_int8, mask_id, v, q_scale, k_scale, 
            is_causal=is_causal, tensor_layout=tensor_layout, output_dtype=output_dtype
        )
        return o
    except Exception as e:
        print(f"[SparseSage] Original sparse attention failed, falling back to SDPA: {e}")
        return sdpa_attention(q, k, v, is_causal, tensor_layout)


# 添加一个函数来动态设置 attention mode
def set_attention_mode(mode: str):
    """Set the attention mode globally"""
    global ATTENTION_MODE
    mode = mode.lower()
    if mode in ["sdpa", "sparse"]:
        ATTENTION_MODE = mode
        print(f"[SparseSage] Attention mode set to: {ATTENTION_MODE}")
    else:
        print(f"[SparseSage] Unknown attention mode: {mode}, using sdpa")
        ATTENTION_MODE = "sdpa"