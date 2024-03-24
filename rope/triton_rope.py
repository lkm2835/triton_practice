import torch

import triton
import triton.language as tl

from typing import Union, Tuple

@triton.jit
def triton_apply_rotary_pos_emb_kernel(
    t_ptr,
    freqs_ptr,
    o_ptr,
    s_size,
    h_size,
    d_size,
    s_block_size: tl.constexpr,
    h_block_size: tl.constexpr,
    d_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    d_pid = tl.program_id(2)
    
    num_h_blocks = tl.cdiv(h_size, h_block_size)
    num_d_blocks = tl.cdiv(d_size, d_block_size)
    
    s_block = pid // num_h_blocks
    h_block = pid // num_d_blocks
    d_block = d_pid % num_d_blocks
    
    s_offsets = tl.arange(0, s_block_size)
    h_offsets = tl.arange(0, h_block_size) + h_block * h_block_size
    d_offsets = tl.arange(0, d_block_size) + d_block * d_block_size
    
    if tl.num_programs(2) == 1:
        d_offsets_ = (tl.arange(0, d_block_size) + (d_block_size//2)) % d_block_size + d_block * d_block_size
    else:
        d_block_ = (d_pid + num_d_blocks//2) % num_d_blocks
        d_offsets_ = tl.arange(0, d_block_size) + d_block_ * d_block_size

    freqs_ptrs = freqs_ptr + s_offsets[:, None, None] * d_size + d_offsets[None, None, :]
    t_ptrs = t_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets[None, None, :]
    t_rot_ptrs = t_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets_[None, None, :]
    o_ptrs = o_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets[None, None, :]

    freq = tl.load(freqs_ptrs)
    cos_ = tl.cos(freq)
    sin_ = tl.sin(freq)

    t = tl.load(t_ptrs)
    t_rot = tl.where(d_offsets[None, None, :] < (d_size//2), 
                                -tl.load(t_rot_ptrs), 
                                 tl.load(t_rot_ptrs))                                
            
    tl.store(o_ptrs, t * cos_ + t_rot * sin_)
            
    return


@triton.jit
def triton_apply_rotary_pos_emb_backward_kernel(
    t_ptr,
    freqs_ptr,
    o_ptr,
    s_size,
    h_size,
    d_size,
    s_block_size: tl.constexpr,
    h_block_size: tl.constexpr,
    d_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    d_pid = tl.program_id(2)
    
    num_h_blocks = tl.cdiv(h_size, h_block_size)
    num_d_blocks = tl.cdiv(d_size, d_block_size)
    
    s_block = pid // num_h_blocks
    h_block = pid // num_d_blocks
    d_block = d_pid % num_d_blocks
    
    s_offsets = tl.arange(0, s_block_size)
    h_offsets = tl.arange(0, h_block_size) + h_block * h_block_size
    d_offsets = tl.arange(0, d_block_size) + d_block * d_block_size
    
    if tl.num_programs(2) == 1:
        d_offsets_ = (tl.arange(0, d_block_size) + (d_block_size//2)) % d_block_size + d_block * d_block_size
    else:
        d_block_ = (d_pid + num_d_blocks//2) % num_d_blocks
        d_offsets_ = tl.arange(0, d_block_size) + d_block_ * d_block_size

    freqs_ptrs = freqs_ptr + s_offsets[:, None, None] * d_size + d_offsets[None, None, :]
    freqs_ptrs_ = freqs_ptr + s_offsets[:, None, None] * d_size + d_offsets_[None, None, :]
    t_ptrs = t_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets[None, None, :]
    t_rot_ptrs = t_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets_[None, None, :]
    o_ptrs = o_ptr + s_offsets[:, None, None] * h_size * d_size * 2 + h_offsets[None, :, None] * d_size * 2 + d_offsets[None, None, :]

    freq = tl.load(freqs_ptrs)
    freq_ = tl.load(freqs_ptrs_)
    cos_ = tl.cos(freq)
    sin_ = tl.where(d_offsets[None, None, :] < (d_size//2), 
                                tl.sin(freq_), 
                                -tl.sin(freq_))

    t = tl.load(t_ptrs)
    t_rot = tl.load(t_rot_ptrs)                            
            
    tl.store(o_ptrs, cos_ + sin_)
            
    return


def triton_apply_rotary_pos_emb(    
    t: torch.Tensor,
    freqs: torch.Tensor,
    ) -> torch.Tensor:
    o = t.clone()

    s_size = freqs.shape[0]
    h_size = t.shape[-2]
    d_size = freqs.shape[-1]
    
    s_block_size = s_size
    h_block_size = h_size
    d_block_size = 1
    
    def grid(meta):
        return (triton.cdiv(s_size, meta["s_block_size"]) * triton.cdiv(h_size, meta["h_block_size"]),1, triton.cdiv(d_size, meta["d_block_size"]))
    
    triton_apply_rotary_pos_emb_kernel[grid](t, freqs, o, s_size, h_size, d_size, s_block_size, h_block_size, d_block_size)
    return o


def triton_apply_rotary_pos_emb_backward(    
    t: torch.Tensor,
    freqs: torch.Tensor,
    ) -> torch.Tensor:
    o = t.clone()

    s_size = freqs.shape[0]
    h_size = t.shape[-2]
    d_size = freqs.shape[-1]
    
    s_block_size = s_size
    h_block_size = h_size
    d_block_size = 1
    
    def grid(meta):
        return (triton.cdiv(s_size, meta["s_block_size"]) * triton.cdiv(h_size, meta["h_block_size"]),1, triton.cdiv(d_size, meta["d_block_size"]))
    
    triton_apply_rotary_pos_emb_backward_kernel[grid](t, freqs, o, s_size, h_size, d_size, s_block_size, h_block_size, d_block_size)
    return o


class TritonRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if tensor_format == "sbhd":
            output = triton_apply_rotary_pos_emb(t, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = triton_apply_rotary_pos_emb_backward(grad_output, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None, None, None