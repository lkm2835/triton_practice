import torch

from time import time
from tqdm import tqdm

import transformer_engine
import transformer_engine.pytorch.attention as tepa

from triton_rope import TritonRoPEFunc


def test_assert_close(seq_len, batch_size, head, dim, N=100):
    base = 10000
    device = "cuda"
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)[:,None,None,:]

    for i in range(N):
        torch_input = torch.randn([seq_len, batch_size, head, dim], device=device, requires_grad=True)
        triton_input = torch_input.detach().clone()
        triton_input.requires_grad = True

        torch_forward_output = tepa.apply_rotary_pos_emb(torch_input, freqs, fused=True)
        torch_forward_output.sum().backward()
        
        triton_forward_output = TritonRoPEFunc.apply(triton_input, freqs)
        triton_forward_output.sum().backward()
        
        torch.testing.assert_close(torch_forward_output, triton_forward_output)
        torch.testing.assert_close(torch_input.grad, triton_input.grad)
    
    print(f'(len:{seq_len}, bs:{batch_size}, head:{head}, dim:{dim}) random input {N} times success.')


def test_wall_time(seq_len, batch_size, head, dim, N=100, warmup=5):
    base = 10000
    device = "cuda"
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)[:,None,None,:]
    
    torch_input = torch.randn([seq_len, batch_size, head, dim], device=device, requires_grad=True)
    triton_input = torch_input.detach().clone()
    triton_input.requires_grad = True

    print(f'\n(len:{seq_len}, bs:{batch_size}, head:{head}, dim:{dim}) wall time test...')
    end = 0
    for i in range(N):
        start = time()
        torch_forward_output = tepa.apply_rotary_pos_emb(torch_input, freqs, fused=True)
        if i > warmup:
            end += time() - start
    print("torch  forward: ", end / (N - warmup) * 1e6, "microseconds")

    end = 0
    for i in range(N):
        start = time()
        triton_forward_output = TritonRoPEFunc.apply(triton_input, freqs)
        if i > warmup:
            end += time() - start
    print("triton forward: ", end / (N - warmup) * 1e6, "microseconds")

    end = 0
    for i in range(N):
        torch_forward_output = tepa.apply_rotary_pos_emb(torch_input, freqs, fused=True)
        z = torch_forward_output.sum()
        start = time()
        z.backward()
        if i > warmup:
            end += time() - start
    print("torch  backward: ", end / (N - warmup) * 1e6, "microseconds")

    end = 0
    for i in range(N):
        triton_forward_output = TritonRoPEFunc.apply(triton_input, freqs)
        z = triton_forward_output.sum()
        start = time()
        z.backward()
        if i > warmup:
            end += time() - start
    print("triton backward: ", end / (N - warmup) * 1e6, "microseconds")


if __name__ == "__main__":
    test_assert_close(64,  1, 16,  64, 100)
    test_assert_close(128, 1, 16, 128, 100)
    test_assert_close(32,  1,  8, 256, 100)

    test_wall_time(64,  1, 16,  64)
    test_wall_time(128, 1, 16, 128)
    test_wall_time(32,  1,  8, 256)
