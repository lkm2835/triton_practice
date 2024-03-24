# Test
- docker image를 생성한 뒤 `python3 test_rope.py`로 테스트를 진행할 수 있습니다.
- `test_assert_close`로 정합성을 체크하고 `test_wall_time`으로 실행 시간을 비교합니다.

### 실행 결과
```
root@45dd1062a2eb:/workspace/triton_practice/rope# python3 test_rope.py
(len:64, bs:1, head:16, dim:64) random input 100 times success.
(len:128, bs:1, head:16, dim:128) random input 100 times success.
(len:32, bs:1, head:8, dim:256) random input 100 times success.

(len:64, bs:1, head:16, dim:64) wall time test...
torch  forward:  23.749000147769326 microseconds
triton forward:  172.8308828253495 microseconds
torch  backward:  188.7798309326172 microseconds
triton backward:  454.3806377210115 microseconds

(len:128, bs:1, head:16, dim:128) wall time test...
torch  forward:  24.61483604029605 microseconds
triton forward:  175.72704114412008 microseconds
torch  backward:  125.41821128443667 microseconds
triton backward:  526.0367142526726 microseconds

(len:32, bs:1, head:8, dim:256) wall time test...
torch  forward:  50.231030112818665 microseconds
triton forward:  169.89206012926604 microseconds
torch  backward:  271.9854053698088 microseconds
triton backward:  463.0113902844881 microseconds
```
- 테스트에 들어가는 freqs 값은 transformers llama 구현을 참고하였습니다.
- [LlamaRotaryEmbedding](https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/models/llama/modeling_llama.py#L96)

### 구현
- 미구현: batch_size 처리, 다양한 block size 지원
- 처음에 output matrix를 만들지 않고 t matrix에 덮어씌우면 [slicing](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1225)과 [cat](https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1230)하는 과정도 생략할 수 있겠다 싶어 구현하다가 문제가 발생하여 수정하였습니다.
- 이 버그를 찾는데 생각보다 많은 시간이 소요되어 구현이 부족하게 되었으나, 해당 버그를 짚고 넘어가면 좋을 것 같아 설명드립니다.
### 버그 설명
- 기존에는 작성한 triton_kernel에서 o_ptr을 받지 않고 t_ptr에 덮어씌웠습니다.
- 해당 [라인](https://github.com/lkm2835/triton_practice/new/main/rope)을 `tl.store(t_ptrs, t * cos_ + t_rot * sin_)` 로 작성했었습니다.
- 그러다보니 같은 입력에 대해 운 좋은 경우에는 문제없이 실행되었지만 운이 나쁜 경우에는 Race Condition이 발생하여 결과가 깨지는 상황이 발생했습니다.
- 현재는 t_ptrs에 덮어씌우지 않고 o_ptrs로 저장하여 버그를 해결하였습니다.
