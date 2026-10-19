# Triton tutorials

CPU tutorials (have a version in `python/tutorials/cpu/`):

| tutorial | kernels |
|---|---|
| 01 vector-add | `add_kernel` |
| 02 fused-softmax | `softmax_kernel` |
| 03 matrix-multiplication | `matmul_kernel` |
| 04 blocked-matmul | `matmul_kernel` (row-major and blocked), `block_transpose_combined_kernel` |
| 05 layer-norm | `_layer_norm_fwd_fused`, `_layer_norm_bwd_dx_fused`, `_layer_norm_bwd_dwdb` |
| 06 matrix-vector-multiplication | `gemv_kernel` |
| 07 matrix-vector-multiplication-bf16 | `gemv_kernel` |
| 08 sfc-matmul | `block_transpose_pack_kernel`, `sfc_kernel` |

GPU-only tutorials (only in `python/tutorials/`):

| tutorial | kernels |
|---|---|
| 04 low-memory-dropout | `_dropout`, `_seeded_dropout` |
| 06 fused-attention | `_attn_fwd` |
| 07 extern-functions | `asin_kernel` |
| 08 grouped-gemm | `grouped_matmul_kernel` |
| 09 persistent-matmul | `matmul_kernel`, `matmul_kernel_persistent`, `matmul_kernel_descriptor_persistent` |
| 10 block-scaled-matmul | `block_scaled_matmul_kernel` |

# Results on Liger-Kernel (board)

29 Triton kernels of 16 Liger-Kernel ops (`~/Liger-Kernel`, `src/liger_kernel/ops/`), each wrapped in an `rvv_liger_<op>_elf.py` harness (triton-cpu `python/tutorials/cpu/`) that imports the kernel from Liger, checks it against a torch reference and runs it on the board. They are compiled through Triton (`TRITON_RISCV_LMUL=8`, flags via `TRITON_RISCV_LLVM_ARGS`, `-verify-machineinstrs`), without `--custom-*` flags (baseline) and with all of them (`--custom-copy --custom-sink --custom-affine --custom-remat --custom-reverse`). Max VP is `ExpandPseudos`' own number before → after its transforms; spills and reloads are `Folded Spill` / `Folded Reload` in the kernel's assembly; fires are the pass's debug lines (`lowerCopy`, sink success, `Affine:`, `Remat`, `ReverseRemat[`). fp32, small shapes (8 or 16 rows of 500 columns, a vocabulary of 1000), with irregular sizes so the masked tails are tested.

| kernel | config | max VP | spills / reloads | fires | board |
|---|---|---|---|---|---|
| `rms-norm-fwd` | baseline | 137 | 31 / 42 | — | PASS 0.0104 ms |
|  | all | 137 → 137 | 26 / 32 | 15 copies, 7 affine | PASS 0.0098 ms |
| `rms-norm-bwd` | baseline | 328 | 84 / 118 | — | PASS 0.0283 ms |
|  | all | 328 → 272 | 84 / 123 | 7 affine, 8 reverse remats | PASS 0.0319 ms |
| `layer-norm-fwd` | baseline | 266 | 47 / 57 | — | PASS 0.0142 ms |
|  | all | 266 → 266 | 47 / 55 | 24 copies, 1 sink, 7 affine | PASS 0.0135 ms |
| `layer-norm-bwd` | baseline | 336 | 122 / 167 | — | PASS 0.0346 ms |
|  | all | 336 → 320 | 122 / 173 | 7 affine, 9 reverse remats | PASS 0.0397 ms |
| `fused-add-rms-norm-fwd` | baseline | 145 | 45 / 60 | — | PASS 0.0552 ms |
|  | all | 145 → 145 | 40 / 60 | 23 copies, 7 affine | PASS 0.081 ms |
| `fused-add-rms-norm-bwd` | baseline | 336 | 96 / 134 | — | PASS 0.037 ms |
|  | all | 336 → 328 | 96 / 149 | 7 affine, 9 reverse remats | PASS 0.1221 ms |
| `poly-norm-fwd` | baseline | 226 | 49 / 76 | — | PASS 0.0136 ms |
|  | all | 226 → 184 | 46 / 74 | 7 copies, 7 affine, 8 reverse remats | PASS 0.0143 ms |
| `poly-norm-bwd` | baseline | 361 | 120 / 207 | — | PASS 0.0482 ms |
|  | all | 361 → 361 | 124 / 201 | 15 copies, 7 affine, 10 reverse remats | PASS 0.0402 ms |
| `group-norm-fwd` | baseline | 361 | 175 / 219 | — | PASS 0.0147 ms |
|  | all | 361 → 354 | 158 / 203 | 7 copies, 1 sink, 15 affine, 15 reverse remats | PASS 0.0149 ms |
| `group-norm-bwd` | baseline | 57 | 59 / 80 | — | PASS 0.0178 ms |
|  | all | 57 → 43 | 53 / 73 | 4 copies, 12 sinks, 4 reverse remats | PASS 0.0488 ms |
| `dyt-fwd` | baseline | 200 | 54 / 182 | — | PASS 0.1907 ms |
|  | all | 200 → 200 | 59 / 187 | 23 copies, 16 affine | PASS 0.0771 ms |
| `dyt-bwd` | baseline | 337 | 174 / 340 | — | PASS 0.1845 ms |
|  | all | 337 → 337 | 174 / 340 | — | PASS 0.1772 ms |
| `swiglu-fwd` | baseline | 195 | 42 / 163 | — | PASS 0.0321 ms |
|  | all | 195 → 193 | 41 / 162 | 3 sinks, 8 reverse remats | PASS 0.0357 ms |
| `swiglu-bwd` | baseline | 266 | 79 / 209 | — | PASS 0.038 ms |
|  | all | 266 → 266 | 78 / 209 | 1 copy, 3 sinks, 8 reverse remats | PASS 0.0357 ms |
| `geglu-fwd` | baseline | 195 | 43 / 174 | — | PASS 0.0792 ms |
|  | all | 195 → 195 | 43 / 172 | 5 sinks | PASS 0.0941 ms |
| `geglu-bwd` | baseline | 394 | 94 / 251 | — | PASS 0.0427 ms |
|  | all | 394 → 394 | 94 / 249 | 1 copy, 5 sinks | PASS 0.0478 ms |
| `relu-squared-fwd` | baseline | 65 | 5 / 5 | — | PASS 0.0025 ms |
|  | all | 65 → 65 | 5 / 5 | — | PASS 0.0289 ms |
| `relu-squared-bwd` | baseline | 113 | 15 / 15 | — | PASS 0.0291 ms |
|  | all | 113 → 113 | 14 / 14 | 4 sinks | PASS 0.0044 ms |
| `softmax-fwd` | baseline | 96 | 50 / 180 | — | PASS 0.0833 ms |
|  | all | 96 → 96 | 45 / 173 | 7 copies, 7 affine | PASS 0.0304 ms |
| `softmax-bwd` | baseline | 152 | 25 / 40 | — | PASS 0.0369 ms |
|  | all | 152 → 152 | 24 / 37 | 15 copies, 7 affine | PASS 0.0341 ms |
| `sparsemax-fwd` | baseline | 321 | 181 / 246 | — | PASS 0.0582 ms |
|  | all | 321 → 321 | 187 / 247 | 14 copies, 7 affine, 4 reverse remats | PASS 0.0601 ms |
| `sparsemax-bwd` | baseline | 154 | 72 / 75 | — | PASS 0.0574 ms |
|  | all | 154 → 137 | 66 / 67 | 39 copies, 9 sinks, 16 affine | PASS 0.0381 ms |
| `rope-fwd` | baseline | 68 | 9 / 11 | — | PASS 0.0281 ms |
|  | all | 68 → 68 | 9 / 10 | 8 sinks | PASS 0.027 ms |
| `rope-bwd` | baseline | 68 | 9 / 11 | — | PASS 0.0262 ms |
|  | all | 68 → 68 | 9 / 10 | 8 sinks | PASS 0.0367 ms |
| `cross-entropy` | baseline | 184 | 168 / 663 | — | PASS 0.2838 ms |
|  | all | 184 → 168 | 155 / 648 | 30 copies, 1 sink, 48 affine, 5 remats | PASS 0.259 ms |
| `kl-div-fwd` | baseline | 202 | 51 / 180 | — | PASS 0.0854 ms |
|  | all | 202 → 168 | 49 / 183 | 15 copies, 8 affine, 8 reverse remats | PASS 0.0952 ms |
| `kl-div-bwd` | baseline | 103 | 16 / 17 | — | PASS 0.0378 ms |
|  | all | 103 → 81 | 8 / 8 | 7 copies, 8 affine | PASS 0.11 ms |
| `jsd` | baseline | 354 | 152 / 856 | — | PASS 0.406 ms |
|  | all | 354 → 338 | 151 / 854 | 15 copies, 8 affine | PASS 0.4642 ms |
| `tvd` | baseline | 216 | 54 / 86 | — | PASS 0.0428 ms |
|  | all | 216 → 216 | 50 / 83 | 15 copies, 8 affine | PASS 0.1428 ms |

- All 29 kernels pass on the board in both configs.
- In total, spills / reloads go from 2121 / 4864 to 2057 / 4801 (−3% / −1%). The fires are 277 copies, 204 affine, 91 reverse remats, 60 sinks and 5 remats.
- `--custom-reverse` fires on 11 kernels, 91 times: every norm backward kernel (rms, layer, fused add + rms, poly, group), poly-norm and group-norm forward, swiglu forward and backward, sparsemax forward and kl-div forward. It lowers max VP in 8 of them (e.g. rms-norm-bwd 328 → 272, poly-norm-fwd 226 → 184, kl-div-fwd 202 → 168), but spills follow only in group-norm (fwd 175 / 219 → 158 / 203, bwd 59 / 80 → 53 / 73) and poly-norm-fwd (49 / 76 → 46 / 74). In rms-, layer- and fused-add-rms-norm backward the spills stay the same and the reloads rise (118 → 123, 167 → 173, 134 → 149): max VP falls at one point, but the allocator spills the same values elsewhere.
- The largest spill reductions come from the other flags: kl-div-bwd 16 / 17 → 8 / 8 and sparsemax-bwd 72 / 75 → 66 / 67 (copies, sinks, affine, no reverse remat), cross-entropy 168 / 663 → 155 / 648 (copies, affine, 5 remats).
- Nothing fires in dyt-bwd and relu-squared-fwd. The board times are single runs of 100 launches and are noisy: relu-squared-fwd, with no fires, measured 0.0025 ms and 0.0289 ms. So the times do not compare the configurations.
- Kernels that write over their inputs (swiglu-bwd, geglu-bwd, rope, cross-entropy) or accumulate into their outputs (group-norm-bwd, atomic adds) are checked after a single launch and timed in a second run without a check.
- Left out: Liger's multi-block softmax kernels, which start from `tl.float32(-float("inf"))`, which this Triton's frontend rejects (`'dtype' object is not callable`). group-norm-bwd's float atomic add becomes a call to libatomic (`__atomic_load_4`), which the kernel `.so` is not linked against; its harness links the runner with `-latomic` so the board can load it. Not covered: the MoE, attention, mHC, GRPO, mRoPE / Llama4 RoPE, multi-token-attention and fused-linear kernels, and the tiled / fused gate-up variants of SwiGLU and the block RMSNorm.

# Results on the lqcd and paper tutorials (board)

The 9 kernels of `rvv_lqcd_*_elf.py` and `rvv_paper_*_elf.py` (triton-cpu `python/tutorials/cpu/`), compiled through Triton (`TRITON_RISCV_LMUL=8`, flags via `TRITON_RISCV_LLVM_ARGS`, `-verify-machineinstrs`) and run on the board, without `--custom-*` flags (baseline) and with all of them (`--custom-sink --custom-a --custom-remat --custom-reverse`, measured when every flag also ran the COPY-to-VMV lowering, i.e. the same as adding `--custom-copy`). Max VP and SLIL are `ExpandPseudos`' own numbers before → after its transforms; spills and reloads are `Folded Spill` / `Folded Reload` in the kernel's assembly. The baseline is a true baseline (the pass changes nothing without a flag); "copies" is the COPY-to-VMV lowering, now `--custom-copy`, which is part of the all-flags numbers.

| kernel | config | max VP | SLIL | spills / reloads | fires | board |
|---|---|---|---|---|---|---|
| complexcjg_times_vector | baseline | 74 | 2126 | 8 / 22 | — | PASS 0.143 ms |
| | all | 74 → 74 | 2126 → 1935 | 17 / 27 | 7 copies, 8 sinks | PASS 0.157 ms |
| complex_times_vector | baseline | 74 | 2126 | 8 / 20 | — | PASS 0.139 ms |
| | all | 74 → 74 | 2126 → 1935 | 15 / 25 | 7 copies, 8 sinks | PASS 0.145 ms |
| hop_t_m | baseline | 409 | 111600 | 171 / 255 | — | PASS 1.06 ms |
| | all | 409 → 394 | 111600 → 108850 | 159 / 241 | 65 copies, 66 sinks, 1 reverse remat | PASS 1.13 ms |
| hop_t_p | baseline | 313 | 59184 | 136 / 201 | — | PASS 0.674 ms |
| | all | 313 → 313 | 59184 → 57889 | 134 / 198 | 41 copies, 42 sinks | PASS 0.695 ms |
| su3_inverse_multiply | baseline | 178 | 15354 | 62 / 91 | — | PASS 0.390 ms |
| | all | 178 → 170 | 15354 → 14427 | 56 / 84 | 23 copies, 24 sinks | PASS 0.309 ms |
| su3_multiply | baseline | 178 | 15354 | 58 / 94 | — | PASS 0.309 ms |
| | all | 178 → 170 | 15354 → 14427 | 52 / 84 | 23 copies, 24 sinks | PASS 0.314 ms |
| vec_times_vec | baseline | 98 | 3206 | 24 / 29 | — | PASS 0.237 ms |
| | all | 98 → 82 | 3206 → 2799 | 22 / 31 | 11 copies, 12 sinks | PASS 0.171 ms |
| paper seq16_f32 | baseline | 129 | 2122 | 13 / 13 | — | PASS 0.394 ms |
| | all | 129 → 25 | 2122 → 775 | 0 / 0 | 13 reverse remats, 1 sink | PASS 0.243 ms |
| paper seq8_f32 | baseline | 65 | 570 | 5 / 5 | — | PASS 0.246 ms |
| | all | 65 → 25 | 570 → 367 | 0 / 0 | 5 reverse remats, 1 sink | PASS 0.311 ms |

- `--custom-reverse` fires on the paper kernels, which are the paper's own sequence (`A[i+1] = A[i] + (i+1)`, then the product of all `A[i]` in reverse): 5 and 13 elementwise remats rebuild each `A[i] = A[i+1] - (i+1)` on the way back, max VP drops to 25 (below the 32 registers) and all spills go away. The inputs are small integers in FP32, so the rounded inverses are exact and the results bit-identical.
- On the lqcd kernels it fires once in total: one elementwise (rounded float) remat in `hop_t_m`. `--custom-a` and `--custom-remat`'s own transforms fire nowhere.
- On the lqcd kernels, all flags lower SLIL in all 7 and max VP in 4 (by sinking), and reduce spills in hop_t_m, hop_t_p and both su3 kernels (e.g. su3_inverse_multiply 62 / 91 → 56 / 84). In the two complex kernels spills roughly double (8 / 22 → 17 / 27, 8 / 20 → 15 / 25): the COPY-to-VMV lowering causes that on its own (17 / 27 and 15 / 25 with `--custom-reverse` and the lowering, while reverse remat fires nothing there). That is why the lowering is now a separate flag, `--custom-copy`. The cause is the greedy allocator, not added pressure; see "Why `--custom-copy` doubles the spills in the complex kernels" below.
- The paper kernels are FP32 (`rvv_paper_seq8_f32_elf.py`, `rvv_paper_seq16_f32_elf.py`). Their former int32 versions gave `--custom-reverse` nothing: LLVM's middle-end folds the int32 chain into `A[i] = A[0] + c_i` and reassociates the product into a tree, so every `A[i]` has a single use and no value has a distant one. Float adds and multiplies are not reassociated without fast-math, so the FP32 chain reaches the backend as written.
- The board times are single runs of 100 launches. They are noisy on this board: with `--custom-reverse` alone, kernels whose code is identical to the baseline (no fires) measured up to 4× slower, so the times do not compare the configurations.

## With `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1`

The same 9 kernels, with `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1` and all flags (`--custom-copy --custom-sink --custom-a --custom-remat --custom-reverse`), run on the board. The columns are the same as above.

| kernel | max VP | SLIL | spills / reloads | fires | board |
|---|---|---|---|---|---|
| complexcjg_times_vector | 48 → 48 | 216 → 216 | 8 / 16 | — | PASS 0.154 ms |
| complex_times_vector | 48 → 48 | 216 → 216 | 8 / 17 | — | PASS 0.158 ms |
| hop_t_m | 240 → 240 | 5488 → 5488 | 128 / 215 | — | PASS 1.53 ms |
| hop_t_p | 224 → 224 | 5112 → 5112 | 103 / 182 | — | PASS 0.925 ms |
| su3_inverse_multiply | 120 → 120 | 832 → 832 | 29 / 76 | — | PASS 0.296 ms |
| su3_multiply | 120 → 120 | 832 → 832 | 28 / 75 | — | PASS 0.407 ms |
| vec_times_vec | 112 → 112 | 336 → 336 | 12 / 22 | — | PASS 0.242 ms |
| paper seq16_f32 | 128 → 24 | 2064 → 712 | 0 / 0 | 13 reverse remats | PASS 0.259 ms |
| paper seq8_f32 | 64 → 24 | 528 → 328 | 0 / 0 | 5 reverse remats | PASS 0.240 ms |

- No transform fires on the lqcd kernels. With `TRITON_VSETVL_MINE`, loads go through a vsetvli copy loop instead of masked loads whose passthru is a zero vector. So there are no zero-splat COPYs to lower, and nothing for the sink, remat or reverse transforms to match.
- Pressure is much lower than in the all-flags runs above. SLIL drops about 10–20× (hop_t_m 108850 → 5488, complex_times_vector 1935 → 216) and max VP drops too (hop_t_m 394 → 240, su3 170 → 120).
- Spills drop in every lqcd kernel compared with all flags above: hop_t_m 159 / 241 → 128 / 215, hop_t_p 134 / 198 → 103 / 182, su3_inverse_multiply 56 / 84 → 29 / 76, su3_multiply 52 / 84 → 28 / 75, vec_times_vec 22 / 31 → 12 / 22. The complex kernels are back to the baseline's 8 spills, with fewer reloads (16 and 17, against 22 and 20).
- The paper kernels get the same 13 and 5 reverse remats and 0 spills as before. Only their max VP before the transforms changes (129 → 128, 65 → 64).
- The board times are not lower despite the fewer spills (hop_t_m 1.53 ms here against 1.13 ms above). These are single noisy runs (see above), and the vsetvli copy loops also add instructions.

## Why `--custom-copy` doubles the spills in the complex kernels

Measured on `complex_times_vector`. Standalone `llc` with Triton's options (`-O3 -mcpu=generic-rv64 -mattr=+m,+f,+d,+v -fp-contract=fast -riscv-v-vector-bits-min=256 -riscv-v-register-bit-width-lmul=8 -riscv-v-fixed-length-vector-lmul-max=8`) reproduces the Triton numbers. To isolate the allocator, the MIR printed right after `ExpandPseudos` is run through `llc -run-pass=greedy,virtregrewriter -debug-only=regalloc`, which gives the same counts as the full compile:

| MIR after `ExpandPseudos` | evictions | spills / reloads |
|---|---|---|
| no flag | 5 | 8 / 20 |
| `--custom-copy` | 14 | 15 / 25 (one of each is the mask, `vs1r` / `vl1r`) |
| `--custom-copy`, zero splat `%62` moved right before its load | 15 | 15 / 25 |

The kernel loads 8 LMUL-8 inputs with a masked load each. Each load's passthru is a zero vector (`vmv.v.i 0`, `%62`), and the tail mask `%12` sits in `v0` for the whole block. That leaves 3 LMUL-8 groups (`v8`, `v16`, `v24`) for the inputs and products.

- **No flag:** the 7 inputs other than `%62` start as `COPY %62`. These 7 uses make `%62` the heaviest interval (spill weight 0.064). It takes `$v8m8`, and the next inputs (`%20` 0.037, `%40`, `%56`, `%30`, `%46`) cannot evict it. They wait for the second round and get region-split.
- **`--custom-copy`:** each copy becomes its own `vmv.v.i`. `%62` keeps only its definition, its own load and two late uses, so its weight drops to 0.026. The first eviction is `%20` evicting `%62` from `$v8m8` (cascade 1). The allocator visits the inputs in order of rising weight, so each newcomer beats the previous one: `%40` evicts `%20`, `%56` evicts `%40`, `%30` evicts `%56`, and a product evicts `%30` (cascades 2–5). The evicted intervals are split later, and those splits become the extra spills. At cascade 13 the pressure is high enough that an LMUL-8 product takes `v0`–`v7` and evicts the mask `%12`. The mask is then spilled, reloaded and re-split several times, so this mask churn is a consequence of the chain, not its cause.
- **Moving `%62` next to its load** removes its long live range, but the same chain still happens in `$v24m8` (`%40` evicts `%20`, `%56` evicts `%40`, `%30` evicts `%56`, `%62` evicts `%30`). The long interval was not the problem. The baseline was helped by one heavy interval occupying a register group, which pushed the other inputs onto the split path instead of evicting each other.

The baseline's 8 / 20 is the fragile result. Several allocator options that have nothing to do with the copies push the no-flag MIR to the same 15 / 25:

| allocator option | no flag | `--custom-copy` |
|---|---|---|
| default | 8 / 20 | 15 / 25 |
| `-consider-local-interval-cost=false` | 15 / 25 | 15 / 25 |
| `-enable-deferred-spilling` | 15 / 25 | 15 / 25 |
| `-riscv-disable-subreg-liveness` | 15 / 25 | 15 / 25 |
| `-split-spill-mode=size` or `=speed`, `-exhaustive-register-search`, `-grow-region-complexity-budget=0` | 8 / 20 | 15 / 25 |
| `-greedy-reverse-local-assignment` | 8 / 19 | 15 / 25 |

`-consider-local-interval-cost` makes the region splitter avoid splits that would start an eviction chain. The baseline benefits from it because its inputs reach the splitter. With `--custom-copy`, the inputs evict each other before any split happens.

So `--custom-copy` does not add register pressure. It removes an accidental anchor, and the greedy allocator falls back to its ordinary result for this kernel. Keeping one copy unlowered to recreate the anchor would only be tuned to this kernel. The proper fix would be in how the greedy allocator decides evictions among same-size LMUL-8 intervals. Until then `--custom-copy` stays opt-in.

# Results on the rvv tutorials 01–10 (board)

The `rvv_*_elf.py` tutorials 01–10 (all in triton-cpu `python/tutorials/cpu/`) come from two sets of upstream tutorials:

| source | tutorials | kernels |
|---|---|---|
| CPU tutorials (`python/tutorials/cpu/`) | 01 vector-add | `add_kernel` |
| | 02 fused-softmax | `softmax_kernel` |
| | 03 matrix-multiplication | `matmul_kernel` |
| | 04 blocked-matmul | `matmul_kernel` (row-major and blocked), `block_transpose_combined_kernel` |
| | 05 layer-norm | `_layer_norm_fwd_fused`, `_layer_norm_bwd_dx_fused`, `_layer_norm_bwd_dwdb` |
| | 06 matrix-vector-multiplication | `gemv_kernel` |
| | 07 matrix-vector-multiplication-bf16 | `gemv_kernel` (BF16) |
| | 08 sfc-matmul | `block_transpose_pack_kernel`, `sfc_kernel` (plain, split-K first, split-K last) |
| GPU-only tutorials (`python/tutorials/`, no CPU version) | 04 low-memory-dropout | `_dropout`, `_seeded_dropout` |
| | 06 fused-attention | `_attn_fwd` (not in the table: does not compile in reasonable time) |
| | 07 extern-functions | `asin_kernel` |
| | 08 grouped-gemm | `grouped_matmul_kernel` |
| | 09 persistent-matmul | `matmul_kernel`, `matmul_kernel_persistent`, `matmul_kernel_descriptor_persistent` (not in the table: does not compile in reasonable time) |
| | 10 block-scaled-matmul | `block_scaled_matmul_kernel` (mxfp8, mxfp4, mixed) |

GPU tutorial 11 (programmatic-dependent-launch) has no RISC-V version.

They are compiled through Triton (`TRITON_RISCV_LMUL=8`, `-verify-machineinstrs`) and run on the board in three configs:

- **baseline:** no `--custom-*` flag.
- **all:** `--custom-copy --custom-sink --custom-a --custom-remat --custom-reverse`.
- **all + VSETVL:** all flags plus `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1`.

Max VP and SLIL are `ExpandPseudos`' own numbers before → after its transforms. Spills and reloads are `Folded Spill` / `Folded Reload` in each kernel's assembly. 06 fused-attention and 09 persistent-matmul are left out because they do not compile in reasonable time. 05's backward-dx kernel is left out because it uses atomics, which fail to load on the board (`__atomic_compare_exchange_4`: Triton's target features lack `+a`); its board run is commented out in `rvv_05-layer-norm_elf.py`.

| kernel | config | max VP | SLIL | spills / reloads | fires | board |
|---|---|---|---|---|---|---|
| 01 add_kernel | baseline | 69 | 1737 | 9 / 9 | — | PASS 0.319 ms |
|  | all | 69 → 69 | 1737 → 1484 | 9 / 9 | 7 copies, 8 sinks, 3 affine | PASS 0.312 ms |
|  | all + VSETVL | 40 → 40 | 136 → 136 | 2 / 2 | — | PASS 0.269 ms |
| 02 softmax_kernel | baseline | 176 | 65169 | 103 / 366 | — | PASS 0.999 ms |
|  | all | 176 → 176 | 65169 → 65169 | 99 / 362 | 15 copies | PASS 1.1 ms |
|  | all + VSETVL | 160 → 160 | 48490 → 48490 | 83 / 341 | — | PASS 1.01 ms |
| 03 matmul_kernel | baseline | 128 | 6005 | 237 / 237 | — | PASS 1.3 ms |
|  | all | 128 → 128 | 6005 → 6005 | 237 / 237 | — | PASS 1.29 ms |
|  | all + VSETVL | 128 → 128 | 6005 → 6005 | 237 / 237 | — | PASS 1.3 ms |
| 04 blocked: matmul (row-major) | baseline | 200 | 27719 | 154 / 146 | — | PASS 1.44 ms |
|  | all | 200 → 200 | 27719 → 27718 | 154 / 146 | 1 sink | PASS 1.44 ms |
|  | all + VSETVL | 200 → 200 | 27719 → 27718 | 154 / 146 | 1 sink | PASS 1.44 ms |
| 04 blocked: block_transpose (encode) | baseline | 22 | 2107 | 24 / 26 | — | PASS 0.107 ms |
|  | all | 22 → 22 | 2107 → 2107 | 24 / 26 | 4 sinks | PASS 0.101 ms |
|  | all + VSETVL | 22 → 22 | 2107 → 2107 | 24 / 26 | 4 sinks | PASS 0.103 ms |
| 04 blocked: matmul (blocked) | baseline | 216 | 28806 | 137 / 124 | — | PASS 1.33 ms |
|  | all | 216 → 216 | 28806 → 28805 | 137 / 124 | 1 sink | PASS 1.34 ms |
|  | all + VSETVL | 216 → 216 | 28806 → 28805 | 137 / 124 | 1 sink | PASS 1.29 ms |
| 04 dropout: _dropout | baseline | 69 | 2304 | 9 / 9 | — | PASS 0.0289 ms |
|  | all | 69 → 62 | 2304 → 2005 | 8 / 8 | 7 copies, 8 sinks, 3 affine | PASS 0.0315 ms |
|  | all + VSETVL | 73 → 73 | 672 → 672 | 3 / 3 | — | PASS 0.033 ms |
| 04 dropout: _seeded_dropout | baseline | 324 | 64607 | 128 / 151 | — | PASS 0.0453 ms, PASS 0.0443 ms |
|  | all | 324 → 300 | 64607 → 64750 | 124 / 143 | 7 copies, 8 sinks, 4 affine, 19 reverse remats | PASS 0.0453 ms, PASS 0.0464 ms |
|  | all + VSETVL | 320 → 296 | 60646 → 62819 | 112 / 128 | 3 copies, 4 sinks, 11 affine, 23 reverse remats | PASS 0.0457 ms, PASS 0.0482 ms |
| 05 layer_norm_fwd_fused | baseline | 208 | 20847 | 126 / 132 | — | PASS 0.184 ms |
|  | all | 208 → 193 | 20847 → 19821 | 116 / 119 | 46 copies, 24 sinks, 21 affine | PASS 0.169 ms |
|  | all + VSETVL | 144 → 144 | 4900 → 4692 | 101 / 95 | 23 copies | PASS 0.194 ms |
| 05 layer_norm_bwd_dwdb | baseline | 1050 | 206636 | 694 / 566 | — | PASS 0.143 ms |
|  | all | 1050 → 1050 | 206636 → 208509 | 695 / 569 | 1 affine, 2 remats | PASS 0.134 ms |
|  | all + VSETVL | 2032 → 2032 | 20494772 → 20494778 | 8391 / 8884 | 2 sinks | PASS 3.07 ms |
| 06 gemv_kernel | baseline | 104 | 1668 | 18 / 18 | — | PASS 0.25 ms |
|  | all | 104 → 104 | 1668 → 1668 | 18 / 18 | — | PASS 0.234 ms |
|  | all + VSETVL | 104 → 104 | 1385 → 1385 | 10 / 10 | — | PASS 0.17 ms |
| 07 extern: asin_kernel | baseline | 152 | 55548 | 65 / 318 | — | PASS 0.404 ms |
|  | all | 152 → 146 | 55548 → 54773 | 65 / 318 | 15 copies, 16 sinks, 15 affine | PASS 0.271 ms |
|  | all + VSETVL | 130 → 130 | 39040 → 39040 | 48 / 299 | — | PASS 0.423 ms |
| 07 gemv bf16 | baseline | 262 | 64985 | 178 / 240 | — | PASS 2.48 ms |
|  | all | 262 → 148 | 64985 → 52508 | 175 / 261 | 41 reverse remats | PASS 1.05 ms |
|  | all + VSETVL | 132 → 132 | 73499 → 73499 | 1097 / 1116 | — | FAIL 1.47 ms |
| 08 grouped_matmul_kernel | baseline | 4056 | 34126141 | 8596 / 9075 | — | PASS 14.7 ms |
|  | all | 4056 → 4056 | 34126141 → 34127011 | 8504 / 9075 | 125 reverse remats | PASS 12.2 ms |
|  | all + VSETVL | 4056 → 4056 | 34126141 → 34127011 | 8504 / 9075 | 125 reverse remats | PASS 11.2 ms |
| 08 sfc: block_transpose_pack | baseline | 32 | 639 | 258 / 203 | — | PASS 0.0783 ms |
|  | all | 32 → 32 | 639 → 575 | 255 / 203 | 30 copies | PASS 0.0805 ms |
|  | all + VSETVL | 32 → 32 | 639 → 575 | 255 / 203 | 30 copies | PASS 0.0785 ms |
| 08 sfc: sfc_kernel | baseline | 1352 | 720843 | 904 / 1106 | — | PASS 0.293 ms |
|  | all | 1352 → 1352 | 720843 → 720783 | 904 / 1106 | 15 copies | PASS 0.268 ms |
|  | all + VSETVL | 1352 → 1352 | 720843 → 720783 | 904 / 1106 | 15 copies | PASS 0.275 ms |
| 08 sfc: sfc_kernel split-K first | baseline | 1352 | 720597 | 956 / 1130 | — | PASS 0.156 ms |
|  | all | 1352 → 1352 | 720597 → 720597 | 956 / 1130 | — | PASS 0.162 ms |
|  | all + VSETVL | 1352 → 1352 | 720597 → 720597 | 956 / 1130 | — | PASS 0.162 ms |
| 08 sfc: sfc_kernel split-K last | baseline | 1352 | 722437 | 965 / 1143 | — | PASS 0.184 ms |
|  | all | 1352 → 1352 | 722437 → 722437 | 965 / 1143 | — | PASS 0.179 ms |
|  | all + VSETVL | 1352 → 1352 | 722437 → 722437 | 965 / 1143 | — | PASS 0.226 ms |
| 10 block-scaled mxfp8 | baseline | 5708 | 11448128 | 4986 / 5825 | — | PASS 2.06 ms |
|  | all | 5708 → 5708 | 11448128 → 11449546 | 5040 / 5898 | 10 sinks, 17 remats, 2 reverse remats | PASS 2.36 ms |
|  | all + VSETVL | 5708 → 5708 | 11444189 → 11444194 | 4982 / 5824 | 1 sink | PASS 2.52 ms |
| 10 block-scaled mxfp4 | baseline | 5292 | 14872928 | 5884 / 9275 | — | PASS 4.54 ms |
|  | all | 5292 → 5292 | 14872928 → 15149413 | 5809 / 9211 | 14 sinks, 17 remats, 258 reverse remats | PASS 4.45 ms |
|  | all + VSETVL | 5292 → 5292 | 14869133 → 15144265 | 5792 / 9182 | 5 sinks, 257 reverse remats | PASS 4.32 ms |
| 10 block-scaled mixed | baseline | 5544 | 13552756 | 5398 / 7543 | — | PASS 3.49 ms |
|  | all | 5544 → 5544 | 13552756 → 13746986 | 5363 / 7530 | 11 sinks, 17 remats, 130 reverse remats | PASS 3.48 ms |
|  | all + VSETVL | 5544 → 5544 | 13548817 → 13741762 | 5350 / 7499 | 2 sinks, 128 reverse remats | PASS 3.34 ms |

- All runs pass the verifier, and every board run passes except 07 gemv bf16 with all + VSETVL (`Y[0]` got 67, expected 66.5). It fails the same way, with the same 1097 / 1116 spills, under `TRITON_VSETVL_MINE=1 TRITON_VSETVL_REDUCE=1` without any `--custom-*` flag. So the VSETVL lowering breaks the kernel's exact BF16 result; our pass does not.
- With all flags, spills drop in 02, 04 dropout (both kernels), 05 forward, 07 bf16 (178 → 175 spills, but reloads rise 240 → 261), 08 grouped-gemm (8596 → 8504), 08 pack and 10 mxfp4 / mixed. They rise slightly in 05 dwdb and 10 mxfp8. 03, 04 blocked, 06 and the 08 sfc kernels are unchanged.
- Reverse remat fires on 04 seeded dropout (19), 07 bf16 (41; max VP 262 → 148), 08 grouped-gemm (125) and 10 block-scaled (2, 258, 130).
- The VSETVL variables lower pressure and spills further in 01, 02, 04 dropout, 05 forward, 06 and 07 asin, and they leave 03, 04 blocked, 08 and 10 almost unchanged. They blow up 05 dwdb: max VP 1050 → 2032, SLIL about 100× larger, spills 695 → 8391 and the time 0.134 → 3.07 ms. Our pass only sinks twice there, so the blowup comes from the VSETVL lowering of its reductions.
- The board times are single runs of 100 launches and noisy (see above).

# Results on 06 gemv and 03 matmul at LMUL 1, 2, 4, 8 (board)

`rvv_06-matrix-vector-multiplication_elf.py` (`gemv_kernel`) and `rvv_03-matrix-multiplication_elf.py` (`matmul_kernel`), compiled through Triton with `TRITON_RISCV_LMUL` = 1, 2, 4 and 8 (`-verify-machineinstrs`) and run on the board, without `--custom-*` flags (baseline) and with all of them (`--custom-copy --custom-sink --custom-a --custom-affine --custom-remat --custom-reverse`). Max VP and SLIL are `ExpandPseudos`' own numbers before → after its transforms; spills and reloads are `Folded Spill` / `Folded Reload` in the kernel's assembly.

This build places the sinks and remats at the minimum stall gap of `DocStall.md` ("Gap used by `ExpandPseudos`"): directly before the consumer at m4 and m8, and as many independent instructions before it as the table asks at m1 and m2. The load-group sink (`FindFirstUseToSinkToGroup`) is enabled. "group sinks" counts sunk load groups; "held back" counts sinks that did not move because the load was already within the stall gap of its use. `RISCVII::getLMul` reads the producer's LMUL correctly at all four settings (checked with `rvv_01-vector-add_elf.py`: `PseudoVLE32_V_M<n>_MASK` and `PseudoVMSLT_VX_M<n>` read as LMUL n).

| kernel | config | max VP | SLIL | spills / reloads | fires | board |
|---|---|---|---|---|---|---|
| 06 gemv m1 | baseline | 107 | 12294 | 81 / 81 | — | PASS 0.293 ms |
| | all | 107 → 36 | 12294 → 4108 | **22 / 22** | 128 group sinks | PASS **0.138 ms** |
| 06 gemv m2 | baseline | 98 | 5788 | 38 / 38 | — | PASS 0.245 ms |
| | all | 98 → 40 | 5788 → 2236 | **22 / 22** | 55 group sinks (9 held back) | PASS **0.150 ms** |
| 06 gemv m4 | baseline | 100 | 3012 | 24 / 24 | — | PASS 0.213 ms |
| | all | 100 → 40 | 3012 → 1220 | **20 / 20** | 32 group sinks | PASS **0.166 ms** |
| 06 gemv m8 | baseline | 104 | 1668 | 18 / 18 | — | PASS 0.225 ms |
| | all | 104 → 48 | 1668 → 820 | **12 / 12** | 16 group sinks | PASS **0.198 ms** |
| 03 matmul m1 | baseline | 120 | 24084 | 463 / 461 | — | PASS 2.27 ms |
| | all | 120 → 120 | 24084 → 34520 | 463 / 456 | 28 group sinks, 41 reverse remats | PASS 2.22 ms |
| 03 matmul m2 | baseline | 122 | 12098 | 330 / 318 | — | PASS 1.55 ms |
| | all | 122 → 122 | 12098 → 12814 | 321 / 311 | 11 group sinks (3 held back), 14 reverse remats | PASS 1.55 ms |
| 03 matmul m4 | baseline | 124 | 8421 | 251 / 244 | — | PASS 1.27 ms |
| | all | 124 → 124 | 8421 → 8857 | 250 / 243 | 7 group sinks, 7 reverse remats | PASS 1.24 ms |
| 03 matmul m8 | baseline | 128 | 6005 | 237 / 237 | — | PASS 1.26 ms |
| | all | 128 → 128 | 6005 → 5957 | 239 / 238 | 7 group sinks | PASS 1.32 ms |

- In 06 gemv the load-group sink moves each row load next to its `vfmul` / `vfmacc`. That cuts max VP by more than half and spills by 33–73% at every LMUL, and the time drops 53% at m1, 39% at m2, 22% at m4 and 12% at m8. The fastest version is now m1 with all flags (0.138 ms), which was the slowest baseline (0.293 ms).
- At m2 the stall gap holds back 9 of gemv's sinks that would have put a load right before its use. At m1 none are held back: the loads already have enough instructions before their uses.
- In 03 matmul the flags change little. Spills drop slightly at m2 (330 → 321) and m4, not at m1. SLIL rises at m1–m4 (by 43% at m1), likely from the reverse remats (41 at m1), but the spills do not follow. The times are within about 3%, apart from m8 with all flags (1.32 against 1.26 ms, with 2 more spills).
- Across LMULs, matmul is fastest at m4 and m8 (about 1.25 ms) and slowest at m1 (about 2.2 ms); its spills fall as LMUL grows.
- The board times are single runs of 100 launches, after a warm-up run.
