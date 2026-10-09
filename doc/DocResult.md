# Results on the reverse kernels (`llc`)

`llc -O3 -verify-machineinstrs` on the `dump2/base` `.llir` kernels. Max VP and SLIL are the pass's own LMUL-weighted numbers before and after its transforms (32 vector registers). Spills and reloads are `Folded Spill` / `Folded Reload` in the assembly.

"none" is a true baseline: without a `--custom-*` flag the pass changes nothing. The COPY-to-VMV lowering (`LowerCopy`, see `DocSink.md`) is its own flag, `--custom-copy` (7, 3, 3 and 3 copies here). The table measures `--custom-reverse --custom-copy`; the last columns give each flag alone.

| kernel | none: max VP | none: spills / reloads | `--custom-reverse --custom-copy`: max VP | SLIL | spills / reloads | fires | `--custom-reverse` alone | `--custom-copy` alone |
|---|---|---|---|---|---|---|---|---|
| `reverse_v1` | 84 | 19 / 26 | 84 → 84 | 2785 → 2857 | 16 / 20 | 2 elementwise | 22 / 31 | 16 / 22 |
| `reverse_v2` | 75 | 15 / 17 | 75 → 67 | 3061 → 2899 | **9 / 9** | 8 elementwise | 10 / 10 | 15 / 17 |
| `reverse_v3` | 108 | 18 / 21 | 108 → 92 | 3575 → 3875 | **14 / 18** | 4 direct (2- and 3-step) | 18 / 23 | 17 / 22 |
| `reverse_v4` | 140 | 26 / 27 | 140 → 92 | 4316 → 4896 | **18 / 23** | 6 direct (2- and 3-step) | 22 / 28 | 23 / 25 |

The remats are the same with or without `--custom-copy`, but the spills depend on both together: on `reverse_v1`, `--custom-reverse` alone is worse than none (22 / 31 against 19 / 26), and `--custom-reverse --custom-copy` reaches 16 / 20.

The former `--custom-forward` reached 14 / 18 on `reverse_v3` and 19 / 24 on `reverse_v4`. The multi-step direct options reproduce the first and do slightly better on the second, because they can also replay the multiply (3 steps). Before the direct options were extended, `--custom-reverse` fired nothing on `reverse_v3` and `reverse_v4`: their direct options were single-step and their inputs dead, and the multiply that follows has no inverse.

On `reverse_v1`, the 2 elementwise remats free a register for only about 5 instructions, so max VP is unchanged and the small change in reloads is the register allocator reacting to slightly different live ranges.

All runs pass `-verify-machineinstrs`, and `--custom-reverse` alone gives the same assembly as `--custom-reverse --custom-remat` on all four kernels. On `s279.mir` (checked with `-run-pass=expandpseudos -verify-machineinstrs`), `--custom-reverse` alone runs the redundant-reload removal and one load remat, the same MIR as `--custom-remat --custom-reverse`; no reverse remat fires there. (`llc -start-before=expandpseudos s279.mir` crashes in the RISC-V assembly printer with an invalid register even without any flag, so `s279` cannot be checked through to assembly that way.)

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

`rvv_06-matrix-vector-multiplication_elf.py` (`gemv_kernel`) and `rvv_03-matrix-multiplication_elf.py` (`matmul_kernel`), compiled through Triton with `TRITON_RISCV_LMUL` = 1, 2, 4 and 8 (`-verify-machineinstrs`) and run on the board, without `--custom-*` flags (baseline) and with all of them (`--custom-copy --custom-sink --custom-a --custom-thresh --custom-remat --custom-reverse`). Max VP and SLIL are `ExpandPseudos`' own numbers before → after its transforms; spills and reloads are `Folded Spill` / `Folded Reload` in the kernel's assembly.

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
