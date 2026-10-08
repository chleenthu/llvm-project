# Vector Reverse Rematerialization: Lane Flip and Permutation Gather

`--custom-reverse` (see `DocReverse.md`) recomputes a value `v` from a later value `y` derived from it, when `v` has a use at least `RematGap` instructions after the previous one. Besides the elementwise inverses (`v = y - k` from `y = v + k`), the options recorded on `v`'s live range include rebuilds that undo a whole *group* of instructions moving data across lanes. This document covers the two that come from shuffles: the lane flip (`flip-of-flip`) and the constant permutation gather (`inverse-permutation-gather`). Both are exact.

Both are matched in `ExpandPseudos.cpp` (`matchFlipPlan`, `matchPermPlan`) and emitted by `ProcessReverseRematChain` like any other option.

## How a shuffle reaches the pass

A Triton `vector.shuffle` becomes an LLVM `shufflevector` and is lowered by the RISC-V backend according to its mask. For the single-source masks seen in these kernels:

| mask (64 x f32 / i32, VLEN 256, LMUL 8) | RVV code | instructions |
|---|---|---|
| reverse all lanes `[63, ..., 1, 0]` | `vid.v`, `vrsub.vx` (index n-1-i), one `vrgather.vv` per LMUL1 register (in reverse register order, VLMAX), `vslidedown.vx` by VLMAX - n | 1 + 1 + 8 + 1 |
| rotation, e.g. swap halves `[32..63, 0..31]` | `vslidedown.vx` + `vslideup.vx` | 2 |
| swap blocks of 16/8/4/2 lanes `[16..31, 0..15, ...]` | `vrgatherei16.vv` at e64 (pairs of 32-bit lanes as one element) | 1 + index |
| swap neighbouring lanes `[1, 0, 3, 2, ...]` | `vrgatherei16.vv` at e32 | 1 + index |

"+ index" is a constant-pool table of lane numbers: `auipc` + `addi` + `vle8.v` + `vsext.vf2` (e8 to e16). The lowering loads it once per block and shares it among all gathers using the same mask.

## Lane flip (`flip-of-flip`)

`tl.flip(x, 0)` on a vector is a lane reverse. Reversing twice gives the input back, so `v` can be rebuilt as `flip(y)` from `y = flip(v)`.

`matchFlipPlan` recognizes the reverse as LLVM lowers it:

```text
idx      = vrsub.vx (vid.v), c            ; c = n - 1, or VLMAX - 1
T.sub_k  = vrgather.vv v.sub_(L-1-k), idx ; k = 0 .. L-1, one per LMUL1 register, VLMAX
y        = vslidedown.vx T, VLMAX - n     ; VL n
```

When `n` fills exactly one LMUL1 register at the minimum VLEN, this is a single gather with `c = n - 1` and VL `n`, and no slide. The rebuild replays the same group on `y`: it clones `vid`, `vrsub`, the gathers and the slide, with `y` as the source and `v`'s new copy as the result. If `idx` is still live at the site, the rebuild reuses it and skips `vid`/`vrsub`.

### Dependency on `TRITON_CPU_FLIP_TO_SHUFFLE`

`tl.flip` is not a dedicated op. The frontend reshapes the flipped dim to `(2, 2, ..., 2)` and applies log2(n) xor-sum steps. The Triton-CPU pass `ConvertFlipToShuffle` turns that chain back into one `vector.shuffle`, but only with `TRITON_CPU_FLIP_TO_SHUFFLE=1` (off by default). Without it, the xor chain is lowered step by step, and there is no single flip group for `matchFlipPlan` to find.

### Result on `flip_live`

`rvv_flip_elf.py`'s `flip_live` keeps `x` live across `y = tl.flip(x, 0)` (64 x i32, LMUL 8) and four products of `y` with loaded weights, then uses `x` again. Compiled with Triton, LMUL 8, `-verify-machineinstrs`. Max RP is the pass's own LMUL-weighted vector register pressure before register allocation (32 registers available). Spills and reloads are `Folded Spill` / `Folded Reload` in the assembly:

| `TRITON_CPU_FLIP_TO_SHUFFLE` | flags | max RP | spills | reloads | remats |
|---|---|---|---|---|---|
| off (default) | none | 144 | 33 | 36 | - |
| off (default) | `--custom-reverse` | 144 | 33 | 36 | 0 |
| `1` | none | 56 | 8 | 11 | - |
| `1` | `--custom-reverse` | **48** | **6** | **8** | 1 `flip-of-flip` |

With the shuffle, `x` is killed right after the flip and rebuilt as `flip(y)` just before the xor that needs it again. The index was not live there, so the rebuild is the whole group: `vid`, `vrsub`, 8 gathers and the slide (11 instructions). The kernel still gets shorter overall (164 to 137 instructions) because of the spill code it removes.

On the board, `flip_live` passes in both shuffle modes, at 0.0395 ms (off) and 0.0326 ms (on), without `--custom-reverse`.

For the other kernels in `rvv_flip_elf.py` and `rvv_cpu_flip_elf.py`, the shuffle setting does not change spills:

| kernel | shuffle | max RP | spills | reloads |
|---|---|---|---|---|
| `flip1d` | off / `1` | 14 / 5 | 0 / 0 | 0 / 0 |
| `flip_kernel` 1x16x64 i32, dim 0 | off / `1` | 128 / 128 | 13 / 13 | 13 / 13 |
| `flip_kernel` 1x16x64 i32, dim 1 | off / `1` | 128 / 128 | 12 / 12 | 12 / 12 |

In `flip_kernel`, flipping dim 0 (size 1) does nothing. Flipping dim 1 reverses the order of 16 rows of 64 lanes, each exactly one LMUL8 group, so it only renames registers and store addresses: both modes emit the same assembly, with no gather or slide. Its pressure comes from the 16 x 64 tile itself (128 registers at VLEN 256).

## Permutation gather (`inverse-permutation-gather`)

`matchPermPlan` recognizes a gather by a constant permutation:

```text
idx = vle<eew> %const.k  [; vsext.vf2 / vzext.vf2]   ; p, a permutation of 0..n-1
y   = vrgather.vv / vrgatherei16.vv  v, idx          ; y[i] = v[p[i]], VL n
```

The first `n` entries of the constant must be a permutation of `0..n-1`. Its inverse `q` (`q[p[i]] = i`) gives `v[j] = y[q[j]]`, which is another gather.

### Reload rebuild

`q` is written to a new constant-pool entry and loaded like the original index:

```text
auipc  t, %pcrel_hi(new entry)
addi   t, t, %pcrel_lo(...)
vle8.v      i8, (t)
vsext.vf2   i16, i8
vrgatherei16.vv  v', y, i16
```

That is 5 instructions (4 without the extension), and a new constant per remat.

### Self-inverse rebuild

When `p` is its own inverse (`q == p`), the gather undoes itself, and the rebuild only needs the index the original gather already used:

```text
vrgatherei16.vv  v', y, idx         ; idx is the original gather's index
```

Every block swap is self-inverse, so this covers each butterfly step of a reduction. `matchPermPlan` then offers a second plan that takes `idx` as an input, alongside `y`. The site check treats `idx` like any other input, so this plan only qualifies where `idx` is still live (or dies right before the site). Where both qualify, it wins on instruction count. `-debug-only=expandpseudos` labels it `ReverseRemat[inverse-permutation-gather, self-inverse]`.

## Results on 07 (BF16 GEMV)

`rvv_07-matrix-vector-multiplication-bf16_elf.py`: `BLOCK_SIZE_M = 16`, `BLOCK_SIZE_N = 64`, so each program reduces 16 rows of 64 lanes per loop iteration. `tl.sum` along the row becomes a 6-level butterfly per row (swap 32, 16, 8, 4, 2, then 1 lanes): 16 x 6 = 96 `vector.shuffle`. The 32-lane swap lowers to a slide pair, the others to `vrgatherei16` with one of 5 shared index tables.

The scheduler interleaves the 16 rows, so one step of a row looks like this:

```text
%371 = vand.vx  %369, %288             ; v: tree value of row r (BF16 truncation)
%373 = vrgatherei16 %371, %314         ; y = swap(v)                 <- U1
  ... 11 instructions of row r+1 ...
%376 = vfadd %373, %371                ; v needed again               <- U2
```

The gap between U1 and U2 is at least `RematGap`, and `y` is live at U2 because the `vfadd` reads it, so `v = swap(y)` qualifies. The same shape repeats at every level of every row.

`06` (FP32 GEMV, `BLOCK_SIZE_M = 1`, `BLOCK_SIZE_N = 512`) has the same reduction but only one row. Its butterfly runs serially, each `vfadd` right after its gather, so no value has a gap and nothing fires.

### Remats

Triton pipeline, LMUL 8, `--custom-reverse`:

| inverted gather | remats |
|---|---|
| e64, swap 16 | 9 |
| e64, swap 8 | 1 |
| e64, swap 4 | 3 |
| e64, swap 2 | 14 |
| e32, swap 1 | 14 |

The slide-pair level (swap 32) never fires: its value is consumed right away.

Before and after the self-inverse rebuild:

| | before | after |
|---|---|---|
| remats | 41, all reload a new table (5 instructions) | 41: 12 self-inverse (1 instruction), 29 reload |
| loop instructions | 511 → 716 | 511 → 668 |
| new constant-pool tables | 41 | 29 |

The 12 self-inverse sites are those where the index table is still read by a later row's gather. At the other 29 the index has died, and reusing it would extend it, so the reload plan is used there.

### Correctness and time on the board

The BF16 kernel truncates every product and every tree step to BF16 (`vand` with `0xffff0000`), sums the 16 blocks in FP32, and truncates the result. A reference that does exactly this matches the board bit for bit (atol 0), with and without `--custom-reverse`:

| flags | result | time per launch |
|---|---|---|
| none | PASS | 0.81 ms |
| `--custom-reverse` (self-inverse rebuild) | PASS | 0.99 ms |

`--custom-reverse` is still about 20% slower on 07: freeing an LMUL-8 value for 11 instructions does not pay for the 29 remaining 5-instruction reloads. The gap rule has no cost model (see Limitations).

## Limitations

- Profitability is the gap rule only. A rebuild of 5 instructions (reload) or 11 (flip without a live index) is applied as readily as a single instruction. Weighing the rebuild's length against the freed live range, or requiring a longer gap for longer rebuilds, would avoid the 07 slowdown.
- The self-inverse rebuild only applies where the original index is live. Extending a small index group (LMUL 2-4) to free an LMUL-8 value is not considered.
- Each reload rebuild adds its own constant-pool entry, even when several remats need the same inverse table.
- The flip rebuild depends on the shuffle form of `tl.flip` (`TRITON_CPU_FLIP_TO_SHUFFLE=1`). The default xor-chain lowering gives `matchFlipPlan` nothing to match.
