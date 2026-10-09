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

## Which RVV instructions can be reversed

The tables below classify single RVV instructions. `y` is the result, `v` the value to rebuild, `K` the other source. "Implemented" refers to `ReversePatterns` / `matchReverseOp`, `matchGroupPlans`, `matchFlipPlan` and `matchPermPlan` in `ExpandPseudos.cpp`. "Possible" means the inverse exists but is not matched yet.

### Reversible

| RVV instruction (`y = ...`) | rebuild of `v` | live at the site | exact | status |
|---|---|---|---|---|
| `vadd.vv/vx` (`v + K`) | `vsub y, K` | y, K | exact | implemented (elementwise) |
| `vadd.vi` (`v + imm`) | `vadd.vi y, -imm` (not for imm = -16) | y | exact | implemented |
| `vsub.vv/vx` (`v - K` / `K - v`) | `vadd y, K` / `vsub K, y` | y, K | exact | implemented |
| `vrsub.vx/vi` (incl. `vneg`) | `vrsub` again | y, K | exact | implemented |
| `vxor.vv/vx/vi` (incl. `vnot`) | `vxor` again | y, K | exact | implemented |
| `vfadd` / `vfsub` / `vfrsub` | inverse op | y, K | rounded | implemented |
| `vfmul` / `vfdiv` / `vfrdiv` | inverse op | y, K | rounded | implemented |
| `vsext.vf2` / `vzext.vf2` | `vnsrl.wi y, 0` | y | exact | implemented (`narrow-of-widen`) |
| `vwadd(u).vx v, x0` (`vwcvt`) | `vnsrl.wi y, 0` | y | exact | implemented |
| `vfwcvt.f.f.v` | `vfncvt.f.f.w` | y | exact | implemented |
| `vfwcvt.f.x(u).v` | `vfncvt.rtz.x(u).f.w` | y | exact | implemented |
| `vrgather(ei16).vv v, idx`, constant permutation | gather by the inverse table | y | exact | implemented (`inverse-permutation-gather`) |
| same, self-inverse permutation (block swaps) | `vrgatherei16 y, idx` | y, idx | exact | implemented (self-inverse) |
| copies (`vmv.v.v`, `vmv<N>r`, `COPY`) | same value | y | exact | equivalence class |
| `vmul.vx v, c`, c an odd constant | `vmul.vx y, c^-1 mod 2^SEW` | y | exact | possible (integer `MUL` is never reversed today) |
| `vmacc` / `vmadd` (`y = a * b + v`) | `vnmsac`: `v = y - a * b` | y, a, b | exact | possible (tied passthru is rejected today) |
| `vfmacc` / `vfmadd` (accumulator `v`) | `vfnmsac` | y, a, b | rounded | possible |
| `vwadd(u).wv/wx` (`y = v + ext(b)`, `v` wide) | `vwsub(u).wv/wx` | y, b | exact | possible |
| `vsext` / `vzext.vf4/vf8` | 2 or 3 `vnsrl.wi 0` | y | exact | possible (only vf2 today) |
| `vfsgnjn.vv v, v` (`vfneg`) | itself | y | exact | possible |
| `vfmul` / `vfdiv` by a power of 2 | inverse op | y | exact without overflow / underflow | implemented as rounded |
| Zvbb `vror` / `vrol`, `vrev8`, `vbrev` | opposite rotate / itself | y | exact | possible (Zvbb only) |

### Not reversible alone

These instructions drop part of `v`. They become reversible only where the dropped part is still held by another value: a partner instruction's result, or a side value that is live at the site.

| RVV instruction | what is lost | reversible when | status |
|---|---|---|---|
| `vslidedown k` | lanes 0..k-1 | paired with `vslideup n-k` of the same `v` (rotation) | implemented (`inverse-slide-rotation`) |
| `vslideup k` | top k lanes | same rotation pair | implemented |
| `vsrl` / `vsll` by s | low / high s bits | paired with the shift by SEW-s and `vor` / `vxor` / `vadd` (bit rotation; s = 32 at e64 swaps adjacent 32-bit lanes) | implemented (`inverse-bit-rotation`) |
| `vsll` by s alone | high s bits | the high bits are known zero (e.g. after `vzext`): `vsrl` | possible (needs known bits) |
| `vwaddu.vv a, b` alone | `b` is mixed in | followed by `vwmaccu.vx -1, b` (zip) | implemented (`deinterleave-of-zip`) |
| `vnsrl.wi 0` / `vnsrl.w* SEW` | the other half | both halves exist (unzip pair): `vwaddu` + `vwmaccu` | implemented (`zip-of-deinterleave`) |
| `vid`, `vrsub`, `vrgather` x L, `vslidedown` (lane reverse) | each step alone | the whole group replayed on `y`; idx reused if live | implemented (`flip-of-flip`) |
| `vrgather` by a non-constant or non-permutation index (`vrgather.vi`, broadcast) | duplicated lanes / unknown inverse | the index is a constant permutation | no |
| `vnsra`, `vnclip(u)`, `vfncvt.*`, `vfcvt.x.f`, `vfcvt.rtz.*` | high bits, rounding, saturation | - | no |
| `vand`, `vor`, `vandn`, `vmin(u)`, `vmax(u)`, `vfmin`, `vfmax` | masked bits / losing operand | the complement is live (`v = (v & m) \| (v & ~m)`) | possible, rare |
| `vmerge`, `vcompress`, masked ops with a live passthru | masked-off lanes | - | no |
| `vmul.vv`, `vmul.vx` by even or unknown K, `vmulh*`, `vdiv`, `vrem`, `vwmul` | high bits, remainder, division by zero | K an odd constant (above) | no |
| `vsadd`, `vssub`, `vaadd`, `vasub`, `vsmul`, `vssrl`, `vssra` | saturation / rounding bit | - | no |
| `vfsqrt`, `vfrec7`, `vfrsqrt7`, `vfabs`, `vfsgnj(x)` with K | sign / precision | - | no |
| `vredsum`, `vredmax`, `vfred*` | lanes collapsed into one | - | no |
| compares (`vmseq`, `vmflt`, ...) | all but one bit | - | no |
| `vslide1up` / `vslide1down` | one lane | the lost lane is in a live scalar | possible, rare |
| `vid`, `vmv.v.x/i`, loads | no vector source | - | not a reverse case (direct remat or RA remat) |

### Data-movement instructions

#### Data movement that can be reversed

| instruction or group (`y = ...`) | effect | rebuild of `v` | live at the site | status |
|---|---|---|---|---|
| `vmv.v.v` (same VL), `vmv1r/2r/4r/8r.v`, `vmmv.m` (`vmand.mm m, m`), `COPY`, a whole `INSERT_SUBREG` / `REG_SEQUENCE` operand | copy | `y` itself | y | equivalence class |
| `vrgather.vv` / `vrgatherei16.vv v, idx`, idx a constant permutation `p` | `y[i] = v[p[i]]` | gather `y` by `q` (new table) | y | implemented (`inverse-permutation-gather`) |
| same, `p == q` (block swaps, neighbour swaps, byte swap through an e8 gather) | self-inverse | `vrgatherei16 y, idx` | y, idx | implemented (self-inverse) |
| `vid`, `vrsub`, `vrgather.vv` x L, `vslidedown` | lane reverse | the same group on `y` | y (idx if live) | implemented (`flip-of-flip`) |
| `vslidedown k` + `vslideup n-k` of the same `v` | rotation by `k` | rotation by `n-k` | y | implemented (`inverse-slide-rotation`) |
| `vslideup.vx/vi P, v, k` (tied passthru), `v`'s VL at most VLMAX - k | subvector insert at `k` (`tl.cat`) | `vslidedown y, k` at `v`'s VL | y | possible |
| `vsll s` + `vsrl SEW-s` + `vor` / `vxor` / `vadd` | bit rotation by `s` (no Zvbb `vror` on the board) | opposite rotation | y | implemented (`inverse-bit-rotation`) |
| `vsext` / `vzext.vf2`, `vwcvt(u).x.x.v` (`vwadd(u).vx v, x0`) | widen each lane | `vnsrl.wi y, 0` (`vncvt.x.x.w`) | y | implemented |
| `vsext` / `vzext.vf4` / `vf8` | widen each lane 4x / 8x | 2 or 3 `vnsrl.wi 0` | y | possible |
| `vfwcvt.f.f.v` (f16 to f32 with Zvfh, f32 to f64) | widen each lane | `vfncvt.f.f.w` | y | implemented |
| `vwaddu.vv a, b` + `vwmaccu.vx -1, b` | zip `a`, `b` into wide lanes | `vnsrl.wi y, 0` / `vnsrl.wi y, SEW` | y | implemented (`deinterleave-of-zip`) |
| `vnsrl.wi w, 0` + `vnsrl.wi w, SEW` | unzip `w` into both halves | `vwaddu` + `vwmaccu` | both halves | implemented (`zip-of-deinterleave`) |
| `y = vcompress v, m` + `z = vcompress v, ~m` | split `v` by `m` | `vrgather z, viota(~m)`, then `vrgather y, viota(m)` masked by `m` into it | y, z, m | possible (about 5 instructions) |
| `vmnot.m` (`vmnand.mm m, m`) | mask not | itself | y | possible (masks are one register) |

#### Data movement that cannot be reversed alone

| instruction | what is lost | reversible only if | status |
|---|---|---|---|
| `vslidedown.vx/vi k` alone | lanes 0..k-1 | the partner `vslideup` exists (rotation), or lanes 0..k-1 of `v` are not demanded (`vslideup y, k`) | rotation implemented; demanded lanes not tracked |
| `vslideup.vx/vi k` alone, `v` fills the register | top k lanes | the partner `vslidedown` exists (rotation) | implemented as a pair only |
| `vslide1up.vx` / `vfslide1up.vf v, x` | `v[n-1]` | `v[n-1]` is in a live scalar: `vslide1down y, s` | possible, rare |
| `vslide1down.vx` / `vfslide1down.vf v, x` | `v[0]` (`x` lands in lane `n-1`) | `v[0]` is in a live scalar: `vslide1up y, s` | possible, rare |
| insert element: `vmv.s.x t, x` + `vslideup v, t, idx` at VL `idx+1`, tail undisturbed | `v[idx]` | old `v[idx]` is in a live scalar: the same pair with it | possible, rare |
| `vmv.s.x` / `vfmv.s.f` into `v` | `v[0]` | `v[0]` is in a live scalar: `vmv.s.x y, s` | possible, rare |
| `vmv.x.s` / `vfmv.f.s` (and extract element = `vslidedown` + `vmv.x.s`) | every other lane (scalar result) | `n == 1` | no |
| `vrgather.vv` / `vrgatherei16.vv` by a constant non-permutation (repeated or missing lanes; an index of VLMAX or more gives 0) | lanes not selected | - | no |
| `vrgather.vv` / `vrgatherei16.vv` by a computed index | unknown permutation | - | no |
| `vrgather.vx` / `vrgather.vi` | all lanes but one (splat) | - | no |
| `vcompress.vm` alone | lanes where `m` is 0 | the complementary compress is live (first table) | possible |
| `vmerge.vvm` / `vxm` / `vim`, `vfmerge.vfm` (`vselect`) | lanes of `v` replaced by the other source | the replaced lanes are live in another value (a second `vmerge`) | no, rare |
| `vnsrl.wi 0` / `vnsrl.wi SEW` alone | the other half | the partner is live (first table) | implemented as a pair only |
| `vmv.v.v` with a smaller VL and tail undisturbed (`PseudoVMV_V_V` with a passthru) | lanes from VL on | the tail of `v` is not demanded | no |
| `EXTRACT_SUBREG` of one part of `v` (also tuple extract after `vlseg`) | the other parts | all parts are live: `REG_SEQUENCE` | possible |
| any instruction of the first table, masked (`v0.t`) or tail undisturbed | inactive / tail lanes take the passthru | those lanes of `v` are not demanded | no |
| `vsll` / `vsrl` / `vsra` / `vnsra` by `s` alone | high / low bits | the partner shift exists (rotation), or the lost bits are known | rotation implemented |
| `vredsum`, `vredmax(u)`, `vredmin(u)`, `vredand/or/xor`, `vfred*`, `vwred*` | lanes collapsed into one | - | no |
| `viota.m`, `vmsbf.m`, `vmsif.m`, `vmsof.m`, `vcpop.m`, `vfirst.m` | mask turned into a count or a prefix | - | no |
| `vmv.v.x` / `vmv.v.i`, `vfmv.v.f`, `vid.v`, `vmset.m` / `vmclr.m`, `PseudoClearVR` | no vector source | - | not a reverse case (direct remat) |
| loads: `vle`, `vle*ff`, `vlse`, `vluxei` / `vloxei`, `vlseg` / `vlsseg` / `vluxseg` / `vloxseg`, `vl<N>re<eew>`, `vlm.v`, `PseudoVRELOAD` | data comes from memory | - | not a reverse case (direct or RA remat while memory is unchanged) |
| stores: `vse`, `vsse`, `vsuxei` / `vsoxei`, `vsseg` / ..., `vs<N>r`, `vsm.v`, `PseudoVSPILL` | no vector result | - | not a reverse case |

The two tables above cover only the RVV instructions that move data without computing on it: lane permutations (gathers, slides, compress, merge), lane-width reshapes, scalar-lane moves, register and mask copies. They are limited to instructions that execute on the BPI-F3 (SpacemiT X60, `rv64imafdcv_..._zvfh_zvfhmin_zvkt`). Each listed instruction was run on the board under a SIGILL handler, at a SEW where its encoding is legal.

The tablegen files read for the list are `RISCVInstrInfoV.td`, `RISCVInstrInfoVPseudos.td`, `RISCVInstrInfoVSDPatterns.td`, `RISCVInstrInfoVVLPatterns.td` and every other vector `.td` in `llvm/lib/Target/RISCV`. Data-movement instructions defined there but left out because the board does not run them:

| extension | instructions | on the BPI-F3 |
|---|---|---|
| Zvbb / Zvkb | `vrev8`, `vbrev`, `vbrev8`, `vror`, `vrol`, `vwsll`, `vandn` | SIGILL |
| Zvzip | `vzip.vv`, `vunzipe.v`, `vunzipo.v`, `vpaire.vv`, `vpairo.vv` | SIGILL |
| Zvfbfmin | `vfwcvtbf16.f.f.v`, `vfncvtbf16.f.f.w` | SIGILL |
| XSpacemiT (XSMTVDotII) | `smt.vpack.vv`, `smt.vupack.vv`, `smt.vnpack(4).vv`, `smt.vnspack(4).vv` | SIGILL (`smt.vmadot` runs, but it is a dot product) |
| XSfmm, XSfvcp, XAndes, XTHead, Zvfofp8min, Zvvm | tile moves `sf.vtmv.*`, `sf.vc.*`, `nds.vfwcvt.s.bf16`, tile loads, ... | not in the board's ISA string; not probed |

`n` is the VL, `p` a constant permutation, `q` its inverse.

### Partially reversing a Triton op

A Triton op that lowers to several RVV instructions can be reversed one instruction at a time from its last one, since each rebuilt value joins `v`'s equivalence class and the next step can read it (`DocReverse.md`, "Chained pattern"). Only the instructions on the data path from `v` to `y` must be reversible. Instructions off that path (index builders such as `vid` / `vrsub`, constant loads, masks) are never inverted; their results only have to be live at the site, as the flip rebuild does with `idx`.

For an op `v -> i1 -> t1 -> ... -> i4 -> t4 -> i5 -> y`, peeling `i5` and `i4` gives `t3` back. If `i3` is in the second table:

- If the part `i3` drops is held by a value live at the site (the partner slide, the other `vnsrl` half, the complement of an `vand`), `t2` can be rebuilt and peeling continues down to `v`.
- If not, `v` cannot be rebuilt by reversal. `t3` and `t4` still can, which helps when one of them has a distant use; `v` itself is left to a direct option from a root that is still live.

## Limitations

- Profitability is the gap rule only. A rebuild of 5 instructions (reload) or 11 (flip without a live index) is applied as readily as a single instruction. Weighing the rebuild's length against the freed live range, or requiring a longer gap for longer rebuilds, would avoid the 07 slowdown.
- The self-inverse rebuild only applies where the original index is live. Extending a small index group (LMUL 2-4) to free an LMUL-8 value is not considered.
- Each reload rebuild adds its own constant-pool entry, even when several remats need the same inverse table.
- The flip rebuild depends on the shuffle form of `tl.flip` (`TRITON_CPU_FLIP_TO_SHUFFLE=1`). The default xor-chain lowering gives `matchFlipPlan` nothing to match.
