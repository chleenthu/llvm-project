# Reverse Rematerialization (`--custom-reverse`)

`ExpandPseudos::ProcessReverseRematChain` implements the technique from Bahi & Eisenbeis, ["Register Reverse Rematerialization"](https://inria.hal.science/inria-00607323) (2011): instead of keeping a value alive until its last use, or spilling and reloading it, recompute it *backwards* from a later value it helped produce, using the algebraic inverse of the operation. It generalizes to a whole chain of such inversions (paper §2.2, "sequences with more than one instruction", Figure 2), not just one hop.

Concretely, given
```text
%C = PseudoVFADD_VV_M8_E32 undef %C, %A, %B, ...   # C = A + B
```
`%B` is normally live from its own definition until its last real use. If `%C` is still alive at some later point (its own last use), `%B` can instead be recomputed there as `%C - %A`, and every use of `%B` after that point rewritten to the recomputed value. `%B`'s original live range then ends right after this add, instead of stretching to its last use — the same live-range-shortening goal as `--custom-remat`, but by inverting an arithmetic op instead of reloading from memory.

## Single-hop pattern

```text
%C = PseudoVFADD_VV_M8_E32 undef %C, %A, %B, frm, %vl, sew, policy, implicit $frm
...
%I = <last real use of %C>
%J = <uses %B again, much later>
```

becomes

```text
%C = PseudoVFADD_VV_M8_E32 undef %C, %A, %B, frm, %vl, sew, policy, implicit $frm
...
%I = <last real use of %C>
%B2 = PseudoVFSUB_VV_M8_E32 undef %B2, %C, %A, frm, %vl, sew, policy, implicit $frm   # new
%J = <rewritten to use %B2>
```

## Chained pattern

The same idea repeated over several links of vector-scalar ops, each invertible with the *same* scalar operand:

```text
%B = PseudoVFADD_VFPR32_*(undef %B, %A, %k1, ...)   ; B = A + k1
%C = PseudoVFADD_VFPR32_*(undef %C, %B, %k2, ...)   ; C = B + k2
%D = PseudoVFMUL_VFPR32_*(undef %D, %C, %k3, ...)   ; D = C * k3
...                                                  ; D's only forward uses, no gap
%I = <uses %C again, much later>
%J = <uses %B again, much later>
%K = <uses %A again, much later>
```

Each of `%A`, `%B` and `%C` is only needed once more, much later, so each is reverse-rematerialized: `%C2 = %D / k3` replaces `%C`'s late use, `%B2 = %C2 - k2` replaces `%B`'s late use, `%A2 = %B2 - k1` replaces `%A`'s late use. The key point is that `%B2` is computed from `%C2` (short-lived, still alive near `%I`), not from the original `%C` (which by then has died right after producing `%D`) — otherwise `%B2` would have to be computed too early to shrink anything.

## When and how: remat timing, options on the live range

`ProcessReverseRematChain` decides *when* to rematerialize with the timing of `--custom-remat`, and *how* the way the V8 JIT does it (Vardanyan, Asryan, Buchatskiy, "Integrated Register Rematerialization in JavaScript V8 JIT Compiler"): every operand of a value that tells how to recompute it is recorded on the value's live range, and the rematerialization site picks one of them.

In the V8 paper's example

```text
3: c = a + b      OutputC3: direct,  c = a + b   (from a, b)
5: e = c - d      InputC5:  reverse, c = e + d   (from e, d)
6: f = c + b      InputC6:  reverse, c = f - b   (from f, b)
```

the live range of `c` carries all three options, so each use position of `c` can be recomputed by any of them, as long as the values that option reads are live there.

### Steps

1. **Options on operands.** For each vector vreg `%v` of a block, `BuildInterval` collects a `RematInterval`: the sorted use positions and a list of `RematOption`s.
   - `OutputV` (direct): `%v`'s own def, if `matchDirectOp` accepts it (an unmasked RVV op with an undef passthru that only computes), replayed alone or together with the defs of its vector source, back to a root (see "Multi-step direct options" below). There is one option per chain length.
   - `InputV` (reverse): every use `q` of `%v`, or of a copy of `%v` made by an earlier step, that can be inverted. These are the elementwise inverses below plus the group rebuilds (widen/narrow, zip/unzip, slide and bit rotations, lane flip, permutation gather).
2. **Propagate to the live range.** The options are kept with the interval, keyed by the position of the operand they came from. LLVM's `LiveInterval` has no room for extra data, and `LiveIntervals` is not kept up to date by this pass's transforms, so the interval is modelled per block on instruction positions (`ReverseBlockModel`).
3. **Timing (from `--custom-remat`).** Walk the uses of `%v`. Every pair of consecutive uses `U1 < U2` with at least `RematGap` (6) instructions between them is a candidate site, earliest first. As with cheap ALU ops in `--custom-remat`, one use before the gap is enough.
4. **Pick an option at the site.** An option qualifies only if it comes from an operand before `U2`, and if:
   - the values it reads (`Y`, `K`, or copies of them) hold their value at the insertion point `P`, and none of them was itself computed from `%v`;
   - its scalar operands (VL, amounts) and `$frm` are not redefined before `P`;
   - every moved use reads only the first VL elements (reverse only; a direct clone recomputes the whole register);
   - the register classes fit.

   `P` is one instruction before `U2`, as in `--custom-remat`. If a vector input dies earlier, `P` moves up to just after that input's last use, so the new instruction reuses the dying register. The hole left in `%v`'s live range, `P − U1 − 1`, must still be at least `RematGap − 1`, the same as what `--custom-remat` leaves. Among the options that qualify, the pass prefers the latest `P`, then an exact rebuild, then the fewest new instructions, then the option nearest the site.
5. **Rewrite.** Insert `%v2 = REV(Y', K')` (or the replayed chain) at `P` and rewrite `U2` and every later use of `%v` to `%v2`. `%v2` joins `%v`'s equivalence class, so a later step can recompute another value from it (Figure 2(c) chains). The block is re-analyzed and the walk repeats until nothing changes.

`--custom-reverse` uses both kinds of options, and implies `--custom-remat`: it also runs that flag's redundant-reload removal and load / cheap ALU rematerialization first, so `--custom-reverse` alone gives exactly the code of `--custom-reverse --custom-remat`. `-debug-only=expandpseudos` prints each interval that has a distant use, with its options:

```text
RematInterval %28 uses 44 60
  @44 elementwise: %62 = PseudoVFADD_VV_M8_E32 undef %62, %28, %44, ...
  @60 elementwise: %78 = PseudoVFADD_VV_M8_E32 undef %78, %28, %71, ...
ReverseRemat[elementwise, rounded]: recompute %28 from %62, %44 ...
```

Here the gap is between uses 44 and 60. Only the option at 44 lies before it. `%28` is recomputed as `%62 − %44` right after `%62`/`%44` die, which leaves a hole of more than `RematGap` instructions.

### Reversible operations

`q` must be an unmasked `PseudoV<OP>_<VV|VX|VI|VFPR*>_<LMUL>[_E<SEW>]` with an undef passthru:

- `ADD`→`SUB` (`ADD_VI`→`ADD_VI` with `-imm`), `SUB`→`ADD` / `SUB` (swapped), `RSUB`→`RSUB`, `XOR`→`XOR`: exact.
- `FADD`↔`FSUB`, `FRSUB`→`FRSUB`, `FMUL`↔`FDIV`, `FRDIV`→`FRDIV`: rounded.
- For VV forms both operand slots are tried. Integer `MUL` and shifts are never reversed.

### Interval shape

A value read in another block (live out) is skipped, because its register stays busy anyway. A vreg defined more than once in the block, for example a masked load whose passthru is tied to `vmv.v.i 0`, is modelled from its last def on. A value whose def reads no vector register (`vid.v`, `vmv.v.x`) is skipped, because RA already rematerializes it.

### Multi-step direct options

A direct option clones `%v`'s def, so its inputs must be live at the site. Often they are not: in `reverse_v3`, `c = (a << s) + k` is used early and again much later, but `a << s` dies right after `c` is computed. Only `a` itself is live late (it is used again at the end of the kernel).

So the direct option also follows the def's vector source (operand 2) back through further defs that `matchDirectOp` accepts, in the same block, up to 8 steps, and records one option per chain length:

```text
%45 = PseudoVSLL_VI_M8 undef %45, %27, s       ; a << s
%47 = PseudoVADD_VX_M8 undef %47, %45, k       ; c = (a << s) + k
...
RematInterval %47 uses 35 67
  1 step:  %47 = ADD(%45, k)          root %45: dead at 33, never qualifies
  2 steps: %47 = ADD(SLL(%27, s), k)  root %27 (a): live late, qualifies
```

The root of each option is its input, like `Y` of a reverse option, so the same liveness rule picks the shortest chain whose root is still live at the site. A link's second source is reused as is: a scalar or immediate is checked like VL, and a vector that the chain does not compute itself becomes a second input (at most one is allowed). At the site, the links are cloned from the root to `%v` with fresh registers, the last one writing `%v2`. The original def, and any link whose result fed only the next one, is erased once `%v2` has taken over all uses. Replaying the original instructions is exact.

This is Figure 2(b) of the paper ("rematerialize B, C, D from A"). It replaces the former `--custom-forward` (`ProcessForwardRematChain`), which replayed such chains on its own, with no gap rule and only for values with exactly two uses.

### What changed from the pressure-driven version

The previous version searched every gap of every value for the one with the largest "freed register × instructions" benefit at points where VR pressure exceeded `--custom-reverse-pressure`. That search, `--custom-reverse-pressure` and `--custom-reverse-peak-only` are gone. Profitability is now the remat gap rule alone. The previous version is kept as `git stash` entry "backup: pressure-driven -custom-reverse before interval/gap rewrite".

### Results

`llc -O3 -verify-machineinstrs` on the `dump2/base` `.llir` kernels. Max VP and SLIL are the pass's own LMUL-weighted numbers before and after its transforms (32 vector registers). Spills and reloads are `Folded Spill` / `Folded Reload` in the assembly.

`--custom-remat` fires nothing on these kernels (they only have masked loads, and the compares it could recompute have inputs that die early), so its column is the baseline every flag starts from: the COPY lowering that runs whenever any `--custom-*` flag is set.

| kernel | none: spills / reloads | `--custom-remat` (baseline): spills / reloads | `--custom-reverse`: max VP | SLIL | spills / reloads | fires |
|---|---|---|---|---|---|---|
| `reverse_v1` | 19 / 26 | 16 / 22 | 84 → 84 | 2785 → 2857 | 16 / 20 | 2 elementwise |
| `reverse_v2` | 15 / 17 | 15 / 17 | 75 → 67 | 3061 → 2899 | **9 / 9** | 8 elementwise |
| `reverse_v3` | 18 / 21 | 17 / 22 | 108 → 92 | 3575 → 3875 | **14 / 18** | 4 direct (2- and 3-step) |
| `reverse_v4` | 26 / 27 | 23 / 25 | 140 → 92 | 4316 → 4896 | **18 / 23** | 6 direct (2- and 3-step) |

The former `--custom-forward` reached 14 / 18 on `reverse_v3` and 19 / 24 on `reverse_v4`. The multi-step direct options reproduce the first and do slightly better on the second, because they can also replay the multiply (3 steps). Before the direct options were extended, `--custom-reverse` fired nothing on `reverse_v3` and `reverse_v4`: their direct options were single-step and their inputs dead, and the multiply that follows has no inverse.

On `reverse_v1`, the 2 elementwise remats free a register for only about 5 instructions, so max VP is unchanged and the small change in reloads is the register allocator reacting to slightly different live ranges.

All runs pass `-verify-machineinstrs`, and `--custom-reverse` alone gives the same assembly as `--custom-reverse --custom-remat` on all four kernels. On `s279.mir` (checked with `-run-pass=expandpseudos -verify-machineinstrs`), `--custom-reverse` alone runs the redundant-reload removal and one load remat, the same MIR as `--custom-remat --custom-reverse`; no reverse remat fires there. (`llc -start-before=expandpseudos s279.mir` crashes in the RISC-V assembly printer with an invalid register even without any flag, so `s279` cannot be checked through to assembly that way.)

## Result on the TSVC `reverse_v1` kernel (single-hop, LMUL8)

`BLOCK_SIZE=128` splits into two interleaved 64-lane LMUL8 chunks (`reverse-v1.mir`). The pass fires on both, reversing `%A` (the `Vs2` operand is tried first) from `%C = %A + %B`:

```text
ReverseRematChain: recompute %20 from %36 after %40 = PseudoVFADD_VV_M8_E32 undef %40, %36, %37, ...   # I0 = C0+F0
  for use in <K0 = A0+J0>
ReverseRematChain: recompute %24 from %35 after %39 = PseudoVFADD_VV_M8_E32 undef %39, %35, %38, ...   # I1 = C1+F1
  for use in <K1 = A1+J1>
```

**Measured**: `grep -c "Folded Spill" reverse-v1.s` went from 4 (baseline) to 7 with `--custom-reverse` — worse, not better. `BLOCK_SIZE=128` interleaves two LMUL8 chains in the same block; shrinking one value's live range in one chain does not reduce the region-wide simultaneous demand across both chains, and the recomputed value is itself spilled almost immediately in the generated assembly. Register pressure here is a two-chain, whole-region problem, not a single live range that this local, per-instruction pattern match can fix by itself.

## Result on the `reverse_v2` kernel (chained, LMUL4)

`BLOCK_SIZE=128` splits into four interleaved 32-lane LMUL4 chunks (`reverse-v2.mir`, from `rvv_reverse_v2_elf.py`, whose source uses the same `a,b,c,d,e,f,g,h,i,j,k` chain as the paper's Figure 2). `f = d & 0xFFFFFFFF` is folded away by earlier passes (bit-pattern identity), so `d` feeds `e`, `g` and `h` directly with no gap and is never itself reverse-rematerialized — but `c`, `b` and `a` each have exactly the "used once early, needed again much later" shape, and the chain fires on all three, four times over (once per interleaved chunk):

```text
ReverseRematChain: recompute %50 from %51 after ...   # c2 = d / scalar
ReverseRematChain: recompute %49 from %52 after ...   # (another chunk's c)
ReverseRematChain: recompute %41 from %134 after ...  # b2 = c2 + 3
...
```

**Measured**: `grep -c "Folded Spill" reverse-v2.s` went from 14 (baseline) to **7** with `--custom-reverse` — spill sites cut in half. Predicted peak vector-register pressure (computed LIS-independently from `Usage`/`VRegLIL`, see `ExpandPseudos::computeCurrentUsage`) dropped from 76 to 40, and SLIL from 2540 to 2065.

## Limitations

- **Floating-point precision**: the recomputed value is not bit-identical to the original under FP rounding (the paper's own §4.5 flags this). This transform trades exactness for register pressure and should not be applied to precision-sensitive code without accepting that tradeoff.
- Profitability is only the `--custom-remat` gap rule (`RematGap`, a fixed constant). There is no register pressure or cost model, and an option is not weighed against a spill reload.
- Only one basic block at a time. Values live out of the block are not split.
- Verified with `llc` on the four `reverse` kernels, and end to end on the RISC-V board: `rvv_reverse_v1_elf.py` to `rvv_reverse_v4_elf.py` pass their `expected=` check with `--custom-reverse` (and `-verify-machineinstrs`). Through Triton's codegen options the same kernels fire 0, 5 (elementwise), 2 (direct) and 3 (direct) times.
- Options such as `--custom-reverse` reach `ExpandPseudos` through Triton via `TRITON_RISCV_LLVM_ARGS` (parsed in `third_party/cpu/llvm.cc`; `libtriton.so` loads the LLVM shared libraries from `~/llvm-project/install`, so `ninja install` is enough after a change). The RVV tutorials `rvv_01` to `rvv_08` were run on the board this way with `--custom-reverse`; see `DocVecReverse.md` for 07, where reverse remat fires and the result is bit-exact.
