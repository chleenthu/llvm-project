# Remat Optimizations (`--custom-remat`)

The hidden `--custom-remat` option enables two related transforms in `ExpandPseudos`, run in this order: redundant masked reload removal, then load rematerialization.  Both shrink live ranges of vector register groups before register allocation, at the pass position right after the machine scheduler.

# Redundant Masked Reload Optimization

`ExpandPseudos::ProcessRedundantReload` is the first transform.  It replaces a masked vector reload that is an identity on an already-loaded register with a fresh full load into a new virtual register.  This creates a hole in the live range of the original register, so the physical register group can be used by other values in between.

The transformation is only applied within one machine basic block.

## Motivation

In TSVC `s279` the same array `@d[i]` is loaded twice in the loop body.  The second load comes from an IR `select(mask, load p, X)` where `X` is the first load of `p`.  It is folded into a masked load whose passthru is the first load:

```text
%60 = VL4RE32_V %38                                   # full load of d[i]
...  uses: PseudoVMUL_VV_M4, COPY %52  (last real use)
...  about ten instructions that do not need %60
$v0 = COPY %32
%60 = PseudoVLE32_V_M4_MASK %60, %38, $v0, ...        # reload, tied passthru %60
%68 = PseudoVMACC_VV_M4 %68, %60, %64, ...            # final use
```

Nothing stores to `@d` in between (the stores go to `@b` and `@c`), so active lanes reload the same data and inactive lanes keep `%60`.  The masked load does not change `%60`.

Because the passthru operand is tied to the destination, the masked load reads the old `%60`.  The register is therefore live across the whole gap, `%60` has a single contiguous segment with no hole, and register allocation keeps a `vrm4` group busy for the entire span.

## Intended pattern

```text
%x = VL<n>RE<eew>_V %p              # full load
...                                 # no store that may alias, no call
%x = PseudoVLE<eew>_V_M<n>_MASK %x, %p, $v0, ...
... uses of %x
```

is rewritten to:

```text
%x = VL<n>RE<eew>_V %p
...
%y = VL<n>RE<eew>_V %p              # fresh vreg, same memoperand
... uses of %y
```

Result on `s279.mir`:

```text
%128:vrm4nov0 = VL4RE32_V %38 :: (load ... from %ir.6 ...)
...
%68 = PseudoVMACC_VV_M4 %68, %128, %64, ...
```

`%60` now dies at `COPY %52`.  The reload defines `%128`, a separate value.

## What `ProcessRedundantReload` actually checks

For each instruction whose name starts with `PseudoVLE` and ends with `_MASK`:

1. The result and the passthru (operand 1) are the same virtual register, and the base (operand 2) is a virtual register.
2. The virtual register has exactly two definitions: this instruction and one earlier definition.
3. The earlier definition is in the same block, is named `VL<digits>RE<digits>_V`, uses the same base register, and has the same memoperand IR value and offset.
4. Between the two loads there is no call, no instruction with unmodeled side effects, and no store that may alias.
5. All uses of the register are in this block.  Uses after the masked load are rewritten.

Two accesses are treated as non-aliasing only if their underlying objects are different identified objects, for example different globals.  Anything else counts as possibly aliasing, so the transform is conservative.

When these hold, the pass:

- clones the earlier full load with a new virtual register as its result,
- inserts it where the masked load was,
- rewrites the later uses to the new register and clears their kill flags,
- erases the masked load.

# Load Rematerialization

`ExpandPseudos::ProcessRematLoads` is the second transform of `--custom-remat`.  It reloads a whole register load, or recomputes a cheap ALU op (see below), right before a use that is far from the previous uses, so the register group is free in between.  RA's own remat cannot do this: loads from memory that is stored to are not trivially rematerializable.

## Intended pattern

```text
%x = VL4RE32_V %p
use1 %x
use2 %x
... at least 6 instructions ...
use3 %x
```

becomes:

```text
%x = VL4RE32_V %p
use1 %x
use2 %x
...
%y = VL4RE32_V %p        # clone of the load, fresh vreg
other                    # one non-debug instruction in between
use3 %y                  # this and all later uses are rewritten
```

Result on `s279`: `%30 = VL4RE32_V %29` is used by `PseudoVMSLE_VI_M4` and `PseudoVMSLT_VV_M4`, then about ten instructions later by `PseudoVMSGT_VI_M4`.  A clone `%129 = VL4RE32_V %29` is inserted two instructions before `VMSGT_VI_M4` (one unrelated instruction, `%55 = PseudoVMUL_VV_M4`, sits in between), so the register of `%30` is free in the middle.

## What `ProcessRematLoads` actually checks

For each `VL<digits>RE<digits>_V` load:

1. The loaded register and the base register are virtual and each has one definition.
2. All uses of the loaded register are in the same block.
3. At least `RematMinUses` (2) uses occur before the gap, so a value used only once early, such as `%60` in `s279`, is not reloaded.
4. The first use that comes at least `RematGap` (6) non-debug instructions after the previous use is the target.
5. No call, side-effecting instruction or possibly aliasing store (see the alias rule above) lies between the original load and the target.
6. The base register is still used at or after the insertion point, or in another block.  Otherwise the clone would only extend the base's live range.

The pass clones the load with a new virtual register and inserts it one non-debug instruction before the target (remat, another instruction, use), so the reload has some distance to the use, rewrites the target and later uses to the new register and clears kill flags.  The clone is queued again, so a later distant use can be split as well.

## Cheap ALU ops

`ProcessRematLoads` also handles cheap pure instructions: `PseudoVMS*` mask compares (such as `PseudoVMSLT_VV_M4`) and `PseudoVMV_V_I*` splats.  They must not touch memory, have no side effects and define one register.  The same gap rule applies, with these differences:

- One use before the gap is enough (loads need `RematMinUses`, 2).
- There is no memory hazard check.
- Every virtual register the instruction reads must have a single def and must already be live at the insertion point, that is used at or after it (or in another block).  Otherwise the clone would only lengthen another live range, so it is skipped.  Undef operands are ignored.
- A tied use of the result (the undef passthru of a splat) is renamed together with the def in the clone.

Example (hand-written MIR, `-custom-remat`).  The compare `%3` is used early and again about ten instructions later, and its inputs `%1` and `%2` are live at the second use:

```text
early-clobber %3 = PseudoVMSLT_VV_M4 %1, %2, ...
%4 = PseudoVMAND_MM_B8 %3, %3, ...
... ten instructions ...
%13 = PseudoVADD_VV_M4 ...
early-clobber %15 = PseudoVMSLT_VV_M4 %16, %2, ...   # clone
%14 = PseudoVADD_VV_M4 ...
$v0 = COPY %15                                        # was COPY %3
```

(`%16` is the reload of `%1` made by the load remat.)

In `s279` the compare `%44 = PseudoVMSLT_VV_M4 %30, %68` is a candidate but is skipped: `%68` has two defs, and `%30` dies before the second use.

## Limitations

- Destinations of loads created by both transforms are recorded in `RematRegs`, and `ProcessInSameBlock` skips them so `--custom-sink` does not move them again.
- The distance and use count are fixed constants, with no register pressure or cost model.  Each remat adds one load.
- It only handles whole register loads and the cheap ALU ops above, with all uses in one block.
- Verified only on the MIR after `-run-pass=expandpseudos` and on the generated assembly of `s279`.  The binary was not executed.

Measured on `s279`, loop body:

| | instructions | loop loads | highest vector reg |
|---|---|---|---|
| base | 110 | 8 | v28 |
| `--custom-remat` | 110 | 9 | v28 |

The extra load comes from the rematerialization.  The `vmv4r.v` copy is removed by the first transform.

# Group Sink to First Use

`ExpandPseudos::FindFirstUseToSinkToGroup` is the fallback used by `ProcessInSameBlock` when `SinkInSameInstructionGroup` fails.  It moves the load group later, but not all the way to the consumer.

## Intended pattern

The group is the masked load, its mask `COPY` to `$v0`, and its `PseudoVMV_V_I_M8` setup.  It is moved so that it ends up two instructions before the first use of the load result:

```text
def A        # group
def E
C = A * 2    # first use
```

## What it actually checks

- It finds the first non-debug use of the load result after the load.  If fewer than two instructions lie in between, it does nothing.  It also does nothing if the insertion point (the instruction before the first use) is already within 2 instructions of the load.
- The group is collected in original block order.
- It refuses to cross a call, an instruction with unmodeled side effects, or a store that may alias a load in the group.
- It refuses any register conflict with an instruction it would cross.  A def of a moved register conflicts with any access, and a use conflicts with a def.  This covers the physical `$v0` mask register.
- The moved instructions' source registers are added to `RegsToClearKillFlags`.

`DBG_VALUE` instructions for the moved definitions are not re-attached.

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

## What `ProcessReverseRematChain` actually checks

Recognized forward/reverse opcode pairs: `PseudoVFADD_VV_{M4,M8}_E32` / `PseudoVFSUB_VV_{M4,M8}_E32` (vector-vector), `PseudoVFADD_VFPR32_{M4,M8}_E32` / `PseudoVFSUB_VFPR32_{M4,M8}_E32` and `PseudoVFMUL_VFPR32_{M4,M8}_E32` / `PseudoVFDIV_VFPR32_{M4,M8}_E32` (vector-scalar).

The block is walked **backward** (from its end), so the link closest to the end of a chain (whose result nothing else in the chain reverse-rematerializes, e.g. `%D` above) is processed first; a map from an original register to its short-lived replacement is populated as deeper links are handled, so a shallower link (e.g. `%C`'s def) can look up whether `%D`'s processing already replaced `%C` with `%C2`, and anchor on that instead of the original (by-then-dead) `%C`.

For each matching instruction `%Y = OP(%X, %K)`:

1. For a vector-vector op, both operand assignments are tried (`Vs2`/`Vs1` and `Vs1`/`Vs2`) since either could be the one with a later second use — but once one succeeds, it adds a new reference to the other operand (as the reused `%K`), so at most one of the two assignments actually goes through per instruction, matching how the original single-hop implementation only ever reversed one of `%A`/`%B` from `%C = %A + %B`. A vector-scalar op only tries `Vs2` as `%X`, since `Vs1` is always a scalar FPR.
2. `%X` must have exactly two *real* uses: this def, and one other instruction (`%UseX`). A masked load that later redefines `%X` also reads it as a tied passthru (mask-undisturbed policy); that self-referential use is excluded from the count.
3. The anchor is `%Y`, or `%Y`'s replacement if a deeper link already reverse-rematerialized it. The new instruction is inserted right after the anchor's last real use in the block, which must come **strictly** before `%UseX` (not the same instruction — e.g. for `%I = %C + %F`, reversing `%C` anchored on `%F` is rejected when `%F`'s only use is that same `%I`, since the insertion would need to land both before and after it).
4. A new virtual register is created and `%X2 = REV(%Anchor, %K)` is inserted there, copying the VL/SEW/policy/`frm` operands and instruction flags (e.g. `nofpexcept`) from `%Y`'s defining instruction. `%UseX`'s use of `%X` is rewritten to `%X2`.

Like the other transforms, this only matches within one basic block and runs at the `ExpandPseudos` position, right after the machine scheduler and before register allocation.

## Result on the TSVC `reverse` kernel (single-hop, LMUL8)

`BLOCK_SIZE=128` splits into two interleaved 64-lane LMUL8 chunks (`reverse.mir`). The pass fires on both, reversing `%A` (the `Vs2` operand is tried first) from `%C = %A + %B`:

```text
ReverseRematChain: recompute %20 from %36 after %40 = PseudoVFADD_VV_M8_E32 undef %40, %36, %37, ...   # I0 = C0+F0
  for use in <K0 = A0+J0>
ReverseRematChain: recompute %24 from %35 after %39 = PseudoVFADD_VV_M8_E32 undef %39, %35, %38, ...   # I1 = C1+F1
  for use in <K1 = A1+J1>
```

**Measured**: `grep -c "Folded Spill" reverse.s` went from 4 (baseline) to 7 with `--custom-reverse` — worse, not better. `BLOCK_SIZE=128` interleaves two LMUL8 chains in the same block; shrinking one value's live range in one chain does not reduce the region-wide simultaneous demand across both chains, and the recomputed value is itself spilled almost immediately in the generated assembly. Register pressure here is a two-chain, whole-region problem, not a single live range that this local, per-instruction pattern match can fix by itself.

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
- Only the six `ADD_VV`/`SUB_VV`/`ADD_VFPR32`/`SUB_VFPR32`/`MUL_VFPR32`/`DIV_VFPR32` opcode pairs above (LMUL4/LMUL8, e32) are matched; there is no generalization to other opcodes, LMULs, or element widths, and no cost model — every match is applied unconditionally.
- Whether reversing a single-hop chain helps or hurts is workload-dependent (compare the two results above); there is no profitability check.
- Verified on `reverse.mir`/`reverse.s` and `reverse-v2.mir`/`reverse-v2.s`. `reverse-v2`'s numeric correctness was also checked end to end on the RISC-V board (`compile_deploy_and_run`'s `expected=` check passed) for the *unmodified* baseline; the flag itself was verified via `llc` directly, since `TRITON_RISCV_LLVM_ARGS` does not currently reach `ExpandPseudos` through the `g.sh`/Triton pipeline (a pre-existing issue, not something this transform introduced).

# Forward Rematerialization (`--custom-forward`)

`ExpandPseudos::ProcessForwardRematChain` implements the paper's **Figure 2(b)** ("multiple instruction rematerialization"), distinct from `--custom-reverse`'s Figure 2(c): "A is alive during all the computation. Thus we can rematerialize B, C, D from A." Instead of reverse-computing a value from something derived from it *later* (needs an algebraic inverse), it replays the *original forward chain of instructions* from a root that is already live for the whole region anyway (e.g. a loaded input also needed again at the very end), recomputing the value fresh right where it is next needed.

Concretely, given
```text
%R = <root, e.g. a load, still needed much later at some other use>
%B = PseudoVADD_VX_M8 undef %B, %R, %k1, ...   ; B = R + k1
%C = PseudoVADD_VX_M8 undef %C, %B, %k2, ...   ; C = B + k2
...
%I = <uses %C again, much later>
%J = <uses %B again, much later>
```
becomes, right before each late use, a fresh replay of the chain from `%R`:
```text
%B2 = PseudoVADD_VX_M8 undef %B2, %R, %k1, ...            # before %J
%J = <rewritten to use %B2>
%Ca = PseudoVADD_VX_M8 undef %Ca, %R, %k1, ...            # before %I
%C2 = PseudoVADD_VX_M8 undef %C2, %Ca, %k2, ...
%I = <rewritten to use %C2>
```

The key difference from `--custom-reverse`: this needs **no algebraic inverse**, so it is exact for *any* chain of ops — including non-invertible ones like shifts or bitwise ops that reverse rematerialization cannot touch — at the cost of redoing however many steps lie between the root and the value, redundantly across different late uses if their chains overlap (here, recomputing `%C` redundantly redoes the `%B` step). The paper explicitly accepts that cost: "we don't consider this tradeoff and consider computation is free."

## What `ProcessForwardRematChain` actually checks

A "chain link" is recognized by mnemonic prefix rather than a fixed opcode list (`ADD`, `SUB`, `RSUB`, `MUL`, `DIV`, `AND`, `OR`, `XOR`, `SLL`, `SRL`, `SRA`, and the float `F`-prefixed equivalents), so it covers LMUL4/M8, integer and float, and VV/VX/VI/VFPR32 forms uniformly — masked forms are excluded since their extra mask/passthru operands don't fit the shape.

For each chain-link instruction `%Y = OP(%X, %K)` (trying both `Vs2`/`Vs1` and `Vs1`/`Vs2` for a vector-vector op, only `Vs2` otherwise, same as `--custom-reverse`):

1. `%X` must have exactly two *real* uses: this def, and one other instruction (`%UseX`), the same check as `--custom-reverse`.
2. `%X`'s own ancestor chain of chain-link defs is walked backward (capped at 8 hops) to a root that is *not* itself a chain-link result (e.g. a load). An empty chain means `%X` already *is* such a root — there is nothing cheaper to replay it from (this is exactly why `%A`, the load, is never itself a candidate: only values *derived* from something get replayed).
3. **Profitability check**: the root must already be live at `%UseX` anyway (another use there or later, or in a different block) — otherwise replaying the chain would just extend the root's own live range instead of shrinking anything, the same check `ProcessRematLoads` makes for its inputs.
4. If all of that holds, the whole chain from root to `%X` is cloned with fresh virtual registers, inserted right before `%UseX`, and `%UseX`'s use of `%X` is rewritten to the final clone.

Unlike `--custom-reverse`, no backward-block-scan or replacement map is needed: every candidate independently replays from the *original*, untouched chain, so processing order doesn't matter.

## Results

`BLOCK_SIZE=128` splits into interleaved chunks at each LMUL (two at LMUL8, four at LMUL4). `grep -c "Folded Spill"`, no flags vs `--custom-forward` alone vs `--custom-reverse` alone vs both vs all five flags together:

| kernel | LMUL | none | `--custom-forward` | `--custom-reverse` | both | all 5 flags |
|---|---|---|---|---|---|---|
| `reverse` | 8 | 4 | 6 | 7 | 7 | 6 |
| `reverse_v2` | 4 | 14 | 11 | 7 | 7 | 7 |
| `reverse_v2` | 8 | 9 | 5 | 3 | 3 | **2** |

- **`reverse_v2` at LMUL8** is the strongest result: `VR Limit 32 Actual 82` before scheduling; `--custom-forward` alone drops the predicted peak pressure 66→42 (SLIL 1189→1088) and spills 9→5; stacking all five flags together reaches 9→**2**.
- **On `reverse` (LMUL8), forward chaining alone makes things *worse*** (4→6): `reverse`'s DAG is shallow — `A` and `B` are raw loads with no cheaper ancestor, so the only candidate left is `C` (from `C=A+B`), replayed right before its own second use. But `C`'s two uses (`F=C*scalar`, `I=C+F`) are adjacent instructions with essentially no gap between them, so the replay buys nothing and just adds dead weight. `reverse_v2`'s deeper `a→b→c→d` chain is exactly the shape Figure 2(b) targets; `reverse`'s shallow one-merge DAG is not.
- **On `reverse_v3`, it fires zero times** — a genuinely interesting negative result, not a bug. A valid 2-step chain does exist (`c` reachable from `a` via `SLL` then `ADD`), but the profitability check correctly rejects it: `a` has no late use at all in this kernel, because LLVM's own optimizer fully cancelled the `+a`/`-a` pair in `j=i-b; k=j+a` before `ExpandPseudos` ever runs. There is no "already alive anyway" root left to exploit, for either `--custom-forward` or `--custom-reverse` (whose own MUL/DIV reversal for `c` would additionally be numerically unsafe for integers, see below) — the compiler's own simplification already consumed the opportunity.

## Limitations

- Correctness itself is unconditional (replaying the original instructions is exact by construction, unlike `--custom-reverse`'s FP-rounding or the deliberately-excluded integer shift/MUL cases), but *profitability* is not: combining `--custom-forward` with other flags can help a lot (`reverse_v2`/LMUL8) or do nothing useful on its own (`reverse`/LMUL8, see above) — there is still no cost model weighing "how many steps redone" against "how much live range actually shrinks".
- The 8-hop cap on chain length is an arbitrary sanity bound, not derived from any cost model.
- Verified via `llc` directly on all four kernels above at their respective LMULs; not deployed to the RISC-V board under this flag (see the same `TRITON_RISCV_LLVM_ARGS` pipeline caveat noted for `--custom-reverse`).

# Results on `reverse_v3` (redesigned) and `reverse_v4`

`rvv_reverse_v3_elf.py` was redesigned after the "fires zero times" finding above: `h=g^d` became `h=g+d`, and `k=j+a` became `k=j-a`. That breaks the algebraic cancellation that used to erase `a`'s only late use (`j=i-b; k=j+a` used to collapse all the way to `k=i-5`, dropping `a` entirely) — with `k=j-a`, `a` is referenced twice late instead (once directly, once folded into a `2*a` computed as `a+a`), confirmed in the MIR. `rvv_reverse_v4_elf.py` is new: like the redesigned v3, but `d` is shifted two different ways for `e` (`srli`) and `f` (`srai`, via an explicit int32 bitcast round-trip) instead of `f` being an identity on `d`.

`grep -c "Folded Spill"`, none vs `--custom-forward` alone vs `--custom-reverse` alone vs both vs all five flags:

| kernel | LMUL | none | `--custom-forward` | `--custom-reverse` | both | all 5 flags |
|---|---|---|---|---|---|---|
| `reverse_v3` | 4 | 9 | **6** | 9 | 6 | 6 |
| `reverse_v3` | 8 | 5 | **3** | 5 | 3 | 3 |
| `reverse_v4` | 4 | 13 | **10** | 13 | 10 | 10 |
| `reverse_v4` | 8 | 8 | **6** | 8 | 6 | **8** |

- **`--custom-forward` now helps on both**, confirming the earlier "zero times" result was about the *kernel's* shape, not a bug in the pass: with `a` genuinely live late again, the same 2-step `SLL`-then-`ADD` replay from `a` that was rejected before now fires (once per interleaved chunk), recomputing `c` right before `i = h & c`. The replayed chain is `c = (a << shift) + scaled_const`, not `b=a+5; c=b<<shift` literally — LLVM's middle-end distributes the shift over the add before `ExpandPseudos` ever runs (the same fold noted for the original `reverse_v3`), so `b` never exists as a separate register to target at all; only `c` does, reachable in one fused 2-hop replay straight from `a`.
- **`--custom-reverse` still fires zero times on both**, unchanged from baseline in every row. Unaffected by the kernel redesign: the only invertible-looking step is `d = c*scalar` (integer `MUL`), which was deliberately never given a reverse pair (see "Limitations" above — needs a runtime-unknown modular inverse, not plain division), and `SLL`/`SRL`/`AND`/`OR` aren't reversible either.
- **One anomaly, not yet explained**: on `reverse_v4` at LMUL8, `--custom-forward` alone and "both" both reach 6, but stacking all five flags regresses back to 8 (baseline) — some interaction with `--custom-sink` or `--custom-a` undoes the forward chain's benefit on this specific kernel/LMUL combination. Not investigated further.
