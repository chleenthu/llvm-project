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

## Helper: `isWholeRegLoadName`

Whole register loads are matched by name so the pattern is not tied to one opcode: `VL1RE8_V`, `VL2RE16_V`, `VL4RE32_V`, `VL8RE64_V`, and so on.  Names such as `PseudoVLE32_V_M4` and `VLE32_V` do not match.

## Limitations

- The new load stays at the position of the masked load.  It is not sunk next to its first use.  The sink path in `ProcessInSameBlock` can move it later, but it matches by name and offset patterns.
- The `$v0 = COPY` that fed the removed masked load becomes dead when another `$v0` copy follows.  It is left for MachineDCE.
- Verified only on the MIR after `-run-pass=expandpseudos`.  Final assembly and tests have not been checked.

## Usage

```text
llc -mtriple=riscv64 -mattr=+v -custom-remat -run-pass=expandpseudos s279.mir -o -
```

`--custom-remat` can be combined with `--custom-sink`.  The remat transforms run first.

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
