# Group Sink Optimization

`ProcessInSameBlock` has a targeted group-sinking path for selected masked
32-bit vector loads.  It recognizes selected loads and moves the load together with the setup needed to execute it immediately before its vector arithmetic consumer.

The group is only moved within the same machine basic block.

## Intended pattern

A masked `PseudoVLE32_V_M8_MASK` load is selected when its memory operand has an offset of at least half a block:

```text
memory offset >= (BlockSize / 2) * 4
```

The full expected idiom is:

```text
%47 = PseudoVMV_V_I_M8 undef %47, 0                 # VMV zero setup
$v0 = COPY %21                                      # COPY mask to $v0
%47 = PseudoVLE32_V_M8_MASK %47, base+{256|384}, $v0, %vl
%60  = PseudoVFADD_VV_M8_E32 ..., %31, %47, ..., %vl
```

`PseudoVFMAX_VV_M8_E32` is also accepted as the consumer.

## What `ProcessInSameBlock` actually checks

For a masked load, the pass collects these instructions in this order:

1. The `PseudoVLE32_V_M8_MASK` load.
2. The nearest preceding definition of the first physical register used by the
   load.  For this MIR that register is `$v0`, and the definition must be a `COPY`.
3. A different definition of the load's tied destination whose opcode name
   contains `PseudoVMV_V_I_M8`; this is the zero/undisturbed-vector setup.

# Single-Instruction Sink Optimization

`ProcessInSameBlock` contains a narrow single-instruction sink optimization.  It moves an eligible zero-initializing vector move immediately before the masked vector load that reuses its tied destination.

## Intended pattern

It starts at an `PseudoVMV_V_I` instruction.  Before sinking, unrelated instructions may separate the initialization from its load:

```text
%55 = PseudoVMV_V_I_M8 undef %55, 0
... unrelated instructions ...
%55 = PseudoVLE32_V_M8_MASK %55, %54, $v0, %vl, 5, 1
```

The pass sinks the `PseudoVMV_V_I` directly before the load:

```text
... unrelated instructions ...
%55 = PseudoVMV_V_I_M8 undef %55, 0
%55 = PseudoVLE32_V_M8_MASK %55, %54, $v0, %vl, 5, 1
```

## What `ProcessInSameBlock` actually checks

- It starts at the `PseudoVMV_V_I`.
- It looks specifically for a following `PseudoVLE32_V_M8_MASK` with the same tied destination.

# Sink-to-First-Use Optimization

`ProcessInSameBlock` sinks a mask-producing compare, `PseudoVMSLE_VI_M*`, with `FindFirstUseToSinkTo`.  It moves the compare immediately before the first instruction that uses its result.

## Intended pattern

Before sinking, unrelated instructions may separate the compare from its first use:

```text
%33 = PseudoVMSLE_VI_M4 %30, 0, %vl, 5
... unrelated instructions ...
$v0 = COPY %33
```

The pass sinks the compare directly before that use:

```text
... unrelated instructions ...
%33 = PseudoVMSLE_VI_M4 %30, 0, %vl, 5
$v0 = COPY %33
```

## What `FindFirstUseToSinkTo` actually checks

- The instruction has exactly one explicit def, and it is a virtual register.
- `isSafeToMove` holds and the instruction is not convergent.
- All uses of the result are in the same block.  The first one after the instruction is the target.
- It does nothing if there is no such use, or if the instruction is already just before it.
- No call, side-effecting instruction, or store that could clobber a load lies between the instruction and the target.
- No register conflict with the instructions crossed: a def of the moved instruction's registers conflicts with any access, and a use conflicts with a def.
- The moved instruction's source registers are added to `RegsToClearKillFlags`.

Unlike the group sink, which stops two instructions before the use, this path places the instruction just before it.

# Expand-Pseudos COPY-to-VMV Optimization

This phase runs before the sink diagnostics.  `LowerCopy` removes a redundant virtual vector `COPY` by recreating its VMV producer directly in the COPY destination.

## Intended pattern

The pass looks for a VMV that defines the COPY source:

```text
src:VRM8 = PseudoVMV_V_I_M8 undef src, 0, %vl, ...
dst:VRM8 = COPY src
```

The COPY is erased and a new VMV is inserted at the former COPY location:

```text
dst = PseudoVMV_V_I_M8 undef dst, 0
```

A vector-from-scalar `PseudoVMV_V_X_M8` move is also accepted.

## Observed instances and connection to group sinking

The example records seven immediate-form rewrites of
`%dst:vrm8nov0 = COPY %55:vrm8nov0`: destinations `%27`, `%31`, `%35`, `%39`,
`%43`, `%47`, and `%51`.  These replacements create VMV definitions at the
per-load destinations.  The later group-sink path can then find the
`PseudoVMV_V_I_M8` setup for `%31`, `%35`, `%47`, and `%51`, bundle it with the
`$v0` mask COPY and masked VLE32, and sink that group to its VFADD consumer.

## Observed instances and connection to single-instruction sinking

The `LowerCopy` phase also supplies one of the VMVs later handled by the single-instruction sink path.  It first lowers:

```text
%27 = COPY %55  ->  %27 = PseudoVMV_V_I_M8 undef %27, 0, %9, 5, 0
```

`ProcessInSameBlock` subsequently recognizes `%27` as an eligible VMV and finds its later tied-destination masked load.  It then moves the materialized VMV immediately before this load. The flow is therefore:

```text
original %55 VMV -> COPY to a destination -> LowerCopy creates VMV -> single sink places before VLE32
```

This shows that `LowerCopy` creates independent zero/undisturbed setups for copied vector destinations.

## Observed instances and connection to sink-to-first-use

In `s279` the mask compare `%33` is produced early and only consumed by the mask `COPY` to `$v0` about ten instructions later.  Before sinking:

```text
early-clobber %33:vr = PseudoVMSLE_VI_M4 %30, 0, -1, 5
... about ten unrelated instructions ...
$v0 = COPY %33
```

`FindFirstUseToSinkTo` moves the compare next to the `COPY`:

```text
... about ten unrelated instructions ...
early-clobber %33:vr = PseudoVMSLE_VI_M4 %30, 0, -1, 5
$v0 = COPY %33
```

The debug log shows `Sink early-clobber %33 ... before $v0 = COPY %33` followed by `Sink success.`  The loop later visits the moved compare again, finds it already just before its use, and reports `No first use, or already just before it.` without moving it.

This shortens the live range of the `%33` mask register, which is now defined only where it is needed.  The `$v0 = COPY` is the mask copy that the group sink also collects, so the compare, its copy and the masked load end up close together.
