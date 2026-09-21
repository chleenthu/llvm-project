# Register Pressure and Live Interval Length Update

`ExpandPseudos` keeps two pieces of vector register usage and updates them incrementally when `FindFirstUseToSinkTo` sinks an instruction, instead of recomputing them for the block.  Only vector registers are tracked.

- **RP** (`VRPressure`): the `VR` register pressure right after each instruction.
- **LIL** (`VRegLIL`, per block in `BlockUsage::LIL`): the live interval length of each vector vreg.  The sum over the vregs is the SLIL.

## Definitions

### RP

RP is the register pressure after an instruction.  The last use of a register does not count: a register is live after its first reference (DEF) and dies at its last reference (LAST USE).  Weights are the LMUL of the register class: 1 for a mask, 2, 4 or 8 for a register group.  So for one instruction

```text
RP(I) = RP(previous instruction) + sum of LMUL of the DEFs of I - sum of LMUL of the LAST USEs of I
```

A register that is live out of the block has no LAST USE.

### LIL

```text
LIL(R) = (row(Last) - row(First)) * LMUL(R)
```

Rows count only instructions that touch a vector register, so scalar instructions are skipped.  Because the last use does not count, `def A; def B; C = A * 2` gives A a length of `2 * LMUL`.  The value is the same as `computeSLIL` in `RISCVRegisterPressure.cpp`, and its sum is checked against it.

### Positions and events

Each block keeps, in `BlockUsage`:

- `Pos`: the index of each instruction among the non-debug instructions.  Scalar instructions are counted here.
- `VecPrefix[i]`: the number of instructions touching a vector register before position `i`.  A window of instructions has `Rows = VecPrefix[end + 1] - VecPrefix[begin]` rows.
- `Range`: the first and last position of each vector vreg.
- `Events`: the DEFs and LAST USEs of the vector registers, sorted by position.  An event holds the position, the register, its LMUL, whether it is a DEF, and whether it counts towards the pressure (a live-out LAST USE does not).
- `NextEvent[i]`: the index in `Events` of the first event at position `i` or later.  It lets a scan start at any position without a search.

Undef uses (the passthru tied to a def) are not references.  Physical vector registers such as `$v0` are events too, with LMUL 1, from a def to its last use.

## The move

Sinking `MI` moves it down past the `N` instructions between it and its first use, so that it ends up just before that use:

```text
MI           P            MI's position
I1 ... IN    P+1 .. P+N   the instructions it passes (the window)
U            P+N+1        first use of MI's result
```

`N` counts all the non-debug instructions, scalars included, because each one has an RP.  `Rows` is the number of them that touch a vector register.  Sinking is only done when the register's uses are all in this block and nothing crossed conflicts with `MI`.

## The five updates

The names below are the ones used in the comments of `FindFirstUseToSinkTo`.  "Sunk" is the instruction `MI` that is moved, and "passby" is one of the `N` instructions it passes (or a register of one of them).  Before the move `MI`'s DEF is live after each `Ik`.  After it, `MI` has not executed yet.

| # | Update | Rule |
|---|---|---|
| 1 | passby RP | `RP(Ik) += DeltaMI`, with `DeltaMI` = LAST USE LMUL of `MI` − DEF LMUL of `MI` |
| 2 | sunk RP | `RP(MI) += sum over the window's events` (`WindowNet`) |
| 3 | sunk LIL | `MI` defines R: `-= Rows * LMUL`. `MI` is R's last reference: `+= Rows * LMUL` |
| 4 | passby LIL | a DEF in the window whose register is used after it: `+= LMUL`. A LAST USE in the window with an earlier first reference: `-= LMUL` |
| 5 | sunk is not last use but becomes last use after sink | passby LIL `+= LMUL * rows after the old last use`, and passby RP `+= LMUL` from the old last use onward |

### 1. passby RP

Every passed instruction gets `DeltaMI`:

```text
RP(Ik) += DeltaMI      DeltaMI = sum of LAST USE LMUL of MI - sum of DEF LMUL of MI
```

`MI`'s DEFs are not live yet (`-LMUL` each), and the registers of `MI`'s LAST USEs are still live (`+LMUL` each), whether the passed instruction is a DEF, a LAST USE or a scalar.  For a mask `MI` with one DEF and no LAST USE this is `-LMUL(MI's DEF)`.  It only uses `MI`'s own events, those at position `P`.

### 2. sunk RP

`MI`'s own point gains what the passed instructions changed:

```text
RP(MI) += sum over the events in the window of  (+LMUL for a DEF, -LMUL for a LAST USE)
```

This is the same as `RP(MI) = old RP(IN)`.  A live-out LAST USE is not counted, because the register never dies in the block.

### 3. sunk LIL

`MI` is one row.  Moving it past the window moves its row `Rows` rows later, so for the registers `MI` itself references:

| Register | Change |
|---|---|
| `MI` defines it (its first reference is `MI`, `Range.first == P`) | `LIL -= Rows * LMUL` |
| `MI` reads it and its last reference is `MI` (`Range.second == P`) | `LIL += Rows * LMUL` |
| `MI` reads it and it is referenced before and after `MI` | unchanged |

`==` is used because the range of a register only changes if `MI` is its first or its last reference.  When both hold (a dead def, or a tied def and use) the two changes cancel.  A register that appears twice in `MI`'s operands is counted once.

### 4. passby LIL

The other registers change only if `MI`'s row was between their first and last reference before the move, or is after it:

| Event in the window | Condition | Change |
|---|---|---|
| DEF of R | R's last use is after the window (`Range.second > P + N`) | `LIL(R) += LMUL`, `MI` is now between its first and last reference |
| LAST USE of R | R's first reference is before `MI` (`Range.first < P`) | `LIL(R) -= LMUL`, `MI` is no longer between them |

A register that has both its DEF and its LAST USE in the window does not change, and neither does one that spans the whole window.

### 5. sunk is not last use but becomes last use after sink

If `MI` reads a register R whose last use is a passed instruction, `MI` is not R's last use before the move, but it is after it, because `MI` ends up later.  R stays live from its old last use up to `MI`.  Two updates go with it, and they replace the LAST USE rule of update 4 for that R:

- **passby LIL:** `LIL(R) += LMUL * (rows after the old last use, up to the end of the window)`.
- **passby RP:** every passed instruction from the old last use onward (its position `Q >= E.Pos`) gets `RP += LMUL(R)`.

Without this, `RP` and `LIL` would treat R as dead at its old last use.  It is the case in the `%30` example below.

The SLIL of the block changes by the sum of the LIL changes.

## The event walk

Updates 2, 4 and 5 are done in one walk over the DEFs and LAST USEs in the window (updates 1 and 3 use `MI`'s own events and operands):

```text
i = NextEvent[P + 1]
while Events[i].Pos <= P + N:
    E = Events[i]
    if E.Pressure:   WindowNet += E.IsDef ? +LMUL : -LMUL     # update 2, sunk RP
    if E is a DEF and Last(E.Reg) > P + N:      LIL(E.Reg) += LMUL      # update 4, passby LIL
    if E is a LAST USE:
        if MI reads E.Reg:  LIL(E.Reg) += LMUL * rows after E; remember (E.Pos, LMUL) for the RP    # update 5
            (passby RP: RP(Ik) += LMUL for the passed instructions with position >= E.Pos)
        elif First(E.Reg) < P:  LIL(E.Reg) -= LMUL     # update 4, passby LIL
    i++
```

The walk starts at the first event after `MI` and visits no other instruction, so instructions that have no DEF or LAST USE are skipped.  The per-instruction RP entries of the `N` passed instructions are still written one by one, because each has its own entry.

## Example

`s279` with `-custom-sink`, sinking `%33 = PseudoVMSLE_VI_M4 %30` to just before `$v0 = COPY %33`.  `%33` has LMUL 1.

```text
      early-clobber %33 = PseudoVMSLE_VI_M4 %30, 0        P
+1 R  %36 = VL4RE32_V %35                                    DEF %36
+2    %46 = ADD %45, %27                                     scalar
+3 R  %68 = PseudoVSUB_VV_M4 undef %68, %40, %36             DEF %68, LAST USE %40, LAST USE %36
+4 R  %47 = VL4RE32_V %46                                    DEF %47
+5    %50 = ADD %49, %27                                     scalar
+6 R  %44 = PseudoVMSLT_VV_M4 %30, %68                       DEF %44
+7 R  %51 = VL4RE32_V %50                                    DEF %51
+8 R  %52 = COPY %60                                         DEF %52
+9 R  %52 = PseudoVMADD_VV_M4 %52, %51, %47
      $v0 = COPY %33                                          first use
```

`N = 9`, `Rows = 7`.

RP:

- Each of the 9 passed instructions: `DeltaMI = -1` (one mask DEF, no LAST USE of `MI`).
- `RP(MI)` = 13 + 4 (`%36`) - 4 (`%68` DEF 4, `%40` and `%36` LAST USE -8) + 4 (`%47`) + 1 (`%44`) + 4 (`%51`) + 4 (`%52`) = 26.

LIL:

| Register | Before | After | Reason |
|---|---|---|---|
| `%33` | 10 | 3 | its DEF moves 7 rows: `10 - 7 * 1` |
| `%40` | 12 | 8 | LAST USE in the window, first reference before `MI` |
| `%52` | 64 | 68 | DEF in the window, used after it |
| `%47` | 76 | 80 | DEF in the window, used after it |
| `%68` | 104 | 108 | DEF in the window, used after it |
| `%51` | 32 | 36 | DEF in the window, used after it |
| `%44` | 17 | 18 | DEF in the window, used after it |

`%36` has its DEF and LAST USE in the window and does not change.  The SLIL goes from 583 to 589.

### A register that `MI` reads

After the remat passes `%30` is not used after the `VMSLT`, which is its last use, at `+6`.  `MI` also reads `%30`, and after the move `MI` is at the end of the window, so `%30` stays live for the 3 rows after the `VMSLT` (`%51`, `%52`, `%52`):

```text
+6 R  %44 = PseudoVMSLT_VV_M4 %30, %68        was the last use of %30
+7 R  %51 = VL4RE32_V %50
+8 R  %52 = COPY %60
+9 R  %52 = PseudoVMADD_VV_M4 ...
      early-clobber %33 = PseudoVMSLE_VI_M4 %30, 0        new last use of %30
```

The LIL of `%30` goes from 24 to `24 + 4 * 3 = 36`, and the instructions at `+7`, `+8` and `+9` each get `+4` in RP.

## Verification

Under `LLVM_DEBUG` (`-debug-only=expandpseudos`) each sink checks the prediction:

- RP is compared for every instruction of the block with a fresh `RegPressureTracker` run (`RP verified against RPTracker: OK`).  This needs valid `LiveIntervals`.  The sink calls `LIS->handleMove`, but the remat transforms and the group sink create or move instructions without updating them, so after those the check prints `RP verification skipped: LiveIntervals are stale`.
- LIL is compared for each changed register with the recomputed value, and the block SLIL with `computeSLIL` (a copy of the function in `RISCVRegisterPressure.cpp`).

## Limitations

- The update is only written for `FindFirstUseToSinkTo`.  After the group sink or the remat transforms the per-block data is recomputed with `computeBlockUsage`, not updated.
- A live-out register is recognised by a reference in another block, or by a use before its first def in the block.  A live-in vector register is counted from its first use in the block.
- Only the first `RegPressureTracker` check is independent of the prediction when the transforms before it did not change the block.
