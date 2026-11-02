# Operand commutation / two-address reuse (generic MIR view)

**The problem.** Some instructions overwrite one of their sources: the destination is *tied* to a source, as in `x = x op y`. Examples:

- x86 `add rax, rbx`.
- RVV `vmacc.vv vd, vs1, vs2`, which computes `vd = vd + vs1·vs2`. The accumulator is both an input and the output.
- RVV masked or tail-undisturbed ops, where `vd` is also the passthru.

In SSA form, `%d = op %a, %b` uses a fresh `%d`. Before allocation, LLVM's `TwoAddressInstructionPass` has to make `%d` and the tied source the same register. If the tied source is still live after the instruction, it must be copied first:

```
%d = COPY %a          ; extra register: %a is still needed later
%d = op %d, %b        ; %b dies here
... use %a ...
```

During the instruction, `%a`, `%b` and `%d` are all live, so three registers are needed instead of two.

**The fix.** If the operation is commutative, and the *other* source `%b` dies at this instruction, swap the operands so that `%b` takes the tied slot:

```
%d = op %b, %a        ; %d reuses %b's register, no COPY
... use %a ...
```

The copy disappears, and `%b`'s register becomes `%d`'s register. That saves a register and an instruction.

**RVV FMA variants.** Sometimes the fix is more than swapping two inputs. For multiply-add there are two forms:
- `vmacc vd, vs1, vs2`: `vd = vs1·vs2 + vd`. The addend is overwritten.
- `vmadd vd, vs1, vs2`: `vd = vs1·vd + vs2`. One multiplicand is overwritten.

Both compute the same `a·b + c`, but they tie different operands. `RISCVInstrInfo::findCommutedOpIndices` / `commuteInstructionImpl` can switch between them. Whichever of `a`, `b` or `c` dies at the instruction becomes the tied operand, and none is copied. This works the same way for `vfmacc`/`vfmadd`, `vnmsac`/`vnmsub`, and so on.

**Its limit.** It only works when the operation is commutative, or has a commuted form like vmacc/vmadd, and when some source actually dies at the instruction. If every source is still live afterwards, a copy is needed whichever operand is tied.

---

# Operand commutation and two-address reuse

## The problem: tied operands

Some instructions are **two-address**: the destination register is also one of the sources. The instruction reads it and then overwrites it in place.

RVV's multiply-accumulate works this way:

```
vmacc.vv vd, vs1, vs2      # vd = vs1 * vs2 + vd
```

`vd` holds the accumulator going in and the result coming out. The old value in `vd` is gone afterwards.

So the register allocator needs the value in `vd` to **die** at this instruction, meaning nothing reads it later. If something does, the compiler has to save it first:

```
# want: r = acc + a*b, but acc is still used later; a dies here
vmv.v.v  v11, v8          # copy acc so we can destroy the copy
vmacc.vv v11, v9, v10     # v11 = v9*v10 + v11
```

That costs an extra instruction and an extra live register (`v11`). With LMUL=8 there are only 4 usable register groups, so one extra register can push the code into spilling.

## The fix: commute the operands

Often a different input dies at this instruction. In the example above, `a` (in `v9`) dies but `acc` does not. If the compiler can put `a` in the tied slot instead, no copy is needed.

RVV has a companion instruction that ties the other operand:

| Instruction | Semantics | Tied (overwritten) operand |
|---|---|---|
| `vmacc.vv vd, vs1, vs2` | `vd = vs1*vs2 + vd` | the **addend** |
| `vmadd.vv vd, vs1, vs2` | `vd = vs1*vd + vs2` | a **multiplicand** |

So the compiler rewrites the example as:

```
vmadd.vv v9, v10, v8      # v9 = v9*v10 + v8  → a*b + acc
```

The result is the same, `v8` (acc) is left intact, and the dead `v9` gets reused. Other pairs work the same way: `vfmacc`↔`vfmadd`, `vnmsac`↔`vnmsub`, `vfnmacc`↔`vfnmadd`, and so on. The two multiplicands can also be swapped freely, since `a*b = b*a`.

In general, the transformation is: **if the operand that dies is in the wrong slot, swap the operands (changing the opcode if needed) so it lands in the tied slot.**

## Where LLVM does this

- **`TwoAddressInstructionPass`**: for each tied def, if the tied source is still live afterwards, it asks whether the instruction is commutable (`TargetInstrInfo::findCommutedOpIndices`). If commuting puts a dying source in the tied slot, it commutes. Otherwise it inserts a COPY.
- **`RISCVInstrInfo::commuteInstructionImpl`**: handles the RVV FMA pseudos. Swapping the addend with a multiplicand also changes the opcode (e.g. `PseudoVFMACC_VV_*` becomes `PseudoVFMADD_VV_*`). Swapping the two multiplicands keeps the opcode.
- **Register coalescer**: it can also commute (`removeCopyByCommutingDef`) to get rid of a copy that already exists.

## RVV-specific catch: vector policy

In RVV, `vd` also supplies the **passthru** values:
- the **tail** elements (past `vl`) under a tail-undisturbed policy (`tu`)
- the **masked-off** elements under a mask-undisturbed policy (`mu`)

Commuting changes which value sits in `vd`, which changes what those tail or inactive lanes end up holding. LLVM therefore only commutes the addend into a multiplicand slot when that's safe, typically when the policy is tail-agnostic/mask-agnostic or the passthru is undef.

For Triton CPU kernels on RISC-V, this means a reduction loop that emits `vfmacc` with `tu` policy, or with a live passthru, can end up with extra `vmv.v.v` copies. If the loop's IR uses agnostic policies or an undef passthru, the backend is free to pick `vfmadd` and reuse the dying register.
