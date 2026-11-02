# A three-tier taxonomy of register-pressure (RP) reduction

Each tier is defined by what it keeps the same. Moving down the list, more of the program changes, and each technique costs more and needs more analysis to apply safely.

---

## Tier I: Live-Range Reshaping
*(other possible names: Placement-Level Transformations, Schedule- and Allocation-Level Techniques)*

**Definition.** These transformations keep the program's computation exactly the same. The dataflow graph and the set of operations are unchanged. They only change *where* and *when* values are defined, used or kept in a register. RP drops because live ranges get shorter or are split up; no instruction's meaning changes. In the worst case an existing instruction is duplicated as-is.

| Technique | Description |
|---|---|
| **Sinking / RP-aware instruction scheduling** | Moves definitions closer to their uses, and orders instructions to minimise the maximum number of values live at once (MaxLive). Examples: Sethi–Ullman ordering and list scheduling with a pressure heuristic. |
| **Direct rematerialisation** | Instead of keeping a cheap value live (a constant, an address, a special register), recomputes it with the *same* instruction where it is needed. |
| **Coalescing / live-range splitting** | Merges copy-related live ranges, or splits a long range at points of low pressure, so the allocator can colour the graph with fewer registers. |
| *+ Anti-LICM (de-hoisting)* | Undoes loop-invariant code motion when a hoisted value would stay live across the whole loop. |
| *+ Sub-register liveness tracking* | Tracks liveness per lane or sub-register of a register tuple, so the parts that are already dead are freed early. |
| *+ Operand commutation / two-address reuse* | Swaps commutative operands so that the destination can reuse a source register that dies at that instruction. |

---

## Tier II: Value Re-derivation
*(other possible names: Computation-Level Rewriting, Algebraic Value Reconstruction)*

**Definition.** These transformations keep the program's *observable semantics* but change *how particular values are computed*. A live value is replaced by an equivalent expression built from values that are already live anyway, using algebraic identities, inverse operations or induction-variable relationships. Individual instructions are rewritten, but the structure of the loops and of the kernel stays the same.

| Technique | Description |
|---|---|
| **Reverse rematerialisation** | Recovers a dead operand from a live result using an inverse operation, e.g. `x = y − c` when `y = x + c` is still live. |
| **Affine re-expression** | Rewrites values as affine functions `a·iv + b` of a single induction variable or base pointer, so that many correlated values become one live register. |
| *+ Induction-variable merging (LSR-style)* | Combines several induction variables or address streams into one shared base with immediate offsets. |
| *+ Addressing-mode / immediate folding* | Moves base+offset arithmetic and constants into instruction encodings, so no register is needed to hold them. |
| *+ Reverse CSE* | Undoes common-subexpression elimination when holding the shared result live costs more than recomputing it. Unlike direct remat, this can involve a multi-instruction expression tree. |
| *+ Reassociation for pressure* | Rebalances expression trees from wide (high ILP, high RP) to deep (low RP), or the reverse. |
| *+ Precision narrowing / value packing* | Demotes values to narrower types (FP32 → FP16/BF16, i32 → i16) and packs several of them into one physical register. |
| *+ Uniformity promotion (scalarisation)* | Moves values that are uniform across a warp or vector from per-lane registers to scalar registers. Examples: VGPR → SGPR on AMD, or vector → scalar registers in RVV. |

---

## Tier III: Kernel Restructuring
*(other possible names: Loop- and Kernel-Level Transformations)*

**Definition.** These transformations change the *iteration space, the granularity or the decomposition* of the kernel. Loop bounds, tiling, the amount of work per thread or per instruction, and kernel boundaries can all change. The output is a new kernel that computes the same result with a different working-set size.

| Technique | Description |
|---|---|
| **Operator rewrite / fusion (and fission)** | Fuses kernels to remove intermediate values, or splits them (fission / loop distribution) so that each piece has a smaller set of simultaneously live values. |
| **Dynamic block size / LMUL selection** | Changes the tile size, threads per block or vector register grouping (RVV LMUL) to trade parallelism for registers per lane. |
| **Loop unrolling / re-rolling** | Lowers the unroll factor (or re-rolls an unrolled loop) to reduce the number of values live across iterations. |
| *+ Unroll-and-jam / register-tiling factor* | Picks register-tile dimensions, e.g. the micro-kernel tile in GEMM, under an explicit register budget. |
| *+ Pressure-constrained software pipelining* | Raises the initiation interval (II) or limits modulo variable expansion so that MaxLive fits the register file. |
| *+ Loop interchange / data-layout change* | Reorders loops or the memory layout to shorten reuse distances and reduce values carried between iterations. |
| *+ Warp specialisation / producer–consumer split* | Splits work across roles, e.g. loader warps and compute warps, so each role needs fewer registers. |
| *+ Occupancy / launch-bound tuning* | Sets a per-thread register cap (`__launch_bounds__`, `maxnreg`), which forces the compiler to apply the tiers above. |

---

## Suggested addition, Tier 0: Storage Relocation
*(another possible name: Spill-Space Diversification)*

This tier is not in the original list, but it fills a gap. It changes neither the computation nor the live ranges, only **where a live value is stored**. Classic spilling is the baseline that every other technique is trying to avoid, so it is worth naming explicitly.

- **Memory spilling**: to the stack, local memory or scratch.
- **On-chip spilling**: to shared memory, LDS or scratchpad.
- **Cross-register-file spilling**: SGPR → VGPR lanes, VGPR → AGPR, or GPR → vector/FP registers.

---

## The ordering axis in one line

> **Tier 0** keeps the computation and changes the *location* of values. **Tier I** keeps the operations and changes their *placement*. **Tier II** keeps the semantics and changes the *expressions*. **Tier III** keeps the result and changes the *kernel structure*.
