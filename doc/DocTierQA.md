# Q1: Does fusion still increase RP if both matmuls are in one program and you tile M → K2 → N?

**Yes, it does in general.** Your analogy to scheduling and sinking is right about the intermediate. The fused kernel also creates new long-lived state that the unfused version doesn't have.

Take `O = (A @ B) @ D`, with `S = A @ B` as the intermediate and K2 as the shared dimension:

```
forall m-tile (8 rows):
    accO[8, 64] = 0                              # must hold ALL N columns
    for k2 in 0..64 step 32:
        S[8, 32] = A[m, :] @ B[:, k2:k2+32]      # matmul1's acc: 8×32, computed once
        for n in 0..64 step 16:
            accO[:, n:n+16] += S @ D[k2:k2+32, n:n+16]
    store accO
```

- **Intermediate `S`:** its live range really is short. It is consumed as soon as it's produced, which is exactly your "pass the block to the consumer right away" idea. This part is pure benefit.
- **Consumer accumulator `accO`:** the second matmul is a reduction over K2. Its accumulator must survive every K2 iteration, so it stays live during the producer's work too. Peak RP while computing `S` (matmul1) becomes `S + accO + A/B operands`. Unfused, each kernel's peak is only `acc + operands`.
- **Size of `accO`:** with this loop order it is `8 × 64` (BM × N_full), not `8 × 16` (BM × BN). In f16 that is 1 KB, which is 32 VRs at VLEN=256: the whole register file, before `S` (8×32, 16 VRs) or any operands. With an f32 accumulator it is 64 VRs. It will spill, and it gets worse as N grows.

The rule behind the saying "fusion increases RP" is this:

- If the **consumer is elementwise** (bias, activation, quantization epilogue), fusion barely raises RP. The intermediate dies immediately, so this is the scheduling or sink case.
- If the **consumer is a reduction over the producer's tiling dimension** (matmul→matmul, attention), fusion adds carried state. That state stays live across the producer's loop. This is where the RP increase comes from.

Being in the same workgroup or program doesn't change this. What matters is what is simultaneously live at the innermost loop body.

Your levers:
1. **Loop order M → N → K2:** `accO` becomes `BM × BN`, but you recompute `S` once per N tile. That is rematerialization at kernel level, trading FLOPs for RP.
2. **Small BM.**
3. **Accept spilling `accO` to a stack or L1 buffer.** On CPU this is often fine, because L1 is close and the spill traffic is still far below a DRAM round-trip of `S`.

# Q2: Reverse remat on the op DDG

The idea is sound and has a name at graph level: tensor rematerialization (Chen's √n checkpointing, Checkmate, DTR). If you add equivalence rewrites such as associativity, slicing or concatenating tiles, or deriving one tensor from another live one, you get equality-saturation-style search (TASO, Tensat) with peak memory as the cost. Changing an edge's source changes the set of legal topological orders, so the minimum-peak schedule changes too.

Some cautions:

- **Pressure moves, it doesn't disappear.** Redirecting a use from source X to source Y shortens X's live range but may extend Y's. The rewrite only wins when Y is already live at that point anyway (a "free rider"), or when Y is much smaller than X. Use that as the legality and profit test.
- **Keep the DAG acyclic.** A new source must not depend, transitively, on the consumer.
- **Search cost.** Minimum-peak topological ordering is already NP-hard (it is the register sufficiency problem). Adding rewrite choices multiplies the search space. You need either a restricted rewrite set (slice-of-live-tensor, transpose view, recompute cheap elementwise ops) or ILP/heuristic search like Checkmate's.
- **Recompute cost.** Each rewrite should carry a FLOP cost so the cost model can trade peak memory against compute.
- **Where to do it in Triton-CPU:** at the TTIR/TTCIR tensor level, before vector lowering. After lowering, the op structure is gone.

# Q3: Is loop unrolling not beneficial to RP?

Unrolling doesn't inherently increase RP. What increases it is the interleaving that unrolling enables.

- **Rolled loop:** temporaries from different iterations never overlap, so the allocator reuses the same VRs, as you said. Only loop-carried values (accumulators, pointers) are live across iterations.
- **Unrolled but kept sequential:** RP is about the same, because the allocator still reuses registers. Unrolling can even lower scalar RP slightly: induction-variable updates and address arithmetic fold into constant offsets.
- **Unrolled and interleaved by the scheduler** (for ILP and latency hiding, or LoopVectorize's interleave count): live ranges from U iterations now overlap, so per-iteration temporaries take about U× the registers. This is the usual source of the "unrolling raises RP" effect. Software pipelining (Triton's `num_stages`) works the same way on purpose.

For Triton-CPU the bigger factor is spatial unrolling: the tile itself. A `tl.arange(0, BLOCK)` tensor gets laid out across `BLOCK / (VLEN/SEW) / LMUL` register groups and is fully live. Shrinking BLOCK and adding an explicit loop ("rolling") is usually the most effective RP lever. That's the reason your Dynamic Block Size / LMUL category exists.

So unrolling by itself is roughly RP-neutral, and the benefit of a rolled loop is that it keeps iterations from overlapping. Unroll when you have spare VRs to spend on ILP. On RVV with 32 VRs and LMUL > 1, you usually don't.

# Spatial unrolling in the tl.arange(0, BIG_BLOCK) case

The situation you describe is what I meant by spatial unrolling. It's an unrolled loop that the type legalizer creates, not a loop-unroll pass.

What happens

tl.arange(0, BLOCK) becomes one wide vector, e.g. <BLOCK x float>. The hardware has no register that wide, so ISel type legalization splits it into legal pieces:

chunks = BLOCK / (VLEN/SEW × LMUL)

Each chunk becomes its own SelectionDAG node and later its own virtual register. For vector-add you get:

```
x0..xC = load chunks of x
y0..yC = load chunks of y
z0..zC = x_i + y_i
store z0..zC
```

This is exactly a strip-mined loop over chunks, fully unrolled. "Spatial" means it's spread across registers (space) instead of across iterations (time). There's no loop that a later pass could roll back up. Any reuse has to come from scheduling.
