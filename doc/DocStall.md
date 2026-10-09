# RVV dependency-stall measurements on BPI-F3 (SpacemiT X60)

Board: `ssh chlee@140.114.78.64`, Spacemit X60, VLEN=256, 1.6 GHz. Cycles come from
the hardware cycle counter (`perf_event_open`, user-only), with the program pinned to core 4.
Measured 2026-10-11.

## Question
If a vector instruction directly uses the result of the instruction before it
(e.g. `vfadd` right after the `vle32` that loads its input), how many independent
instructions must sit between them to avoid a stall? Measured separately for:

| Pattern | Producer -> consumer |
|---|---|
| LC | load -> compute (`vle32.v` -> `vfadd.vv`) |
| CS | compute -> store (`vfadd.vv` -> `vse32.v`) |
| LS | load -> store (`vle32.v` -> `vse32.v`) |
| CC | compute -> compute (`vfadd.vv` -> `vfadd.vv`) |

The compute op is `vfadd.vv` at e32 (f32) unless a section says otherwise. The i32 section
repeats the tests with `vadd.vv` and `vmul.vv` as the compute op, and the f16 section with
`vfadd.vv` at e16 (`vle16`/`vse16`).
The `vfmul.vv` section repeats them with floating-point multiply at e32 and e16.

## Method
- `gen.py` creates, for each pattern, filler type and N = 0..MAXN, a loop of 16 unrolled
  copies of: producer, then N filler instructions, then consumer.
- Each test has two versions: **dep** (the consumer reads the producer's register) and
  **nodep** (the consumer reads an unrelated register, with everything else identical).
  **stall = dep - nodep** cycles per copy.
- Filler types: scalar `add` (`alu`), `nop`, and independent vector `vadd.vv` or `vfadd.vv`.
- **Throughput mode** (`CHAIN=0`): the copies are independent, so the core can overlap them.
- **Chain mode** (`CHAIN=1`, LC/CC only): each copy's consumer result passes through
  `vmv.x.s` into the next copy's load address or compute input. This serializes the copies,
  so latency that throughput mode hides becomes visible.

## Results: f32 (`vfadd.vv`, e32)

| Pair | m1: stall at distance 0 | m1: vector / scalar instrs needed | m2: stall at distance 0 | m2: vector / scalar instrs needed | m4, m8 |
|---|---|---|---|---|---|
| LC `vle32`->`vfadd` | 1.0 (chain only) | 1 / 4 | 1.0 (chain only) | 1 / 4 | 0 |
| CC `vfadd`->`vfadd` | 2.1 (chain only) | 2 / 10 | 2.0 (chain only) | 1 / 7 | 0 |
| CS `vfadd`->`vse32` | 1.4 | 2-3 / 4 (`nop`), 5 (`alu`) | 0 | 0 / 0 | 0 |
| LS `vle32`->`vse32` | 3.0 | 4 / 7 (`nop`), 8 (`alu`) | 0 (~0.5 with 1-2 vector fillers) | 0, or >=3 / 0 | 0 (m4 noisy) |

"(chain only)" means the stall shows in chain mode and is 0 in throughput mode (LC 0.01 and
CC 0.00 at m1). CS and LS can only be measured in throughput mode.

Notes, all LMULs:
- Scalar fillers cover the gap much less effectively than vector fillers, which suggests the
  vector unit runs separately from the scalar pipeline. A scalar instruction only helps once
  it delays when the next vector instruction is issued.

Notes, m2:
- The m2 stalls only appear in chain mode. In throughput mode, independent copies overlap and
  hide them. That's the reason the first throughput-only runs showed no stall at m2.
- Scalar fillers don't remove the stall steadily as N grows; it switches on and off with N,
  probably because of dual-issue pairing:
  - LC with `alu` stalls at N = 0, 1, 3 and is clean from N = 4 (N = 2 is also clean).
  - CC with `alu`/`nop` stalls at N = 0, 1, 2, 4, 6 and is clean from N = 7.

  The table gives the smallest N from which every larger N is clean.
- CS and LS can't be chained (a store has no register result), so throughput mode is the only
  measurement for them. A stall that throughput mode hides, like LC/CC above, would not show.
- LS has a small cost of about 0.5 cycles with 1-2 vector fillers. It's gone at distance 0
  and at 3 or more fillers, so it looks like a pairing effect rather than a latency stall.

Notes, m4:
- Measured values at distance 0: LC 0.00 (throughput) and -0.23 to 0.16 (chain), CC 0.00 and
  -0.01, CS -0.01 to 0.00, LS -0.18 to 1.09.
- No stall in chain mode either, so m4 hides the latency that m1 and m2 show.
- LS readings at m4 are noisy, about +/-1 cycle. At N = 0 the four filler types run identical
  code, yet they read -0.18, 0.13, 0.01 and 1.09. The stall readings at larger N look like
  noise too: they swing both ways and don't fall as N grows.

Notes, m8:
- Measured values at distance 0: LC 0.03 (throughput) and 0.04 (chain), CC 0.01 and 0.01,
  CS -0.02 to 0.09, LS -0.01 to 0.01.
- Measured at distances 0-12 with every filler type, in both throughput mode and chain mode
  (chain mode for LC/CC only). The largest |stall| at any distance is 0.39 cycles, apart from
  a single 1.8-cycle outlier (LS, `nop` filler, N=6) that doesn't recur at N=5 or N=7.
- Each m8 instruction keeps its unit busy for about 8 cycles or more, which covers the
  latency fully.

### Cost per pattern (all LMULs)
- From m4 up there's no dependency stall: each vector instruction keeps its unit busy
  for several cycles (roughly 4-8 at m4, about 8 at m8), and that covers the latency. m2 still
  has the LC/CC stalls above, but only when copies are chained.
- A structural cost appears instead: a `vle32`+`vse32` pair costs far more than either op
  alongside a compute op. This happens whether or not the store depends on the load.

Cycles per copy at distance 0 (throughput mode, `alu` filler; dep and nodep are equal
except where the nodep value is given):

| LMUL | LC `vle`+`vfadd` | CS `vfadd`+`vse` | LS `vle`+`vse` | CC `vfadd`+`vfadd` |
|---|---|---|---|---|
| m1 | 5.0 | 4.0 (nodep 2.6) | 7.0 (nodep 4.0) | 4.0 |
| m2 | 4.0 | 4.8 | 8.0 | 4.0 |
| m4 | 8.0 | 8.0 | **18.1** | 8.0 |
| m8 | 16.1 | 18.7 | **48.1** | 16.0 |

At m8, each added `vadd`/`vfadd` filler costs about 8 cycles. For m8 the register
allocation is compact, because only 4 register groups exist: the unrelated and source
operands share `v0`, and the producer always writes `v8`.

Raw tables: `stall/f32/vfadd_m{1,2,4,8}.txt` (throughput) and
`stall/f32/vfadd_chain_m{1,2,4,8}.txt` (chain).

## Results: i32 (`vadd.vv` and `vmul.vv`, e32)
Same tests with the compute op set to `vadd.vv` or `vmul.vv` (`./run.sh L 12 "" CHAIN vadd|vmul`).
LS doesn't involve the compute op, so its row is the same for both ops and for f32.

| Pair | m1: stall at distance 0 | m1: vector / scalar instrs needed | m2: stall at distance 0 | m2: vector / scalar instrs needed | m4, m8 |
|---|---|---|---|---|---|
| LC `vle32`->`vadd` | 1.0 (chain only) | 1 / 4 | 1.0 (chain only) | 1 / 4 | 0 |
| LC `vle32`->`vmul` | 1.0 (chain only) | 1 / 4 | 2.0 (chain only) | 1 / 4 | 0 |
| CC `vadd`->`vadd` | 2.1 (chain only) | 2 / 10 | 2.1 (chain only) | 1 / 7 | 0 |
| CC `vmul`->`vmul` | 3.2 (chain only) | 3 / 12 (`alu`), 10 (`nop`) | 3.2 (chain only) | 1 / 12 (`alu`), 10 (`nop`) | 0 |
| CS `vadd`->`vse32` | 1.4 | 2 / 4 | 0 | 0 / 0 | 0 |
| CS `vmul`->`vse32` | 2.4 | 3 / 6 (`alu`), 5 (`nop`) | ~0.35 | 2 / 6 | 0 |
| LS `vle32`->`vse32` | 3.0 | 4 / 7 (`alu`), 6 (`nop`) | 0 | 0 / 0 | 0 (m4 noisy) |

Notes:
- `vadd` gives the same numbers as `vfadd` at every LMUL, so the f32 tables also apply to
  integer add.
- `vmul` has a longer latency. At m1 and m2, CC stalls about 3 cycles instead of 2, and CS
  about 2.4 instead of 1.4 at m1. At m2 the LC stall is 2 cycles instead of 1, even though
  the load is the producer; the multiply probably reads its operands earlier than `vadd`.
- The vmul m2 CS stall is small (about 0.35 cycles at N = 0-4) but steady, and disappears
  from N = 6 with scalar fillers or N = 2 with vector fillers.
- As with f32, scalar fillers don't remove the stall steadily as N grows. The vmul m1 CC row with `alu` stays near
  3 cycles up to N = 7 and only reaches 0 at N = 12. The table gives the smallest N from
  which every larger N is clean.
- m4 LS readings swing by +/-1-3 cycles in both directions at every N, with no trend. That's
  noise, consistent with the f32 m4 and m8 results.
- vmul m8 chain CC has scattered +/-1-cycle readings at N >= 4 but nothing at N = 0-3, so it's
  noise rather than a stall.

Raw tables: `stall/i32/{vadd,vmul}_m{1,2,4,8}.txt` (throughput) and
`stall/i32/{vadd,vmul}_chain_m{1,2,4,8}.txt` (chain). `stall/summ.py` turns a pair of them
into a first draft of these tables; noisy rows were checked by hand.

## Results: f16 (`vfadd.vv`, e16)
Same tests at SEW=16: `vle16.v` / `vfadd.vv` / `vse16.v` (`SEW=16 ./run.sh L 12 "" CHAIN vfadd`).
The X60 supports Zvfh, so f16 arithmetic runs natively.

| Pair | m1: stall at distance 0 | m1: vector / scalar instrs needed | m2: stall at distance 0 | m2: vector / scalar instrs needed | m4, m8 |
|---|---|---|---|---|---|
| LC `vle16`->`vfadd` | 1.0 (chain only) | 1 / 4 | 1.0 (chain only) | 1 / 4 | 0 |
| CC `vfadd`->`vfadd` | 2.0 (chain only) | 2 / 10 | 2.1 (chain only) | 1 / 7 | 0 |
| CS `vfadd`->`vse16` | 1.4 | 2 / 4 (`nop`), 5 (`alu`) | 0 | 0 / 0 | 0 |
| LS `vle16`->`vse16` | 3.0 | 4 / 7 (`alu`), 6 (`nop`) | 0 (~0.5 with 1-2 vector fillers) | 0, or >=3 / 0 | 0 |

- **f16 behaves exactly like f32 at every LMUL.** The stall sizes match, the distance needed
  matches, and even the cycles per copy match. For example, at distance 0 LC costs 5.0, 4.0,
  8.0 and 16.1 cycles at m1, m2, m4 and m8, and LS costs 7.0, 8.0, 17.9 and 48.1.
- At the same LMUL an e16 register holds twice as many elements as e32 for the same time, so
  the time per instruction seems to depend on register size in bytes, not element count.
  Per element, f16 has twice the throughput of f32.
- m4 LS readings swing by +/-1-2 cycles in both directions with no trend (noise, as for f32
  and i32). At m8 there is one -2.5 reading (LS, `vadd` filler, N=2), also noise. The m1 LS
  row with `vfadd` fillers has a single 0.60 reading at N=12; the real requirement is 4, as
  for f32.

bf16 can't be tested: the X60 has no Zvfbfmin/Zvfbfwma. A hand-encoded
`vfwcvtbf16.f.f.v` (`.word 0x4a169157`) traps with SIGILL on the board, while the f16
convert `vfwcvt.f.f.v` runs. Clang doesn't vectorize a bf16 loop for `-march=rv64gcv`
either: `a[i] = b[i] + c[i]` on `__bf16` becomes a scalar loop that converts to f32 with a
16-bit shift and calls `__truncsfbf2` per element.

Raw tables: `stall/f16/vfadd_m{1,2,4,8}.txt` (throughput) and
`stall/f16/vfadd_chain_m{1,2,4,8}.txt` (chain).

## Results: `vfmul.vv` (f32 and f16)
Same tests with `vfmul.vv` as the compute op, at e32 (f32) and e16 (f16):
`./run.sh L 12 "" CHAIN vfmul` and `SEW=16 ./run.sh L 12 "" CHAIN vfmul`.

| Pair | m1: stall at distance 0 | m1: vector / scalar instrs needed | m2: stall at distance 0 | m2: vector / scalar instrs needed | m4, m8 |
|---|---|---|---|---|---|
| LC load -> `vfmul` | 1.0 (chain only) | 1 / 4 | 1.0 (chain only) | 1 / 4 | 0 |
| CC `vfmul` -> `vfmul` | 2.0-2.1 (chain only) | 2 / 10 | 2.1 (chain only) | 1 / 7 | 0 |
| CS `vfmul` -> store | 1.4 | 2 / 4 | 0 | 0 / 0 | 0 |
| LS load -> store | 3.0 | 4 / 7 (`alu`), 6 (`nop`) | 0 | 0, or >= 3 / 0 | 0 |

The table is the same for f32 and f16.

- **`vfmul` behaves exactly like `vfadd`, for both f32 and f16.** Stall sizes, gaps and
  cycles per copy all match. At distance 0, LC costs 5.0, 4.0, 8.0 and 16.1 cycles at m1,
  m2, m4 and m8, and CC costs 4.0, 4.0, 8.0 and 16.0.
- So floating-point multiply has the same latency as floating-point add here, unlike integer
  `vmul`, which needs more distance than `vadd` (see the i32 section).
- Noise: f32 m4 LS reads up to about +/-1 cycle with no trend, and single readings of
  0.5-0.6 cycles appear in the m4 CS rows (f32, N = 6 and 8) and in the m8 LS rows
  (f16, `vfadd`/`nop` fillers).
  None of them repeat at neighbouring N.

Raw tables: `stall/f32/vfmul_m{1,2,4,8}.txt`, `stall/f32/vfmul_chain_m{1,2,4,8}.txt`,
`stall/f16/vfmul_m{1,2,4,8}.txt`, `stall/f16/vfmul_chain_m{1,2,4,8}.txt`.

## Minimum gap when sinking (m1, m2)
When moving a producer down toward its consumer (or a consumer up toward its producer), keep
at least this many independent instructions between them. These are the "instrs needed
between" values from the tables above, rounded up where the measured range was uncertain.

f32 and f16 `vfadd`/`vfmul`, and i32 `vadd`:

| Pair | m1: vector instrs | m1: scalar only | m2: vector instrs | m2: scalar only |
|---|---|---|---|---|
| LC load -> compute | **1** | 4 | **1** | 4 |
| CC compute -> compute | **2** | 10 | **1** | 7 |
| CS compute -> store | **3** | 5 | 0 | 0 |
| LS load -> store | **4** | 8 | 0 | 0 |

i32 `vmul` as producer or consumer (`vmadd`/`vmacc` are likely similar, not measured):

| Pair | m1: vector instrs | m1: scalar only | m2: vector instrs | m2: scalar only |
|---|---|---|---|---|
| LC load -> `vmul` | **1** | 4 | **1** | 4 |
| CC `vmul` -> `vmul` | **3** | 12 | **1** | 12 |
| CS `vmul` -> store | **3** | 6 | **2** | 6 |
| LS load -> store | **4** | 8 | 0 | 0 |

At m4 and m8 the gap is 0 for every pair, so sinking costs no stall there.

How to count the gap:
- Count only the instructions strictly between producer and consumer that don't depend on
  the producer. `dist.py` prints this as `dist(vec)` (vector instructions only) and
  `dist(all)` (all instructions), so it can check a schedule after sinking.
- One vector instruction covers about as much as 4 scalar ones at m1. Use the vector column
  if there are vector instructions in between, and the scalar column only when everything
  in between is scalar.
- Mixed vector and scalar fillers weren't measured. A safe rule is to count the vector
  instructions, plus a quarter of the scalar ones.
- If one producer feeds several consumers, or one consumer reads several producers, check
  each pair separately.

Caveats:
- At m1 and m2, the LC and CC stalls only appear in chain mode, where nothing else can
  overlap with them. In a loop with enough independent work they may already be hidden; the
  gap is still the safe choice.
- With scalar fillers only, some smaller gaps happen to be clean (e.g. CC at m2 is clean at
  N = 3 and 5 but stalls at 4 and 6). Those are pairing effects; the table gives the gap
  from which every larger gap is clean.
- The LS gap at m2 is 0, but 1-2 vector fillers cost about 0.5 cycles there; 0 or >= 3 is
  clean.

### Example
```
%92:vrm8  = PseudoVLE32_V_M8 %92, %x_base3, %vl            # x
%101:vrm8 = PseudoVLE32_V_M8 %101, %y_base3, %vl           # y
%sum3:vrm8 = PseudoVFADD_VV_M8_E32 %sum3, %92, %101, 7, %vl # x + y
```
- Pairs: `%92` load -> `vfadd` is LC at distance 1 (the y load is in between), and `%101`
  load -> `vfadd` is LC at distance 0.
- At m8 the LC gap is 0, so neither pair stalls: expected stall 0 cycles.
- The same code at m1 needs a gap of 1 vector instruction for LC. The x pair has it, but the
  y pair doesn't and stalls about 1 cycle when latency-bound. Moving one independent vector
  instruction between the y load and the `vfadd` removes it.
- The loads also pass their own result register as the passthru operand. If that operand
  isn't undef and the policy is tail- or mask-undisturbed, each load also reads the old
  value, which adds a pair with whatever last wrote `%92` / `%101`.

## Check with s279
`stall/s279.sh` compiles `~/tsvc/s279.c` at LMUL 1, 2, 4 and 8 with six llc scheduling options,
runs each version on the board, and uses `dist.py` to print every producer->consumer distance
in the vector loop (`stall/s279.txt`, `stall/s279_m8.txt`). All versions produce the same output hash.

Cycles per call (1000 elements):

| | default | nosched | topdown | bottomup | x60 | generic |
|---|---|---|---|---|---|---|
| m1 | 10527 | **12298** | 11315 | 10556 | 10581 | 10579 |
| m2 | 7807 | 8475 | 7849 | 7727 | 7753 | 7805 |
| m4 | 6880 | 7151 | **6479** | 6844 | 6896 | 6906 |
| m8 | 16447 | **12860** | 15292 | 16530 | 16550 | 16604 |

The m1-m4 numbers are from a rerun and are within about 1% of the first run.

The extra llc flag for each option (the clang command is the same for all):

| Option | llc flag |
|---|---|
| default | (none) |
| nosched | `-enable-misched=false` |
| topdown | `-misched-prera-direction=topdown` |
| bottomup | `-misched-prera-direction=bottomup` |
| x60 | `-mcpu=spacemit-x60` |
| generic | `-mtune=generic` |

- **`-mtune=generic` changes nothing:** it produces byte-identical assembly to `default` at
  every LMUL, because generic is already llc's default tuning. Its cycle differences from
  `default` (up to about 1%) are therefore run-to-run noise.
- **m8 needs forcing:** with `--riscv-v-register-bit-width-lmul=8` alone, the vectorizer still
  picks `<vscale x 8 x i32>` (m4), and the IR is identical to the m4 build. `s279.sh` gets a
  real m8 loop by adding
  `#pragma clang loop vectorize_width(16, scalable)` (written as `_Pragma`) to a copy of the
  source, `stall/s279out/s279_m8.c`. Check in the LLIR: `stall/s279out/m8.llir` has
  31 `<vscale x 16 x i32>` and 12 `<vscale x 16 x i1>` (`m4.llir` has `vscale x 8`), and
  every m8 `.s` file uses `vsetvli ... e32, m8`.
  - The `-lmul=8` flag does allow m8: `-debug-only=loop-vectorize` reports
    `Found feasible scalable VF = vscale x 16`. The cost model then rejects it, because its
    estimated cost per element is 4.3 at `vscale x 16` (m8) and 3.59 at `vscale x 8` (m4).
  - Adding `--riscv-v-fixed-length-vector-lmul-max=8` (to clang via `-mllvm`, to llc, or
    both) doesn't change this. The VF costs and choice stay the same, and with the pragma
    the IR code and assembly are identical to the build without it. The flag only affects
    fixed-length vectors (`<N x i32>`), and the loop vectorizer uses scalable vectors
    (`<vscale x N x i32>`) here. Its default is already 8: forcing a fixed width with
    `-mllvm -force-vector-width=64` produces `<64 x i32>`, which llc lowers to `e32, m8`
    with or without the flag.

- **m1 agrees with the microbenchmark:** `nosched` has the most distance-0 pairs
  (LC 3, CS 4, CC 4 at distance 0) and is 17% slower than `default`.
- **At m4, distance does not explain performance:** `topdown` is the fastest even though
  it has distance-0 LC and CS pairs. It alternates loads with compute so the load and compute
  units run in parallel. `nosched` is the slowest and places the most loads after stores in
  the same iteration (3, against 1 for `default` and `topdown`). That fits the `vle`/`vse`
  cost above, but it has not been confirmed with a dedicated store->load test.

- **m8 is 1.9-2.6x slower than m4, and register spills dominate.** Only 4 register groups
  exist, so every m8 variant spills: 18-23 spill/reload instructions per loop, against 0 at
  m4. The loop grows to 112-159 instructions, against 35 at m4. `nosched` is the fastest m8
  variant because it has the fewest instructions (112), not because of its distances.
  `dist.py` also counts spill stores and reloads as `vse`/`vle`, so the m8 distance lists
  are not comparable with the other LMULs.

Vector op order in the m4 loop:
```
default : L L C C L C L C L C C S C S C C L C C S L C C C C L C S
topdown : L C L C L C L C L C S C C C C S C L C S L C C C C L C S
nosched : L C C L L C C S C L C L C C S L C C C S L C C C C L C S
```

## Conclusions
1. The dependency-distance rule matters at **LMUL = 1 and 2**. See "Minimum gap when
   sinking" for the gap to keep per pair.
   - m1: use 1 vector instruction between for LC, 2 for CC, 2-3 for CS and 4 for LS.
   - m2: use 1 vector instruction between for LC and CC; CS and LS showed no stall.
   - Scalar fillers need several times as many (see the tables).
   - The LC and CC stalls (m1 and m2) only appear when nothing else can overlap with them.
   - f16 behaves exactly like f32, including cycles per copy.
   - `vfmul` behaves exactly like `vfadd`, for f32 and f16.
   - i32: `vadd` behaves like `vfadd`. `vmul` needs more: 3 vector instructions for CC and
     CS at m1, plus a 2-cycle LC stall and a small CS stall at m2.
2. At **LMUL = 4 and 8**, spacing producer and consumer apart gains nothing, for f32,
   f16 and i32, add and multiply alike.
   What helps is overlapping loads with compute and limiting how loads and stores mix.
3. For s279, m8 loses to m4 because of spills, whatever the schedule. m4 with `topdown`
   is the fastest measured version.
4. Possible next test: a store->load (`vse32` then an unrelated `vle32`) pattern, to
   measure directly the cost of a load placed after a store at m4.

## Files (`~/tsvc/stall/`)
| File | Purpose |
|---|---|
| `run.sh [LMUL] [MAXN] [PATTERN] [CHAIN] [OP]` | generate, copy to the board, build and run the microbenchmark (e.g. `./run.sh 1 12 "" 1 vmul`); OP = `vfadd` (default), `vfmul`, `vadd`, `vmul`; prefix `SEW=16` for e16 (f16) |
| `gen.py` | writes `bench.S` and `tests.h` |
| `driver.c` | runs each test with the cycle counter and prints dep / nodep / stall |
| `s279.sh [LMULs]` | s279 check, default LMULs 1 2 4 8 (`BUILD=1` rebuilds llvm first, as `g.sh` does) |
| `s279drv.c` | timing driver for `s279()` |
| `summ.py thr.txt [chain.txt]` | summarize raw output into a stall table (noise threshold 0.6 cycles) |
| `dist.py file.s` | producer->consumer distances in the `vector.body` loop (LMUL read from `vsetvli`) |

Note: the board's binutils is older than llc. `s279.sh` strips the `.attribute` and
`.option arch` lines before assembling on the board.

## Gap used by `ExpandPseudos`
Minimum gap (vector / scalar-only instructions) that `FindFirstUseToSinkTo`, `FindFirstUseToSinkToGroup`,
`ProcessRematLoads` and `ProcessReverseRematChain` keep between a producer and its consumer
(`minStallGap` in `ExpandPseudos.cpp`):

| | m1 (and fractional) | m2 | m4, m8 |
|---|---|---|---|
| LC | 1 / 4 | 1 / 4 | 0 |
| CC | 2 / 10 (int mul: 3 / 12) | 1 / 7 (int mul: 1 / 12) | 0 |
| CS | 3 / 5 (int mul: 3 / 6) | 0 / 0 (int mul: 2 / 6) | 0 |
| LS | 4 / 8 | 0 / 0 | 0 |
