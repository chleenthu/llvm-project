//===-- ExpandPseudos.cpp - Expand Pseudos ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpandPseudos.h"
#include <climits>
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCContext.h"
#include "llvm/IR/Constants.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "expandpseudos"
#define RISCV_INSERT_VSETVLI_NAME "Custom Expand to Pseudos and Sink pass"

char ExpandPseudos::ID = 0;
char &llvm::ExpandPseudosID = ExpandPseudos::ID;

INITIALIZE_PASS(ExpandPseudos, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static cl::opt<bool>
    Copy("custom-copy", cl::init(false), cl::Hidden,
         cl::desc("Turn a COPY of a vector splat (vmv.v.i 0 / vmv.v.x) into "
                  "the splat written directly to the COPY destination"));

static cl::opt<bool>
    Sink("custom-sink", cl::init(false), cl::Hidden,
               cl::desc("Enable sinking"));

static cl::opt<bool>
    a("custom-a", cl::init(false), cl::Hidden,
               cl::desc("Enable affine"));

static cl::opt<bool>
    Thresh("custom-thresh", cl::init(false), cl::Hidden,
           cl::desc("Turn a bound check (vid.v + s1 + ... + sk) < n into "
                    "vid.v < (n - s1 - ... - sk), with the offsets "
                    "subtracted from n in scalar registers, so the "
                    "vadd.vx / disjoint vor.vx chain before the vmslt.vx "
                    "goes away"));

static cl::opt<bool>
    Remat("custom-remat", cl::init(false), cl::Hidden,
          cl::desc("Turn redundant masked reloads into full loads and reload "
                   "whole register loads before distant uses"));

// Minimum distance, in instructions, between two uses of a loaded value for
// the later use to get its own reload, and the number of uses required before
// that gap.
static constexpr unsigned RematGap = 6;
static constexpr unsigned RematMinUses = 2;

static cl::opt<bool>
    Reverse("custom-reverse", cl::init(false), cl::Hidden,
            cl::desc("Reverse rematerialize vector values: with the timing "
                     "of -custom-remat (a use at least RematGap instructions "
                     "after the previous one), recompute the value before "
                     "the distant use by inverting one of its earlier uses "
                     "or by replaying its def chain from a value still live "
                     "there, chosen among the options recorded on its live "
                     "range. Implies -custom-remat"));

ExpandPseudos::ExpandPseudos() : MachineFunctionPass(ID) {
  initializeExpandPseudosPass(*PassRegistry::getPassRegistry());
}

void ExpandPseudos::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervalsWrapperPass>();
  //AU.addRequired<MachineDominatorTreeWrapperPass>();
  //AU.addRequired<MachineCycleInfoWrapperPass>();
  //AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static bool isVectorReg(Register R, const MachineRegisterInfo &MRI) {
  if (R.isVirtual())
    return RISCVRI::isVRegClass(MRI.getRegClass(R)->TSFlags);
  return R.isPhysical() && RISCV::VRRegClass.contains(R);
}

// The independent instructions a consumer needs after the vector producer it
// reads so that it does not stall on the X60 (doc/DocStall.md, "Minimum gap
// when sinking"): Vec when vector instructions are among them, Scalar when
// they are all scalar. From m4 up there is no stall: each instruction keeps
// its unit busy long enough to cover the latency, so the producer can sit
// directly before the consumer.
struct StallGap {
  unsigned Vec = 0, Scalar = 0;
};

// Integer multiplies need more distance than adds; vfmul does not.
static bool isIntMulName(StringRef Name) {
  return Name.starts_with("PseudoVMUL") || Name.starts_with("PseudoVMADD") ||
         Name.starts_with("PseudoVMACC") || Name.starts_with("PseudoVNMSAC") ||
         Name.starts_with("PseudoVNMSUB") || Name.starts_with("PseudoVWMUL") ||
         Name.starts_with("PseudoVWMACC");
}

// LMUL in registers (1 for a fractional LMUL) of an RVV pseudo, or 0 if the
// opcode does not carry one.
static unsigned pseudoLMul(const MCInstrDesc &Desc) {
  if (!RISCVII::hasSEWOp(Desc.TSFlags))
    return 0;
  auto [LMul, Fractional] =
      RISCVVType::decodeVLMUL(RISCVII::getLMul(Desc.TSFlags));
  return Fractional ? 1 : LMul;
}

// LC / CC / CS / LS rows of the table, by the producer's LMUL.
static StallGap minStallGap(unsigned LMul, bool ProducerLoads,
                            bool ConsumerStores, bool IntMul) {
  if (LMul >= 4)
    return {0, 0};
  bool M1 = LMul <= 1;
  if (ProducerLoads && ConsumerStores) // LS
    return M1 ? StallGap{4, 8} : StallGap{0, 0};
  if (ProducerLoads) // LC
    return {1, 4};
  if (ConsumerStores) // CS
    return M1 ? (IntMul ? StallGap{3, 6} : StallGap{3, 5})
              : (IntMul ? StallGap{2, 6} : StallGap{0, 0});
  // CC
  return M1 ? (IntMul ? StallGap{3, 12} : StallGap{2, 10})
            : (IntMul ? StallGap{1, 12} : StallGap{1, 7});
}

// The gap Consumer needs after Producer, an existing instruction. Its LMUL
// comes from the pseudo, or from its result's register class (COPY, whole
// register loads).
static StallGap minStallGap(const MachineInstr &Producer,
                            const MachineInstr &Consumer,
                            const MachineRegisterInfo &MRI,
                            const TargetRegisterInfo &TRI,
                            const TargetInstrInfo &TII) {
  unsigned LMul = pseudoLMul(Producer.getDesc());
  if (!LMul) {
    LMul = 1;
    if (Producer.getNumExplicitDefs() && Producer.getOperand(0).isReg()) {
      Register R = Producer.getOperand(0).getReg();
      if (R.isVirtual() && isVectorReg(R, MRI))
        LMul = TRI.getRegClassWeight(MRI.getRegClass(R)).RegWeight;
    }
  }
  bool IntMul = isIntMulName(TII.getName(Producer.getOpcode())) ||
                isIntMulName(TII.getName(Consumer.getOpcode()));
  StallGap Need =
      minStallGap(LMul, Producer.mayLoad(), Consumer.mayStore(), IntMul);
  LLVM_DEBUG(dbgs() << "  Stall gap: LMUL " << LMul << ", "
                    << (Producer.mayLoad() ? "L" : "C")
                    << (Consumer.mayStore() ? "S" : "C")
                    << (IntMul ? " int mul" : "") << ", need " << Need.Vec
                    << " vector / " << Need.Scalar << " scalar, producer "
                    << TII.getName(Producer.getOpcode()) << "\n");
  return Need;
}

static bool isVectorInstr(const MachineInstr &MI,
                          const MachineRegisterInfo &MRI) {
  return llvm::any_of(MI.operands(), [&](const MachineOperand &MO) {
    return MO.isReg() && MO.getReg() && isVectorReg(MO.getReg(), MRI);
  });
}

// Whether Vec vector and Scalar scalar instructions cover Need. Mixed
// fillers were not measured; a scalar one counts a quarter of a vector one.
static bool coversStallGap(StallGap Need, unsigned Vec, unsigned Scalar) {
  if (Vec)
    return 4 * Vec + Scalar >= 4 * Need.Vec;
  return Scalar >= Need.Scalar;
}

// The latest point before Consumer to insert a producer at so that Need sits
// between them, without walking back to Limit (the producer's own position,
// or an instruction the new value must follow). None if the gap does not fit
// after Limit. With no gap needed, that is Consumer itself.
static std::optional<MachineBasicBlock::iterator>
stallGapInsertPoint(MachineInstr &Consumer, const MachineInstr *Limit,
                    StallGap Need, const MachineRegisterInfo &MRI) {
  MachineBasicBlock &MBB = *Consumer.getParent();
  MachineBasicBlock::iterator It = Consumer.getIterator();
  unsigned Vec = 0, Scalar = 0;
  while (!coversStallGap(Need, Vec, Scalar)) {
    do {
      if (It == MBB.begin())
        return std::nullopt;
      --It;
    } while (It->isDebugInstr());
    if (&*It == Limit)
      return std::nullopt;
    ++(isVectorInstr(*It, MRI) ? Vec : Scalar);
  }
  return It;
}

// Sum of live interval lengths (SLIL) of the vector virtual registers used in
// the block, copied from RISCVRegisterPressure.cpp. The last use of a register
// does not count towards register pressure, so the length of a register is
// (Last - First) * LMUL, where First and Last are the first and last
// instruction referencing it, counted in instructions that touch a vector
// register. Used to check the incremental update of the lengths.
static unsigned computeSLIL(const MachineBasicBlock &MBB,
                            const MachineRegisterInfo &MRI,
                            const TargetRegisterInfo &TRI) {
  DenseMap<Register, std::pair<unsigned, unsigned>> Range;
  unsigned Row = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    bool Touches = false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg() || !isVectorReg(MO.getReg(), MRI))
        continue;
      Touches = true;
      if (MO.getReg().isVirtual()) {
        auto It = Range.try_emplace(MO.getReg(), Row, Row).first;
        It->second.second = Row;
      }
    }
    if (Touches)
      ++Row;
  }
  unsigned Sum = 0;
  for (auto &KV : Range) {
    unsigned LMUL = TRI.getRegClassWeight(MRI.getRegClass(KV.first)).RegWeight;
    Sum += (KV.second.second - KV.second.first) * LMUL;
  }
  return Sum;
}

// Matches whole register loads named VL<NF>RE<EEW>_V, e.g. VL4RE32_V.
static bool isWholeRegLoadName(StringRef Name) {
  if (!Name.consume_front("VL") || !Name.consume_back("_V"))
    return false;
  StringRef NF, EEW;
  std::tie(NF, EEW) = Name.split("RE");
  auto IsNumber = [](StringRef S) {
    return !S.empty() && llvm::all_of(S, isDigit);
  };
  return IsNumber(NF) && IsNumber(EEW);
}

// Turn %dst = COPY %src, where %src is a splat (vmv.v.i of 0, or vmv.v.x)
// with an undef passthru, into the same splat writing %dst directly, at any
// LMUL. %dst then no longer depends on %src, so %src can die earlier.
bool ExpandPseudos::LowerCopy(MachineBasicBlock &MBB, MachineInstr &MI) {
  if (MI.getNumOperands() < 2 || !MI.getOperand(0).isReg() ||
      !MI.getOperand(1).isReg() || MI.getOperand(0).getSubReg() ||
      MI.getOperand(1).getSubReg())
    return false;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  if (!DstReg.isVirtual() || !SrcReg.isVirtual() ||
      !isVectorReg(DstReg, *MRI) || !isVectorReg(SrcReg, *MRI))
    return false;

  // The value the COPY reads: the last def of SrcReg before it in the block.
  MachineInstr *Def = nullptr;
  for (auto It = MI.getIterator(); It != MBB.begin();) {
    --It;
    if (It->modifiesRegister(SrcReg, TRI)) {
      Def = &*It;
      break;
    }
  }
  if (!Def)
    return false;
  StringRef Name = TII->getName(Def->getOpcode());
  bool IsVI = Name.starts_with("PseudoVMV_V_I_");
  bool IsVX = Name.starts_with("PseudoVMV_V_X_");
  if ((!IsVI && !IsVX) || Name.contains("MASK") ||
      Def->getNumExplicitDefs() != 1 || Def->getNumExplicitOperands() < 4 ||
      Def->getOperand(0).getSubReg())
    return false;
  const MachineOperand &Passthru = Def->getOperand(1);
  const MachineOperand &Val = Def->getOperand(2);
  if (!Passthru.isReg() || !(Passthru.isUndef() || !Passthru.getReg()))
    return false;
  if (IsVI ? !(Val.isImm() && Val.getImm() == 0) : !Val.isReg())
    return false;
  // Its register operands (the scalar, VL) must still hold the same values.
  for (const MachineOperand &MO : Def->explicit_uses()) {
    if (!MO.isReg() || !MO.getReg() || MO.isUndef())
      continue;
    for (auto It = std::next(Def->getIterator()); It != MI.getIterator(); ++It)
      if (It->modifiesRegister(MO.getReg(), TRI))
        return false;
  }
  if (const TargetRegisterClass *RC = TII->getRegClass(Def->getDesc(), 0))
    if (!MRI->constrainRegClass(DstReg, RC))
      return false;

  LLVM_DEBUG(dbgs() << "lowerCopy: " << MI);
  MachineInstr *NewMI = MBB.getParent()->CloneMachineInstr(Def);
  NewMI->getOperand(0).setReg(DstReg);
  NewMI->getOperand(0).setIsDead(false);
  NewMI->getOperand(1).setReg(DstReg); // undef passthru, tied to the def
  for (MachineOperand &MO : NewMI->explicit_uses())
    if (MO.isReg() && MO.getReg() && !MO.isUndef()) {
      MO.setIsKill(false);
      // Def may have been their last use.
      if (MO.getReg().isVirtual())
        MRI->clearKillFlags(MO.getReg());
    }
  NewMI->setDebugLoc(MI.getDebugLoc());
  MBB.insert(MI.getIterator(), NewMI);
  MI.eraseFromParent();
  // NewMI was inserted, and MI erased, without updating SlotIndexes /
  // LiveIntervals: LIS no longer matches the instruction list, so later code
  // must not call into it.
  LISValid = false;
  return true;
}

// Locate the first non-debug use of Reg after Def in Def's block. The uses come
// from the register's use list, so the walk only tests set membership instead
// of scanning the operands of every instruction. Returns null if some use is in
// another block, or there is none. Between, if given, is the number of
// non-debug instructions between Def and the first use.
static MachineInstr *findFirstUseInBlock(MachineInstr &Def, Register Reg,
                                         MachineRegisterInfo &MRI,
                                         unsigned *Between = nullptr) {
  MachineBasicBlock *MBB = Def.getParent();
  SmallPtrSet<const MachineInstr *, 8> Uses;
  for (MachineInstr &U : MRI.use_nodbg_instructions(Reg)) {
    if (U.getParent() != MBB)
      return nullptr;
    if (&U != &Def)
      Uses.insert(&U);
  }
  unsigned Count = 0;
  for (auto It = std::next(Def.getIterator()); It != MBB->end(); ++It) {
    if (It->isDebugInstr())
      continue;
    if (Uses.count(&*It)) {
      if (Between)
        *Between = Count;
      return &*It;
    }
    ++Count;
  }
  return nullptr;
}

bool ExpandPseudos::FindFirstUseToSinkTo(
    MachineInstr &MI, AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  FindFirstUseToSinkTo.\n");
  if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg())
    return false;
  Register Reg = MI.getOperand(0).getReg();
  if (!Reg.isVirtual())
    return false;
  MachineBasicBlock *MBB = MI.getParent();
  bool SawStore = false;
  if (!MI.isSafeToMove(SawStore) || MI.isConvergent()) {
    LLVM_DEBUG(dbgs() << "  Not safe to move.\n");
    return false;
  }

  // Find the first use in this block. Uses elsewhere keep the value live past
  // the block, so leave those alone.
  MachineInstr *FirstUse = findFirstUseInBlock(MI, Reg, *MRI);
  if (!FirstUse) {
    LLVM_DEBUG(dbgs() << "  No first use.\n");
    return false;
  }
  // Stop short of the first use by the gap that keeps it from stalling on MI
  // (none at m4 / m8). N is the number of instructions MI passes.
  std::optional<MachineBasicBlock::iterator> InsertPt = stallGapInsertPoint(
      *FirstUse, &MI, minStallGap(MI, *FirstUse, *MRI, *TRI, *TII), *MRI);
  unsigned N = 0;
  if (InsertPt)
    for (auto It = std::next(MI.getIterator()); It != *InsertPt; ++It)
      if (!It->isDebugInstr())
        ++N;
  if (!N) {
    LLVM_DEBUG(dbgs() << "  Already within the stall gap of its first use.\n");
    return false;
  }

  // Nothing crossed may conflict with the registers or memory MI touches.
  for (auto It = std::next(MI.getIterator()); It != *InsertPt; ++It) {
    MachineInstr &I = *It;
    if (I.isDebugInstr())
      continue;
    if (I.isCall() || I.hasUnmodeledSideEffects())
      return false;
    if (MI.mayLoad() && I.mayStore())
      return false;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg())
        continue;
      for (const MachineOperand &IO : I.operands()) {
        if (!IO.isReg() || !IO.getReg())
          continue;
        Register A = MO.getReg(), B = IO.getReg();
        bool Overlap = A == B || (A.isPhysical() && B.isPhysical() &&
                                  TRI->regsOverlap(A, B));
        if (Overlap && (MO.isDef() || IO.isDef())) {
          LLVM_DEBUG(dbgs() << "  Conflict with: " << I);
          return false;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Sink " << MI << "  before " << **InsertPt
                    << "  for use " << *FirstUse);
  LLVM_DEBUG(dbgs() << "  Between: " << N << "\n");

  // Predict the effect on the vector register pressure and live interval
  // lengths. Sinking MI moves its DEF (and its LAST USEs) N instructions
  // later, so:
  //  - the N instructions it passes lose MI's net change (+DEF, -LAST USE),
  //  - MI's own point gains the net change of the N instructions passed,
  //    which are the DEFs and LAST USEs found among the events in the window,
  //  - the registers of MI: a first reference moves N instructions later (the
  //    interval gets shorter), a last reference moves later (longer).
  BlockUsage &BU = Usage[MBB];
  unsigned P = BU.Pos.lookup(&MI);
  unsigned Rows = BU.VecPrefix[P + N + 1] - BU.VecPrefix[P + 1];
  auto Weight = [&](Register R) {
    return (int)TRI->getRegClassWeight(MRI->getRegClass(R)).RegWeight;
  };

  // What every instruction MI passes gains in RP: MI's DEFs are not live yet
  // (-LMUL each) and the registers of its LAST USEs are still live (+LMUL each).
  int DeltaMI = 0;
  for (unsigned I = BU.NextEvent[P];
       I < BU.Events.size() && BU.Events[I].Pos == P; ++I)
    if (BU.Events[I].Pressure)
      DeltaMI += BU.Events[I].IsDef ? -BU.Events[I].W : BU.Events[I].W; // update passby RP

  // LIL is counted in rows (instructions touching a vector register) and MI is
  // one row. The registers of MI: a first reference moves Rows later (shorter
  // by Rows * LMUL), a last reference moves Rows later (longer).
  DenseMap<Register, int> LILChange;
  SmallSet<Register, 4> Seen; // a register can be an operand more than once
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual() ||
        !isVectorReg(MO.getReg(), *MRI) || !Seen.insert(MO.getReg()).second)
      continue;
    Register R = MO.getReg();
    auto Range = BU.Range.lookup(R);
    if (Range.first == P)
      LILChange[R] -= (int)Rows * Weight(R); // update sunk LIL
    if (Range.second == P)
      LILChange[R] += (int)Rows * Weight(R);
  }

  // One walk over the DEFs and LAST USEs in (P, P + N], starting at the first
  // event after MI and visiting no other instruction. Each one updates both:
  //  - RP: MI's point gains a DEF and loses a LAST USE it passes,
  //  - LIL: MI is one row that moves from before to after the window.
  //    A DEF whose register is used after the window now has MI between its
  //    first and last reference (+LMUL). A LAST USE whose register was first
  //    referenced before MI no longer has MI between them (-LMUL).
  // A register MI reads whose last use is in the window is now last used by
  // MI, so it stays live through the rest of the window: its LIL grows by the
  // rows after that last use and the RP of those instructions by its LMUL.
  SmallSet<Register, 4> MIUses;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isUse() && !MO.isUndef() && MO.getReg().isVirtual() &&
        isVectorReg(MO.getReg(), *MRI))
      MIUses.insert(MO.getReg());
  SmallVector<std::pair<unsigned, int>, 4> Extend; // (position, LMUL)
  int WindowNet = 0;
  unsigned NumEvents = 0;
  for (unsigned I = BU.NextEvent[P + 1];
       I < BU.Events.size() && BU.Events[I].Pos <= P + N; ++I) {
    const BlockUsage::RefEvent &E = BU.Events[I];
    ++NumEvents;
    if (E.Pressure)
      WindowNet += E.IsDef ? E.W : -E.W; // update sunk RP
    if (E.Reg.isVirtual()) {
      auto Range = BU.Range.lookup(E.Reg);
      if (E.IsDef && Range.second > P + N) // R's last use is after the window
        LILChange[E.Reg] += E.W; // update passby LIL
      if (!E.IsDef && MIUses.count(E.Reg)) {
        LILChange[E.Reg] += // update passby LIL, sunk is not last use but becomes last use after sink
            E.W * (int)(BU.VecPrefix[P + N + 1] - BU.VecPrefix[E.Pos + 1]);
        Extend.push_back({E.Pos, E.W});
      } else if (!E.IsDef && Range.first < P) { // R's first reference is before MI
        LILChange[E.Reg] -= E.W;
      }
    }
    LLVM_DEBUG(dbgs() << "    " << (E.IsDef ? "DEF" : "LAST USE") << " of "
                      << printReg(E.Reg, TRI) << " at +" << E.Pos - P
                      << ", LMUL " << E.W << "\n");
  }
  int LILDelta = 0;
  unsigned LILBefore = 0;
  for (auto &KV : BU.LIL)
    LILBefore += KV.second;
  for (auto &KV : LILChange)
    LILDelta += KV.second;
  LLVM_DEBUG(dbgs() << "  RP: passed instructions " << DeltaMI
                    << " each, MI point " << WindowNet << " from " << NumEvents
                    << " events\n");

  MBB->splice(*InsertPt, MBB, MI.getIterator());
  if (LISValid)
    LIS->handleMove(MI);
  for (MachineOperand &MO : MI.all_uses())
    RegsToClearKillFlags.insert(MO.getReg());

  // Apply the prediction. The N passed instructions are now just before MI.
  unsigned OldRP = VRPressure.lookup(&MI);
  auto Passed = MI.getIterator();
  for (unsigned i = 0; i < N;) {
    --Passed;
    if (Passed->isDebugInstr())
      continue;
    unsigned Q = P + N - i; // position it had before the move
    VRPressure[&*Passed] += DeltaMI;
    for (auto &X : Extend)
      if (X.first <= Q)
        VRPressure[&*Passed] += X.second; // update passby RP, sunk is not last use but becomes last use after sink
    ++i;
  }
  VRPressure[&MI] += WindowNet;
  LLVM_DEBUG(dbgs() << "  RP of MI: " << OldRP << " -> " << VRPressure[&MI]
                    << "\n");
  DenseMap<Register, unsigned> PredictedLIL;
  for (auto &E : LILChange)
    PredictedLIL[E.first] = BU.LIL.lookup(E.first) + E.second;
  unsigned PredictedSLIL = LILBefore + LILDelta;

  computeBlockUsage(*MBB);

  // Verify against a fresh RegPressureTracker run and the recomputed lengths.
  LLVM_DEBUG({
    if (LISValid) {
      DenseMap<const MachineInstr *, unsigned> Fresh;
      computeBlockPressure(*MBB, Fresh);
      unsigned Bad = 0;
      for (MachineInstr &I : *MBB) {
        if (I.isDebugInstr())
          continue;
        if (Fresh.lookup(&I) != VRPressure.lookup(&I)) {
          dbgs() << "  RP MISMATCH at " << I << "    predicted "
                 << VRPressure.lookup(&I) << ", RPTracker " << Fresh.lookup(&I)
                 << "\n";
          ++Bad;
        }
      }
      if (!Bad)
        dbgs() << "  RP verified against RPTracker: OK\n";
      for (auto &KV : Fresh)
        VRPressure[KV.first] = KV.second;
    } else {
      dbgs() << "  RP verification skipped: LiveIntervals are stale\n";
    }
    for (auto &E : PredictedLIL) {
      unsigned Now = Usage[MBB].LIL.lookup(E.first);
      dbgs() << "  LIL of " << printReg(E.first, TRI) << ": "
             << (int)E.second - LILChange[E.first] << " -> predicted "
             << E.second << ", recomputed " << Now
             << (Now == E.second ? "  OK\n" : "  MISMATCH\n");
    }
    unsigned SLILNow = computeSLIL(*MBB, *MRI, *TRI);
    dbgs() << "  SLIL: " << LILBefore << " -> predicted " << PredictedSLIL
           << ", computeSLIL " << SLILNow
           << (SLILNow == PredictedSLIL ? "  OK\n" : "  MISMATCH\n");
  });
  return true;
}

bool ExpandPseudos::FindFirstUseToSinkToGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "  FindFirstUseToSinkToGroup with "
                    << InstrsToSink.size() << " instructions.\n");
  if (InstrsToSink.empty())
    return false;
  MachineInstr &Primary = *InstrsToSink[0];
  MachineBasicBlock *MBB = Primary.getParent();
  for (MachineInstr *MI : InstrsToSink)
    if (MI->getParent() != MBB)
      return false;
  Register Reg = Primary.getOperand(0).getReg();

  MachineInstr *FirstUse = findFirstUseInBlock(Primary, Reg, *MRI);
  if (!FirstUse) {
    LLVM_DEBUG(dbgs() << "  First use not found.\n");
    return false;
  }

  // The load goes as late as the stall gap to its first use allows (none at
  // m4 / m8), the rest of the group right before it.
  std::optional<MachineBasicBlock::iterator> InsertPt = stallGapInsertPoint(
      *FirstUse, &Primary, minStallGap(Primary, *FirstUse, *MRI, *TRI, *TII),
      *MRI);
  unsigned Dist = 0;
  if (InsertPt)
    for (auto It = std::next(Primary.getIterator()); It != *InsertPt; ++It)
      if (!It->isDebugInstr())
        ++Dist;
  if (!Dist) {
    LLVM_DEBUG(dbgs() << "  Already within the stall gap of its first use.\n");
    return false;
  }
  MachineBasicBlock::iterator Target = *InsertPt;
  LLVM_DEBUG(dbgs() << "  Between: " << Dist << "\n");
  LLVM_DEBUG(dbgs() << "  Sink before: " << *Target);

  auto Overlaps = [&](Register A, Register B) {
    if (A == B)
      return true;
    return A.isPhysical() && B.isPhysical() && TRI->regsOverlap(A, B);
  };

  // Check that nothing crossed conflicts with the moved instructions.
  for (MachineInstr *M : InstrsToSink) {
    for (auto It = std::next(M->getIterator()); It != Target; ++It) {
      MachineInstr &I = *It;
      if (I.isDebugInstr())
        continue;
      if (I.isCall() || I.hasUnmodeledSideEffects())
        return false;
      if (M->mayLoad() && I.mayStore())
        return false;
      if (M->mayStore() && (I.mayLoad() || I.mayStore()))
        return false;
      for (const MachineOperand &MO : M->operands()) {
        if (!MO.isReg() || !MO.getReg())
          continue;
        for (const MachineOperand &IO : I.operands()) {
          if (!IO.isReg() || !IO.getReg() || !Overlaps(MO.getReg(), IO.getReg()))
            continue;
          // Def of M conflicts with any access; use of M conflicts with a def.
          if (MO.isDef() || IO.isDef()) {
            LLVM_DEBUG(dbgs() << "  Conflict with: " << I);
            return false;
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "  Sinking group of instructions\n");
  for (MachineInstr *MI : InstrsToSink) {
    LLVM_DEBUG(dbgs() << "  " << *MI);
  }
  MachineBasicBlock::iterator InsertPos = Target;
  for (auto It = InstrsToSink.begin(); It != InstrsToSink.end(); ++It) {
    MachineInstr *MI = *It;
    MachineBasicBlock::iterator CurrPos = MI->getIterator();
    MBB->splice(InsertPos, MBB, CurrPos);
    InsertPos = MI->getIterator();
    for (MachineOperand &MO : MI->all_uses())
      RegsToClearKillFlags.insert(MO.getReg());
  }
  LISValid = false;
  computeBlockUsage(*MBB);
  return true;
}

void ExpandPseudos::ProcessInSameBlock(MachineFunction &MF) {
  int64_t BlockSize = 128;
  for (const auto &BB : MF.getFunction()) {
    for (const auto &I : BB) {
      if (auto *VTy = dyn_cast<FixedVectorType>(I.getType())) {
        BlockSize = VTy->getNumElements();
      }
    }
  }
  LLVM_DEBUG(dbgs()<<"BlockSize: "<<BlockSize<<"\n");
  DenseSet<Register> SinkRegs;
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs()<<"ProcessInSameBlock.\n");
    AllSuccsCache AllSuccessors;

    // Sink LOAD
    // Walk the basic block bottom-up, as MachineSink.cpp does: sinking moves
    // an instruction later in the block, so processing from the end means a
    // sunk instruction is never revisited and never invalidates the
    // iterator for the instruction still to be processed.
    if (!MBB.empty()) {
      MachineBasicBlock::iterator It = std::prev(MBB.end());
      bool ProcessedBegin;
      do {
        MachineInstr &MI = *It;

        // Predecrement It (if it's not begin) so that it isn't invalidated by
        // sinking MI later in the block.
        ProcessedBegin = It == MBB.begin();
        if (!ProcessedBegin)
          --It;

        const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
        StringRef Name = TII->getName(MI.getOpcode());
        // Sink SinkMI's result toward its first use, once per register.
        auto TrySink = [&](MachineInstr &SinkMI) {
          if (!SinkRegs.insert(SinkMI.getOperand(0).getReg()).second)
            return;
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<SinkMI);
          if (FindFirstUseToSinkTo(SinkMI, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"  Sink success.\n");
          }
        };
        // A splat sinks only when its passthru (operand 1) is its own result.
        if (Name.contains("PseudoVMV_V_I")) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &MO = MI.getOperand(1);
          if (Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg())
            TrySink(MI);
        }
        // A compare's operand 1 is its source.
        if (Name.contains("PseudoVMSLE_VI_M")) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &Src = MI.getOperand(1);
          if (Dst.isReg() && Src.isReg())
            TrySink(MI);
        }
        if (Name.contains("PseudoVLE32_V_M") || isWholeRegLoadName(Name)) {
          const MachineOperand &Dst = MI.getOperand(0);
          const MachineOperand &Src = MI.getOperand(1);
          if (Dst.isReg() && Src.isReg()) {
            Register DstReg = Dst.getReg();
            if (!SinkRegs.count(DstReg) && !RematRegs.count(DstReg) &&
                MI.memoperands_begin() != MI.memoperands_end()) {
              const MachineMemOperand *MMO = *MI.memoperands_begin();
              if (const Value *PtrVal = MMO->getValue()) {
                SinkRegs.insert(DstReg);
                LLVM_DEBUG(dbgs()<<"Prepare to sink Group "<<MI);
                SmallVector<MachineInstr*, 4> InstructionsToSink;
                InstructionsToSink.push_back(&MI);
                Register LDReg = MI.getOperand(0).getReg();
                Register MaskReg = 0;
                for (MachineOperand &MO : MI.all_uses()) {
                  if (MO.isReg() && !MO.getReg().isVirtual()) {
                    LLVM_DEBUG(dbgs() <<"  MaskReg: " <<MO<<"\n");
                    MaskReg = MO.getReg();
                    break;
                  }
                }
                if (MaskReg) {
                  MachineInstr *MaskDef = nullptr;
                  for (auto It = MI.getIterator(); It != MI.getParent()->begin(); --It) {
                    MachineInstr &Candidate = *std::prev(It);
                    //LLVM_DEBUG(dbgs() <<"  Reverse to find Mask: " <<Candidate);
                    for (MachineOperand &DefMO : Candidate.all_defs()) {
                      if (DefMO.isReg() && DefMO.getReg() == MaskReg) {
                        MaskDef = &Candidate;
                        break;
                      }
                    }
                    if (MaskDef)
                      break;
                  }
                  if (MaskDef &&
                      TII->getName(MaskDef->getOpcode()).contains("COPY")) {
                    LLVM_DEBUG(dbgs() <<"  Mask: " <<*MaskDef);
                    InstructionsToSink.push_back(MaskDef);
                  }
                }
                for (MachineInstr &UseMI : MRI->def_instructions(LDReg)) {
                  if (&UseMI == &MI) continue;
                  StringRef Name = TII->getName(UseMI.getOpcode());
                  if (Name.contains("PseudoVMV_V_I")) {
                    LLVM_DEBUG(dbgs() <<"  VMV setup: " <<UseMI);
                    InstructionsToSink.push_back(&UseMI);
                    break;
                  }
                }
                if (FindFirstUseToSinkToGroup(InstructionsToSink, AllSuccessors)) {
                  LLVM_DEBUG(dbgs()<<"  Sink Group to first use success.\n");
                }
              }
            }
          }
        }
      } while (!ProcessedBegin);
    }
    SeenDbgUsers.clear();
    SeenDbgVars.clear();
    CachedRegisterPressure.clear();
  }
}

static std::optional<unsigned> findPseudo(const TargetInstrInfo *TII,
                                          const Twine &Name);

// Turn
//   %id = PseudoVID_V_<L> undef, VL, SEW
//   %a  = PseudoVADD_VX_<L> undef, %id, %s1, VL, SEW   ; or a disjoint VOR_VX
//   %b  = PseudoVADD_VX_<L> undef, %a, %s2, VL, SEW
//   %m  = PseudoVMSLT_VX_<L> %b, %n, VL, SEW
// into
//   %t1 = SUB %n, %s2
//   %t2 = SUB %t1, %s1
//   %m  = PseudoVMSLT_VX_<L> %id, %t2, VL, SEW
// and erase the chain once it is dead. This assumes the index arithmetic does
// not overflow SEW, as Triton offsets do not; a disjoint OR is exactly an add.
// Only done when %b has no other use, so the chain really goes away and only
// %id stays live.
bool ExpandPseudos::ProcessThreshold(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool BlockChanged = false;
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      StringRef Name = TII->getName(MI.getOpcode());
      if (!Name.consume_front("PseudoVMSLT_VX_") || Name.contains("MASK"))
        continue;
      std::optional<unsigned> VIDOpc = findPseudo(TII, "PseudoVID_V_" + Name);
      std::optional<unsigned> VADDOpc =
          findPseudo(TII, "PseudoVADD_VX_" + Name);
      std::optional<unsigned> VOROpc = findPseudo(TII, "PseudoVOR_VX_" + Name);
      if (!VIDOpc || !VADDOpc || !VOROpc)
        continue;
      const MachineOperand &VL = MI.getOperand(3);
      int64_t SEW = MI.getOperand(4).getImm();
      auto SameVL = [&](const MachineOperand &Op) {
        return VL.isImm() ? Op.isImm() && Op.getImm() == VL.getImm()
                          : Op.isReg() && Op.getReg() == VL.getReg();
      };
      auto UsableScalar = [](const MachineOperand &Op) {
        return Op.isReg() &&
               (Op.getReg().isVirtual() || Op.getReg() == RISCV::X0);
      };
      // Whether R is used other than by Except and by its def's own tied
      // undef passthru.
      auto UsedElsewhere = [&](Register R, const MachineInstr *Except) {
        for (const MachineInstr &U : MRI->use_nodbg_instructions(R))
          if (&U != Except && !U.modifiesRegister(R, TRI))
            return true;
        return false;
      };
      Register Head = MI.getOperand(1).getReg();
      const MachineOperand &N = MI.getOperand(2);
      if (!Head.isVirtual() || UsedElsewhere(Head, &MI) || !UsableScalar(N))
        continue;
      // Walk from the compare up to the vid.v, collecting the scalar offsets.
      SmallVector<MachineInstr *, 4> Chain;
      SmallVector<Register, 4> Offsets;
      MachineInstr *VID = nullptr;
      for (Register R = Head; R.isVirtual();) {
        MachineInstr *Def = MRI->getUniqueVRegDef(R);
        if (!Def)
          break;
        unsigned Opc = Def->getOpcode();
        if (Opc == *VIDOpc) {
          if (SameVL(Def->getOperand(2)) && Def->getOperand(3).getImm() == SEW)
            VID = Def;
          break;
        }
        if (!(Opc == *VADDOpc ||
              (Opc == *VOROpc && Def->getFlag(MachineInstr::Disjoint))))
          break;
        const MachineOperand &Pass = Def->getOperand(1);
        if ((Pass.isReg() && Pass.getReg() && !Pass.isUndef()) ||
            !SameVL(Def->getOperand(4)) || Def->getOperand(5).getImm() != SEW ||
            !UsableScalar(Def->getOperand(3)))
          break;
        Chain.push_back(Def);
        Offsets.push_back(Def->getOperand(3).getReg());
        R = Def->getOperand(2).getReg();
      }
      if (!VID || Chain.empty())
        continue;
      LLVM_DEBUG(dbgs() << "Threshold: " << MI);
      // n - s_k - ... - s_1, right before the compare.
      Register T = N.getReg();
      for (Register Off : Offsets) {
        Register NewT = MRI->createVirtualRegister(&RISCV::GPRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(RISCV::SUB), NewT)
            .addReg(T)
            .addReg(Off);
        T = NewT;
      }
      // The offsets, n and the vid.v are now used later than before.
      for (Register Off : Offsets)
        if (Off.isVirtual())
          MRI->clearKillFlags(Off);
      if (N.getReg().isVirtual())
        MRI->clearKillFlags(N.getReg());
      MRI->clearKillFlags(VID->getOperand(0).getReg());
      MI.getOperand(1).setReg(VID->getOperand(0).getReg());
      MI.getOperand(2).setReg(T);
      MI.getOperand(2).setIsKill(false);
      // Erase the chain from the compare up while it is dead.
      for (MachineInstr *Def : Chain) {
        if (UsedElsewhere(Def->getOperand(0).getReg(), nullptr))
          break;
        Def->eraseFromParent();
      }
      LLVM_DEBUG(dbgs() << "  Threshold VMSLT: " << MI);
      BlockChanged = true;
    }
    if (BlockChanged) {
      computeBlockUsage(MBB);
      Changed = true;
    }
  }
  if (Changed)
    LISValid = false;
  return Changed;
}

void ExpandPseudos::ProcessInSameAffine(MachineFunction &MF) {
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs() << "ProcessInSameAffine.\n");
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI = *I;
      LLVM_DEBUG(dbgs() << MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      // vid.v at any LMUL; its partners below must have the same LMUL.
      StringRef VIDName = TII->getName(MI.getOpcode());
      if (VIDName.consume_front("PseudoVID_V_") && !VIDName.contains("MASK") &&
          MI.getOperand(2).isReg()) {
        std::optional<unsigned> VOROpc =
            findPseudo(TII, "PseudoVOR_VX_" + VIDName);
        std::optional<unsigned> VADDOpc =
            findPseudo(TII, "PseudoVADD_VX_" + VIDName);
        std::optional<unsigned> VMSLTOpc =
            findPseudo(TII, "PseudoVMSLT_VX_" + VIDName);
        if (!VOROpc || !VADDOpc || !VMSLTOpc)
          continue;
        MachineOperand &VIDDest = MI.getOperand(0);
        MachineInstr *VOR_Orig = nullptr;
        Register VIDReg = VIDDest.getReg();
        SmallVector<MachineInstr *, 8> Uses;
        for (MachineInstr &UseMI : MRI->use_instructions(VIDReg)) {
          if (&UseMI == &MI) continue;
          if (UseMI.getOpcode() == *VOROpc) {
            MachineOperand &SrcReg = UseMI.getOperand(2);
            if (SrcReg.isReg() && SrcReg.getReg() == VIDReg) {
              LLVM_DEBUG(dbgs() <<"  VOR_Orig: " <<UseMI);
              VOR_Orig = &UseMI;
            }
          } else if (UseMI.getOpcode() == *VADDOpc) {
            MachineOperand &SrcReg = UseMI.getOperand(2);
            if (SrcReg.isReg() && SrcReg.getReg() == VIDReg) {
              LLVM_DEBUG(dbgs() <<"  Use VID result: " <<UseMI);
              Uses.push_back(&UseMI);
            }
          }
        }
        if (!VOR_Orig) {
          LLVM_DEBUG(dbgs() << "  No VOR_Orig found, skipping\n");
          continue;
        }
        // VOR_Orig would like to move up to right after MI (the VID), but
        // that is only safe if nothing between MI and VOR_Orig's current
        // position defines a register VOR_Orig reads (e.g. its scalar
        // operand, such as a base address computed after the VID but before
        // the OR): moving past such a def would use it before it is
        // defined. Instead of giving up in that case, move VOR_Orig up only
        // as far as right after the last such dependency, which is still
        // safe and still shortens the gap.
        MachineInstr *LastDep = nullptr;
        for (auto It = std::next(MI.getIterator());
             It != VOR_Orig->getIterator(); ++It) {
          for (const MachineOperand &Def : It->all_defs()) {
            if (!Def.isReg() || !Def.getReg())
              continue;
            for (const MachineOperand &Use : VOR_Orig->all_uses()) {
              if (Use.isReg() && Use.getReg() == Def.getReg()) {
                LastDep = &*It;
                break;
              }
            }
          }
        }
        MachineBasicBlock::iterator InsertPt =
            LastDep ? std::next(LastDep->getIterator())
                    : std::next(MI.getIterator());
        if (LastDep)
          LLVM_DEBUG(dbgs() << "  VOR_Orig depends on a value defined "
                                "between it and the VID, moving up only to "
                                "just after: " << *LastDep);
        if (InsertPt == VOR_Orig->getIterator()) {
          LLVM_DEBUG(dbgs() << "  VOR_Orig is already right after its last "
                                "dependency, nothing to move\n");
        } else {
          MBB.splice(InsertPt, &MBB, VOR_Orig->getIterator());
        }
        for (MachineInstr *UseMI : Uses) {
          MachineOperand &ImmReg = UseMI->getOperand(3);
          MachineInstr *ImmDef = MRI->getUniqueVRegDef(ImmReg.getReg());
          if (ImmDef && ImmDef->getOpcode() == RISCV::ADDI) {
            int64_t ImmValue = ImmDef->getOperand(2).getImm();
            LLVM_DEBUG(dbgs() <<"  VADD is Imm: " <<*UseMI);
            MachineInstr *VADD = UseMI;
            MachineInstr *VOR_ForVADD = nullptr;
            Register VADDDest = VADD->getOperand(0).getReg();
            for (MachineInstr &DestMI : MRI->use_instructions(VADDDest)) {
              if (DestMI.getOpcode() == *VOROpc) {
                MachineOperand &SrcReg = DestMI.getOperand(2);
                if (SrcReg.isReg() && SrcReg.getReg() == VADDDest) {
                  LLVM_DEBUG(dbgs() <<"  VOR_ForVADD: " <<DestMI);
                  VOR_ForVADD = &DestMI;
                  break;
                }
              }
            }
            if (!VOR_ForVADD) continue;
            Register BaseReg = VOR_Orig->getOperand(0).getReg();
            MachineInstr *NewVADD = nullptr;
            auto CreateNewVADD = [&](MachineInstr *OldVOR, MachineInstr *OldVADD) -> MachineInstr * {
              MachineInstrBuilder MIB = BuildMI(MBB, *OldVOR, OldVOR->getDebugLoc(),
                                                TII->get(*VADDOpc));
              MIB.addReg(OldVOR->getOperand(0).getReg(), RegState::Define);
              MIB.addReg(OldVOR->getOperand(1).getReg(), RegState::Undef);
              MIB.addReg(BaseReg);
              MIB.addReg(OldVADD->getOperand(3).getReg());
              // VL, SEW and policy of the VOR it replaces.
              for (unsigned I = 4, E = OldVOR->getNumExplicitOperands(); I != E;
                   ++I)
                MIB.add(OldVOR->getOperand(I));
              MIB.copyImplicitOps(*OldVOR);
              return MIB;
            };
            NewVADD = CreateNewVADD(VOR_ForVADD, VADD);
            Register VORDest = VOR_ForVADD->getOperand(0).getReg();
            for (auto &DestMI : MRI->use_instructions(VORDest)) {
              if (DestMI.getOpcode() == *VMSLTOpc) {
                MachineOperand &SrcReg = DestMI.getOperand(1);
                if (SrcReg.isReg() && SrcReg.getReg() == VORDest) {
                  LLVM_DEBUG(dbgs() <<"  VMSLT: " <<DestMI);
                  DestMI.getOperand(1).setReg(NewVADD->getOperand(0).getReg());
                  break;
                }
              }
            }
            VOR_ForVADD->eraseFromParent();
            VADD->eraseFromParent();
            LLVM_DEBUG(dbgs() << "  Replace VMSLT with NewVADD.\n");
          }
        }
        LLVM_DEBUG(dbgs() << "Transformed pattern from A to B.\n");
      }
    }
  }
}

// Two memory accesses are known not to alias if their underlying objects are
// distinct identified objects (e.g. different globals).
static bool isKnownNoAlias(const MachineInstr &A, const MachineInstr &B) {
  if (A.memoperands_empty() || B.memoperands_empty())
    return false;
  for (const MachineMemOperand *MA : A.memoperands())
    for (const MachineMemOperand *MB : B.memoperands()) {
      const Value *VA = MA->getValue(), *VB = MB->getValue();
      if (!VA || !VB)
        return false;
      const Value *OA = getUnderlyingObject(VA);
      const Value *OB = getUnderlyingObject(VB);
      if (OA == OB || !isIdentifiedObject(OA) || !isIdentifiedObject(OB))
        return false;
    }
  return true;
}

// Turn
//   %x = VL4RE32_V %p                           ; full load
//   ... (no store that may alias %p)
//   %x = PseudoVLE32_V_M4_MASK %x, %p, $v0, ... ; masked reload, identity on %x
//   ... uses of %x
// into a fresh full load of the same address defining a new vreg, so %x has a
// hole between its last use before the reload and the reload itself.
bool ExpandPseudos::ProcessRedundantReload(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      StringRef Name = TII->getName(MI.getOpcode());
      if (!Name.starts_with("PseudoVLE") || !Name.ends_with("_MASK"))
        continue;
      if (MI.getNumExplicitDefs() != 1 || MI.getNumOperands() < 3 ||
          MI.memoperands_empty())
        continue;
      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &Passthru = MI.getOperand(1);
      const MachineOperand &Base = MI.getOperand(2);
      if (!DstReg.isVirtual() || !Passthru.isReg() ||
          Passthru.getReg() != DstReg || !Base.isReg() ||
          !Base.getReg().isVirtual())
        continue;

      // %x must have exactly two defs: the earlier full load and this one.
      if (range_size(MRI->def_instructions(DstReg)) != 2)
        continue;
      MachineInstr *FullLoad = nullptr;
      for (MachineInstr &Def : MRI->def_instructions(DstReg))
        if (&Def != &MI)
          FullLoad = &Def;
      if (!FullLoad || FullLoad->getParent() != &MBB ||
          !isWholeRegLoadName(TII->getName(FullLoad->getOpcode())) ||
          FullLoad->getNumExplicitOperands() < 2 ||
          !FullLoad->getOperand(1).isReg() ||
          FullLoad->getOperand(1).getReg() != Base.getReg() ||
          FullLoad->memoperands_empty())
        continue;
      const MachineMemOperand *M0 = *FullLoad->memoperands_begin();
      const MachineMemOperand *M1 = *MI.memoperands_begin();
      if (M0->getValue() != M1->getValue() || M0->getOffset() != M1->getOffset())
        continue;

      // FullLoad must precede MI, and nothing in between may clobber memory.
      bool Ok = true, Seen = false;
      for (auto It = std::next(FullLoad->getIterator()); It != MI.getIterator();
           ++It) {
        if (It == MBB.end()) {
          Ok = false;
          break;
        }
        Seen = true;
        if (It->isDebugInstr())
          continue;
        if (It->isCall() || It->hasUnmodeledSideEffects() ||
            (It->mayStore() && !isKnownNoAlias(*It, *FullLoad))) {
          Ok = false;
          break;
        }
      }
      if (!Ok || !Seen)
        continue;

      // All uses of %x must be in this block; those after MI see the reload.
      SmallVector<MachineOperand *, 4> UsesAfter;
      bool AfterMI = false;
      for (MachineInstr &I : MBB) {
        if (&I == &MI) {
          AfterMI = true;
          continue;
        }
        if (!AfterMI)
          continue;
        for (MachineOperand &MO : I.operands())
          if (MO.isReg() && MO.getReg() == DstReg && MO.isUse())
            UsesAfter.push_back(&MO);
      }
      unsigned NumUses = range_size(MRI->use_operands(DstReg));
      // +1 for MI's own passthru use, which goes away.
      unsigned Before = NumUses - UsesAfter.size() - 1;
      for (MachineOperand &MO : MRI->use_operands(DstReg))
        if (MO.getParent()->getParent() != &MBB) {
          Ok = false;
          break;
        }
      if (!Ok)
        continue;
      (void)Before;

      LLVM_DEBUG(dbgs() << "Redundant reload: " << MI);
      Register NewReg = MRI->createVirtualRegister(MRI->getRegClass(DstReg));
      MachineInstr *NewLoad = MF.CloneMachineInstr(FullLoad);
      NewLoad->getOperand(0).setReg(NewReg);
      NewLoad->getOperand(0).setIsDead(false);
      NewLoad->getOperand(0).setIsDef();
      NewLoad->setDebugLoc(MI.getDebugLoc());
      MBB.insert(MI.getIterator(), NewLoad);
      RematRegs.insert(NewReg);
      for (MachineOperand *MO : UsesAfter) {
        MO->setReg(NewReg);
        MO->setIsKill(false);
      }
      RegsToClearKillFlags.insert(DstReg);
      RegsToClearKillFlags.insert(Base.getReg());
      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// Cheap instructions that may be recomputed instead of kept live: vector mask
// compares and immediate splats. They must be pure and define one register.
static bool isCheapALUOp(const MachineInstr &MI, StringRef Name) {
  if (!(Name.starts_with("PseudoVMS") || Name.starts_with("PseudoVMV_V_I")))
    return false;
  return !MI.mayLoadOrStore() && !MI.hasUnmodeledSideEffects() &&
         !MI.isCall() && MI.getNumExplicitDefs() == 1;
}

// Recompute a whole register load, or a cheap ALU op, right before a use that
// is far from the previous use, so the register is free in between:
//   %x = VL4RE32_V %p          (or %x = PseudoVMSLT_VV_M4 %a, %b)
//   use1 %x
//   use2 %x                    (loads need 2 uses here, ALU ops need 1)
//   ... >= RematGap instructions ...
//   other
//   use3 %x           -->   %y = <same instr> ; other ; use3 %y
// Every register the instruction reads must still be live at the new position,
// otherwise the recomputation would just extend another live range.
bool ExpandPseudos::ProcessRematLoads(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 8> Worklist;
    for (MachineInstr &MI : MBB) {
      StringRef Name = TII->getName(MI.getOpcode());
      if (isWholeRegLoadName(Name) || isCheapALUOp(MI, Name))
        Worklist.push_back(&MI);
    }

    while (!Worklist.empty()) {
      MachineInstr *Cand = Worklist.pop_back_val();
      bool IsLoad = Cand->mayLoad();
      if (Cand->getNumExplicitOperands() < 2 || !Cand->getOperand(0).isReg() ||
          (IsLoad && Cand->memoperands_empty()))
        continue;
      Register X = Cand->getOperand(0).getReg();
      if (!X.isVirtual() || !MRI->hasOneDef(X))
        continue;

      // Every register read must be virtual with a single def, so its value
      // is the same at the new position. Undef operands do not count.
      SmallVector<Register, 4> Inputs;
      bool InputsOk = true;
      for (const MachineOperand &MO : Cand->all_uses()) {
        if (!MO.isReg() || MO.isUndef() || MO.getReg() == X)
          continue;
        Register R = MO.getReg();
        if (!R.isVirtual() || !MRI->hasOneDef(R)) {
          InputsOk = false;
          break;
        }
        if (!is_contained(Inputs, R))
          Inputs.push_back(R);
      }
      if (!InputsOk || Inputs.empty())
        continue;

      // Collect the uses of X, requiring all of them to be in this block.
      SmallPtrSet<MachineInstr *, 8> UseSet;
      bool LocalOnly = true;
      for (MachineInstr &U : MRI->use_nodbg_instructions(X)) {
        if (U.getParent() != &MBB) {
          LocalOnly = false;
          break;
        }
        UseSet.insert(&U);
      }
      if (!LocalOnly || UseSet.size() < 2)
        continue;

      // Loads need the value to be reused before the gap, ALU ops are cheap.
      unsigned MinUses = IsLoad ? RematMinUses : 1;

      auto IsHazard = [&](const MachineInstr &I) {
        if (I.isCall() || I.hasUnmodeledSideEffects())
          return true;
        return IsLoad && I.mayStore() && !isKnownNoAlias(I, *Cand);
      };

      // Walk forward: look for the first use that is far from the previous.
      MachineInstr *PrevUse = nullptr;
      MachineInstr *Target = nullptr;
      unsigned Gap = 0, NumUses = 0;
      for (auto It = std::next(Cand->getIterator()); It != MBB.end(); ++It) {
        MachineInstr &I = *It;
        if (I.isDebugInstr())
          continue;
        if (UseSet.count(&I)) {
          if (NumUses >= MinUses && Gap >= RematGap) {
            Target = &I;
            break;
          }
          PrevUse = &I;
          ++NumUses;
          Gap = 0;
          continue;
        }
        if (!PrevUse)
          continue; // Instructions before the first use do not count as a gap.
        ++Gap;
      }
      if (!Target)
        continue;

      // Nothing between the original and the target may change memory.
      bool Hazard = false;
      for (auto It = std::next(Cand->getIterator());
           It != Target->getIterator(); ++It)
        if (!It->isDebugInstr() && IsHazard(*It)) {
          Hazard = true;
          break;
        }
      if (Hazard)
        continue;

      // Insert the clone as late as the stall gap to the use allows (directly
      // before it at m4 / m8), but after the previous use, so the value keeps
      // its hole.
      std::optional<MachineBasicBlock::iterator> GapPt = stallGapInsertPoint(
          *Target, PrevUse, minStallGap(*Cand, *Target, *MRI, *TRI, *TII),
          *MRI);
      if (!GapPt)
        continue;
      MachineBasicBlock::iterator InsertPt = *GapPt;

      // Only worth it if every input is live at the insertion point anyway.
      bool AllLive = true;
      for (Register R : Inputs) {
        bool Live = false;
        for (MachineOperand &MO : MRI->use_nodbg_operands(R)) {
          MachineInstr *U = MO.getParent();
          if (U == Cand)
            continue;
          if (U->getParent() != &MBB) {
            Live = true;
            break;
          }
          for (auto It = InsertPt; It != MBB.end(); ++It)
            if (&*It == U) {
              Live = true;
              break;
            }
          if (Live)
            break;
        }
        if (!Live) {
          AllLive = false;
          break;
        }
      }
      if (!AllLive) {
        LLVM_DEBUG(dbgs() << "Remat skipped, an input dies before target: "
                          << *Cand);
        continue;
      }

      LLVM_DEBUG(dbgs() << "Remat " << *Cand << "  before " << *InsertPt
                        << "  for use " << *Target);
      Register Y = MRI->createVirtualRegister(MRI->getRegClass(X));
      MachineInstr *NewMI = MF.CloneMachineInstr(Cand);
      for (MachineOperand &MO : NewMI->operands()) {
        if (!MO.isReg())
          continue;
        if (MO.getReg() == X) {
          // The def, and any tied undef use of it.
          MO.setReg(Y);
          if (MO.isDef())
            MO.setIsDead(false);
        } else if (MO.isUse()) {
          MO.setIsKill(false);
        }
      }
      NewMI->setDebugLoc(Target->getDebugLoc());
      MBB.insert(InsertPt, NewMI);
      RematRegs.insert(Y);

      // Rewrite Target and every later use of X.
      for (MachineInstr &I :
           make_range(MachineBasicBlock::iterator(Target), MBB.end()))
        for (MachineOperand &MO : I.operands())
          if (MO.isReg() && MO.isUse() && MO.getReg() == X) {
            MO.setReg(Y);
            MO.setIsKill(false);
          }
      RegsToClearKillFlags.insert(X);
      for (Register R : Inputs)
        RegsToClearKillFlags.insert(R);
      Changed = true;
      // The clone can be split again by a later distant use.
      Worklist.push_back(NewMI);
    }
  }
  return Changed;
}

// Reverse rematerialization (Bahi & Eisenbeis, "Register Reverse
// Rematerialization" / "Impact of Reverse Computing on Information Locality
// in Register Allocation for High Performance Computing"): recompute a value
// from something derived from it, instead of keeping it alive.
//
// Terms from the paper, applied to one basic block (its DDG):
//  - output(v): the instructions using v. Every use q of v of the form
//      %Y = OP(v, %K)  or  %Y = OP(%K, v)
//    with an inverse gives a way to recompute v as REV(%Y, %K), so the set of
//    operand sets v can be recomputed from, R-input(v), has one entry {Y, K}
//    per reversible use of v, not just the one at a particular instruction.
//  - v is reversibly rematerializable at a point P iff some {Y, K} in
//    R-input(v) has all of its members live at P.
//  - v is rematerializable by multiple instructions iff the members are
//    live *or rematerializable*. A value recomputed earlier is the same value
//    as its original, so the members are searched among the original and all
//    of its copies; recomputing a whole chain (Figure 2(c): D from E, C from
//    D, B from C, A from B) then comes from repeating the transform, each
//    step anchored on the copy the previous one made.
//  - The algorithm is iterative: transform, recompute, repeat until no value
//    is rematerializable any more.
//
// When to rematerialize is not the paper's excessive-point search but the
// timing of -custom-remat (a use at least RematGap instructions after the
// previous one), and how is chosen at that site among the options recorded on
// the value's live range, as V8 does; see ProcessReverseRematChain.
//
// The paper ignores precision ("we make the abstract approximation that the
// operations are reversible"); here integer ADD/SUB/RSUB/XOR are exact under
// wraparound, while the float ops (FADD/FSUB/FMUL/FDIV) round like the paper
// accepts. Integer MUL and shifts lose bits and are not reversed.

namespace {
// v at operand Vs (2 = vs2, 3 = vs1/rs1/imm) of Y = Mn(vs2, vs1) is
// recomputed as v = RevMn(Y, K), or RevMn(K, Y) when KFirst.
struct ReversePattern {
  const char *Mn;
  const char *RevMn;
  unsigned Vs;
  bool KFirst;
};
} // namespace

static const ReversePattern ReversePatterns[] = {
    {"ADD", "SUB", 2, false},     {"ADD", "SUB", 3, false},   // v = Y - K
    {"SUB", "ADD", 2, false},                                 // v = Y + K
    {"SUB", "SUB", 3, true},                                  // v = K - Y
    {"RSUB", "RSUB", 2, false},                               // v = K - Y
    {"XOR", "XOR", 2, false},     {"XOR", "XOR", 3, false},   // v = Y ^ K
    {"FADD", "FSUB", 2, false},   {"FADD", "FSUB", 3, false}, // v = Y - K
    {"FSUB", "FADD", 2, false},                               // v = Y + K
    {"FSUB", "FSUB", 3, true},                                // v = K - Y
    {"FRSUB", "FRSUB", 2, false},                             // v = K - Y
    {"FMUL", "FDIV", 2, false},   {"FMUL", "FDIV", 3, false}, // v = Y / K
    {"FDIV", "FMUL", 2, false},                               // v = Y * K
    {"FDIV", "FDIV", 3, true},                                // v = K / Y
    {"FRDIV", "FRDIV", 2, false},                             // v = K / Y
};

namespace {
struct ReverseOp {
  unsigned Opcode;
  bool KFirst;
  bool NegateImm; // ADD_VI is reversed as ADD_VI with -imm (there is no SUB_VI)
};
} // namespace

// The opcode of the RVV pseudo called Name, if there is one.
static std::optional<unsigned> findPseudo(const TargetInstrInfo *TII,
                                          const Twine &Name) {
  static const StringMap<unsigned> ByName = [TII] {
    StringMap<unsigned> M;
    for (unsigned Opc = 0, E = TII->getNumOpcodes(); Opc != E; ++Opc) {
      StringRef Name = TII->getName(Opc);
      if (Name.starts_with("PseudoV"))
        M[Name] = Opc;
    }
    return M;
  }();
  auto It = ByName.find(Name.str());
  if (It == ByName.end())
    return std::nullopt;
  return It->second;
}

// If v at operand Vs of Q can be recomputed from Q's result and its other
// source, the opcode doing it. Q must be an unmasked
// PseudoV<MN>_<VV|VX|VI|VFPR*>_<LMUL>[_E<SEW>] with an undef passthru.
static std::optional<ReverseOp> matchReverseOp(const TargetInstrInfo *TII,
                                               const MachineInstr &Q,
                                               unsigned Vs) {
  StringRef Name = TII->getName(Q.getOpcode());
  if (!Name.consume_front("PseudoV") || Name.contains("MASK") ||
      Name.contains("TIED"))
    return std::nullopt;
  auto [Mn, Rest] = Name.split('_');
  auto [Form, Tail] = Rest.split('_');
  if (Tail.empty() ||
      !(Form == "VV" || Form == "VX" || Form == "VI" || Form.starts_with("VFPR")))
    return std::nullopt;
  // Only a vector-vector op has a vector to reverse in its second source.
  if (Vs == 3 && Form != "VV")
    return std::nullopt;

  const MCInstrDesc &Desc = Q.getDesc();
  if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags) ||
      Q.getNumExplicitDefs() != 1 || Q.getNumExplicitOperands() < 6 ||
      !Q.getOperand(0).isReg() || !Q.getOperand(1).isReg() ||
      !(Q.getOperand(1).isUndef() || !Q.getOperand(1).getReg()) ||
      !Q.getOperand(2).isReg() ||
      !(Q.getOperand(3).isReg() || Q.getOperand(3).isImm()))
    return std::nullopt;

  for (const ReversePattern &P : ReversePatterns) {
    if (Mn != P.Mn || Vs != P.Vs)
      continue;
    StringRef RevMn = P.RevMn;
    bool NegateImm = false;
    if (Form == "VI" && RevMn == "SUB") {
      // Y = v + imm  =>  v = Y + (-imm), if -imm is still a simm5.
      if (Q.getOperand(3).getImm() == -16)
        return std::nullopt;
      RevMn = "ADD";
      NegateImm = true;
    }
    std::optional<unsigned> Opc =
        findPseudo(TII, "PseudoV" + RevMn + "_" + Form + "_" + Tail);
    if (!Opc || TII->get(*Opc).getNumOperands() != Desc.getNumOperands())
      return std::nullopt;
    return ReverseOp{*Opc, P.KFirst, NegateImm};
  }
  return std::nullopt;
}

// If D can be recomputed elsewhere as a clone of itself, with its vector
// source (operand 2) and its second source (operand 3) taken from equal
// values: the paper's direct rematerialization, v = OP(input(v)). D must be
// an unmasked RVV pseudo with an undef passthru that only computes, and its
// other operands (VL, SEW, policy, ...) must not be vector registers.
static std::optional<ReverseOp> matchDirectOp(const TargetInstrInfo *TII,
                                              const MachineInstr &D,
                                              const MachineRegisterInfo &MRI) {
  StringRef Name = TII->getName(D.getOpcode());
  if (!Name.starts_with("PseudoV") || Name.contains("MASK") ||
      Name.contains("TIED"))
    return std::nullopt;
  const MCInstrDesc &Desc = D.getDesc();
  if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags) ||
      D.getNumExplicitDefs() != 1 || D.getNumExplicitOperands() < 6 ||
      D.mayLoadOrStore() || D.hasUnmodeledSideEffects() || D.isCall() ||
      !D.getOperand(0).isReg() || !D.getOperand(1).isReg() ||
      !(D.getOperand(1).isUndef() || !D.getOperand(1).getReg()) ||
      !D.getOperand(2).isReg() || !D.getOperand(2).getReg().isVirtual() ||
      !isVectorReg(D.getOperand(2).getReg(), MRI) ||
      !(D.getOperand(3).isReg() || D.getOperand(3).isImm()))
    return std::nullopt;
  for (unsigned I = 4, E = D.getNumExplicitOperands(); I != E; ++I)
    if (D.getOperand(I).isReg() && D.getOperand(I).getReg() &&
        isVectorReg(D.getOperand(I).getReg(), MRI))
      return std::nullopt;
  return ReverseOp{D.getOpcode(), false, false};
}

// The value of a VL or scalar operand known at compile time: an immediate, or
// a register set by `ADDI $x0, imm` (li).
static std::optional<int64_t> constOperand(const MachineOperand &MO,
                                           const MachineRegisterInfo &MRI) {
  if (MO.isImm())
    return MO.getImm();
  if (!MO.isReg())
    return std::nullopt;
  if (MO.getReg() == RISCV::X0)
    return 0;
  if (!MO.getReg().isVirtual())
    return std::nullopt;
  const MachineInstr *D = MRI.getUniqueVRegDef(MO.getReg());
  if (D && D->getOpcode() == RISCV::ADDI && D->getOperand(1).isReg() &&
      D->getOperand(1).getReg() == RISCV::X0 && D->getOperand(2).isImm())
    return D->getOperand(2).getImm();
  return std::nullopt;
}

// A VL operand known to be a constant element count (VLMAX, -1, is not).
static std::optional<int64_t> constVL(const MachineOperand &MO,
                                      const MachineRegisterInfo &MRI) {
  std::optional<int64_t> C = constOperand(MO, MRI);
  if (!C || *C < 0)
    return std::nullopt;
  return C;
}

namespace {
// The elements of a recomputed value that hold its data: the first VL
// elements of Bits bits each. The rebuild writes them with an instruction of
// SEW Log2SEW and LMUL LMul (which sizes VL = VLMAX); its tail is agnostic.
// Bits differs from the SEW when the rebuild changes the element width, e.g.
// the zip that rebuilds a deinterleaved pair writes 2*SEW bit elements.
struct ElemRange {
  MachineOperand VL = MachineOperand::CreateImm(0);
  unsigned Bits = 0;
  unsigned Log2SEW = 0;
  RISCVVType::VLMUL LMul = RISCVVType::LMUL_1;
};
} // namespace

static MachineOperand copyUse(const MachineOperand &MO) {
  if (MO.isReg())
    return MachineOperand::CreateReg(MO.getReg(), false);
  return MO;
}

static ElemRange makeRange(const MachineOperand &VL, unsigned Log2SEW,
                           const MCInstrDesc &Writer, unsigned Bits) {
  return {copyUse(VL), Bits, Log2SEW, RISCVII::getLMul(Writer.TSFlags)};
}

// Bits of each element that operand OpIdx of the RVV pseudo PseudoV<Name>
// reads, for an instruction with SEW bits per element.
static unsigned elementBitsRead(StringRef Name, unsigned OpIdx, unsigned SEW) {
  auto [Mn, Rest] = Name.split('_');
  // The accumulator of a widening multiply-add is wide.
  if (OpIdx == 1 && (Mn.starts_with("W") || Mn.starts_with("FW")))
    return 2 * SEW;
  if (OpIdx != 2)
    return SEW;
  // vsext.vf<n> / vzext.vf<n>: SEW is the destination width.
  if (Mn.ends_with("EXT") && Rest.starts_with("VF") && Rest.size() > 2 &&
      isDigit(Rest[2]))
    return SEW / (Rest[2] - '0');
  // The wide source of a narrowing op (vnsrl.w*, vfncvt.*.w) or of a
  // vwadd.w* style op.
  if (Rest.starts_with("W") || Mn.contains("NCVT"))
    return 2 * SEW;
  return SEW;
}

// Whether U reads R only within the elements R holds data in after a rebuild
// (ER), i.e. reads a prefix of R no longer than ER's. Element widths may
// differ: a use reading R as 2*VL elements of 32 bits is fine for a rebuild
// writing VL elements of 64 bits. Conservative: any use that may read past
// VL (slides, gathers, segments, ...), reads R as a mask or as a sub-register,
// or keeps R's tail undisturbed is rejected.
static bool readsOnlyPrefix(const TargetInstrInfo *TII,
                            const MachineRegisterInfo &MRI,
                            const MachineInstr &U, Register R,
                            const ElemRange &ER) {
  StringRef Name = TII->getName(U.getOpcode());
  if (!Name.consume_front("PseudoV"))
    return false;
  static const char *Unsafe[] = {
      "SLIDE", "RGATHER", "COMPRESS", "XEI",  "SEG",  "WRED", "_MM",
      "_M_B",  "CPOP",    "FIRST",    "IOTA", "MSBF", "MSIF", "MSOF",
  };
  for (const char *S : Unsafe)
    if (Name.contains(S))
      return false;
  const MCInstrDesc &UD = U.getDesc();
  if (!RISCVII::hasVLOp(UD.TSFlags) || !RISCVII::hasSEWOp(UD.TSFlags))
    return false;
  unsigned Log2SEW = U.getOperand(RISCVII::getSEWOpNum(UD)).getImm();
  if (!Log2SEW) // a mask instruction
    return false;
  bool TailAgnostic = false;
  if (RISCVII::hasVecPolicyOp(UD.TSFlags)) {
    int64_t Policy = U.getOperand(RISCVII::getVecPolicyOpNum(UD)).getImm();
    TailAgnostic = (Policy & RISCVVType::TAIL_AGNOSTIC) &&
                   (!Name.contains("MASK") ||
                    (Policy & RISCVVType::MASK_AGNOSTIC));
  }
  unsigned Bits = 0;
  for (unsigned I = 0, E = U.getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = U.getOperand(I);
    if (!MO.isReg() || MO.getReg() != R)
      continue;
    if (MO.isDef() || MO.getSubReg() || MO.isImplicit() ||
        (MO.isTied() && !TailAgnostic))
      return false;
    Bits = std::max(Bits, elementBitsRead(Name, I, 1u << Log2SEW));
  }
  if (!Bits)
    return false;
  const MachineOperand &UVL = U.getOperand(RISCVII::getVLOpNum(UD));
  std::optional<int64_t> CU = constVL(UVL, MRI), CR = constVL(ER.VL, MRI);
  if (CU && CR)
    return uint64_t(*CU) * Bits <= uint64_t(*CR) * ER.Bits;
  // Both VLMAX: the same count only at the same SEW and LMUL.
  if (UVL.isImm() && ER.VL.isImm() && UVL.getImm() == ER.VL.getImm())
    return Log2SEW == ER.Log2SEW && RISCVII::getLMul(UD.TSFlags) == ER.LMul &&
           Bits == ER.Bits;
  // The same AVL register: the same element count.
  return UVL.isReg() && ER.VL.isReg() && UVL.getReg() == ER.VL.getReg() &&
         Bits == ER.Bits;
}

namespace {
// Positions of the references of every virtual register in a block and the
// vector register pressure after each instruction, rebuilt after every
// transform.
struct ReverseBlockModel {
  static constexpr unsigned Inf = ~0u;
  struct RegRefs {
    SmallVector<unsigned, 4> Reads; // positions reading the register, sorted
    SmallVector<unsigned, 4> Defs;  // positions defining it, sorted
    bool LiveOut = false; // live at the end of the block
    bool HasSubReg = false;    // referenced through a sub-register
    bool HasSubRegDef = false; // defined through a sub-register
    // Last position the register is needed at: Inf if live out.
    unsigned last() const {
      return LiveOut ? Inf : (Reads.empty() ? 0 : Reads.back());
    }
    // Whether some def lies in [From, To).
    bool defIn(unsigned From, unsigned To) const {
      for (unsigned D : Defs)
        if (D >= From && D < To)
          return true;
      return false;
    }
  };
  SmallVector<MachineInstr *, 64> Instrs; // non-debug instructions
  DenseMap<const MachineInstr *, unsigned> Pos;
  DenseMap<Register, RegRefs> Regs;
  SmallVector<Register, 64> Order; // registers by first reference
  SmallVector<unsigned, 64> Pressure; // VR pressure after each instruction

  void build(MachineBasicBlock &MBB, MachineRegisterInfo &MRI,
             const TargetRegisterInfo &TRI, const DenseSet<Register> &LiveOuts) {
    Instrs.clear();
    Pos.clear();
    Regs.clear();
    Order.clear();
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      unsigned P = Instrs.size();
      Pos[&MI] = P;
      Instrs.push_back(&MI);
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        auto [It, Inserted] = Regs.try_emplace(MO.getReg());
        if (Inserted)
          Order.push_back(MO.getReg());
        RegRefs &RR = It->second;
        if (MO.getSubReg())
          RR.HasSubReg = true;
        if (MO.getSubReg() && MO.isDef())
          RR.HasSubRegDef = true;
        SmallVectorImpl<unsigned> &L = MO.isDef() ? RR.Defs : RR.Reads;
        if (MO.isUse() && MO.isUndef())
          continue;
        if (L.empty() || L.back() != P)
          L.push_back(P);
      }
    }
    unsigned N = Instrs.size();
    SmallVector<int, 64> Delta(N + 1, 0);
    for (auto &KV : Regs) {
      Register R = KV.first;
      RegRefs &RR = KV.second;
      // Read before (or by) its first def here: live in. Liveness is only
      // used for profitability: the transform never changes which value a
      // use reads.
      bool LiveIn = RR.Defs.empty() ||
                    (!RR.Reads.empty() && RR.Reads.front() <= RR.Defs.front());
      RR.LiveOut = LiveOuts.count(R);

      if (!isVectorReg(R, MRI))
        continue;
      unsigned Start = LiveIn ? 0 : RR.Defs.front();
      unsigned End = RR.LiveOut
                         ? N
                         : std::max(RR.Reads.empty() ? 0 : RR.Reads.back(),
                                    RR.Defs.empty() ? 0 : RR.Defs.back());
      if (Start < End) {
        int W = TRI.getRegClassWeight(MRI.getRegClass(R)).RegWeight;
        Delta[Start] += W;
        Delta[End] -= W;
      }
    }
    Pressure.assign(N, 0);
    int Running = 0;
    for (unsigned I = 0; I != N; ++I) {
      Running += Delta[I];
      Pressure[I] = Running;
    }
  }
};
} // namespace

// The virtual registers live at the end of each block: a backward dataflow
// over the CFG. A loop-invariant value read inside a loop is live around the
// back edge, so killing it early in the loop body frees nothing.
static DenseMap<const MachineBasicBlock *, DenseSet<Register>>
computeVRegLiveOuts(MachineFunction &MF) {
  DenseMap<const MachineBasicBlock *, DenseSet<Register>> UpwardUses, Defs,
      LiveIns, LiveOuts;
  for (MachineBasicBlock &MBB : MF) {
    DenseSet<Register> &Up = UpwardUses[&MBB], &D = Defs[&MBB];
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      for (const MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.getReg().isVirtual() && MO.readsReg() &&
            !D.count(MO.getReg()))
          Up.insert(MO.getReg());
      for (const MachineOperand &MO : MI.all_defs())
        if (MO.getReg().isVirtual())
          D.insert(MO.getReg());
    }
    LiveIns[&MBB] = Up;
  }
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock *MBB : post_order(&MF)) {
      DenseSet<Register> &Out = LiveOuts[MBB];
      for (MachineBasicBlock *Succ : MBB->successors())
        for (Register R : LiveIns[Succ])
          Out.insert(R);
      DenseSet<Register> &In = LiveIns[MBB];
      const DenseSet<Register> &D = Defs[MBB];
      for (Register R : Out)
        if (!D.count(R) && In.insert(R).second)
          Changed = true;
    }
  }
  return LiveOuts;
}

// Whether physical register R is redefined by an instruction in (From, To).
static bool physRegModifiedIn(const ReverseBlockModel &BM, MCRegister R,
                              unsigned From, unsigned To,
                              const TargetRegisterInfo *TRI) {
  for (unsigned I = From + 1; I < To; ++I)
    if (BM.Instrs[I]->modifiesRegister(R, TRI))
      return true;
  return false;
}

// Non-elementwise reverse rematerialization.
//
// An elementwise inverse (v = Y - K from Y = v + K) is what any reverse
// rematerializer does. Vector code also moves data across lanes, element
// widths and register groups, and LLVM lowers those moves to *groups* of RVV
// instructions none of which is invertible on its own:
//
//   rotate lanes by k   vslidedown.vi r, v, k ; vslideup.vi r, v, n-k
//                       (each slide drops k or n-k lanes)
//   swap lane pairs     vsrl.vx a, v, 32 ; vsll.vx b, v, 32 ; vor.vv y, a, b
//   (e32 viewed as e64)  (each shift drops half the bits, OR merges)
//   interleave a, b     vwaddu.vv z, a, b ; vwmaccu.vx z, -1, b
//                       (z = a + b + b * (2^SEW - 1) = a | b << SEW)
//   deinterleave w      vnsrl.wi lo, w, 0 ; vnsrl.wx hi, w, SEW
//                       (each narrowing drops half of w)
//   widen               vsext.vf2 / vzext.vf2 / vfwcvt.f.f.v / vfwcvt.f.x.v
//                       (changes SEW and LMUL; its inverse narrows)
//
// The group as a whole is a bijection, so v can be rebuilt exactly from the
// group's result(s), in the paper's terms R-input(v) gets an entry whose
// members are the outputs of several instructions (the deinterleave: {lo, hi}
// together, neither alone), and the rebuilt value may live in a register
// group of another size or element width than its source (the widen: v in
// LMUL 2 rebuilt from y in LMUL 4). All of these rebuilds are exact, unlike
// the rounded float inverses.

namespace {
// One way of rebuilding v at a later point.
struct RevPlan {
  enum KindTy {
    Elementwise, // v = REV(Y, K) from Y = OP(v, K)
    Direct,      // v = OP(Y, K) again, Y and K being v's own inputs
    Narrow,      // v = narrow(Y) from Y = widen(v)
    ZipHalf,     // v = vnsrl(Z, 0 | SEW) from Z = zip(v, b) / zip(a, v)
    Unzip,       // v = zip(Lo, Hi) from Lo, Hi = vnsrl(v, 0), vnsrl(v, SEW)
    Rotate,      // v = rotate(Y, n - k) from Y = rotate(v, k)
    BitRotate,   // v = rotl(Y, s) from Y = rotr(v, s) = v >> s | v << SEW-s
    Flip,        // v = flip(Y) from Y = flip(v): the lane reverse again
    PermGather,  // v = gather(Y, inverse(p)) from Y = gather(v, p)
  };
  KindTy Kind = Elementwise;
  // The instructions computing the group's result from v (Fwd[0] is the use
  // q of v the plan was found from). Their scalar operands (VL, shift and
  // slide amounts, ...) are reused by the rebuild.
  SmallVector<MachineInstr *, 3> Fwd;
  // The values the rebuild reads; each must keep its value from position
  // From on. Slots are the (opcode, operand) pairs it is read at, for the
  // register class checks.
  struct Input {
    Register Reg;
    unsigned From = 0;
    SmallVector<std::pair<unsigned, unsigned>, 2> Slots;
  };
  SmallVector<Input, 2> Ins;
  ReverseOp Op{};      // Elementwise / Direct
  unsigned QK = 0;     // Elementwise / Direct: the index of K in q
  bool KIsReg = false; // Elementwise / Direct
  // Rebuild opcodes of the other kinds, in emission order; the last one
  // writes v.
  SmallVector<unsigned, 3> Opcs;
  unsigned Half = 0;   // ZipHalf: 0 = even (low) elements, 1 = odd (high)
  ElemRange Valid;     // what the rebuild leaves valid (not for Direct)
  unsigned TempW = 0;  // pressure of the rebuild's temporaries
  bool Exact = true;   // bit-identical to the original value
  // Flip: the instructions to replay on Y, in program order (vid, vrsub,
  // the gathers, the slide), v as their source and the index vector.
  SmallVector<MachineInstr *, 12> Clone;
  Register FlipSrc, FlipIdx; // also PermGather
  Constant *PermInv = nullptr; // PermGather: the inverse permutation
  Align PermAlign;
  // PermGather of a self-inverse permutation (p = inverse(p), e.g. a swap of
  // blocks): the rebuild is the gather again with Q's own index, Ins[1].
  bool PermReuseIdx = false;
  unsigned dstOpcode() const { return Opcs.empty() ? Op.Opcode : Opcs.back(); }
  // Instructions the rebuild adds, roughly: a direct chain replays its
  // links, the others emit their opcodes and replay their clones.
  unsigned numNew() const {
    if (Kind == Direct)
      return Clone.size();
    return std::max<size_t>(Opcs.size(), 1) + Clone.size();
  }
};
} // namespace

static const char *planName(RevPlan::KindTy K) {
  switch (K) {
  case RevPlan::Elementwise: return "elementwise";
  case RevPlan::Direct: return "direct";
  case RevPlan::Narrow: return "narrow-of-widen";
  case RevPlan::ZipHalf: return "deinterleave-of-zip";
  case RevPlan::Unzip: return "zip-of-deinterleave";
  case RevPlan::Rotate: return "inverse-slide-rotation";
  case RevPlan::BitRotate: return "inverse-bit-rotation";
  case RevPlan::Flip: return "flip-of-flip";
  case RevPlan::PermGather: return "inverse-permutation-gather";
  }
  return "?";
}

// The LMUL part of an RVV pseudo name and whatever follows it: "M4" for
// PseudoVSEXT_VF2_M4, "M2_E32" for PseudoVFWCVT_F_F_V_M2_E32.
static StringRef lmulSuffix(StringRef Name) {
  for (size_t P = Name.find("_M"); P != StringRef::npos;
       P = Name.find("_M", P + 2)) {
    StringRef S = Name.substr(P + 2);
    if (!S.empty() &&
        (isDigit(S[0]) || (S[0] == 'F' && S.size() > 1 && isDigit(S[1]))))
      return Name.substr(P + 1);
  }
  return "";
}

// The LMUL name of half of LMUL L ("M4" -> "M2", "M1" -> "MF2").
static std::string halfLMul(StringRef L) {
  static const char *Order[] = {"MF8", "MF4", "MF2", "M1", "M2", "M4", "M8"};
  for (unsigned I = 1; I != std::size(Order); ++I)
    if (L == Order[I])
      return Order[I - 1];
  return "";
}

// An unmasked RVV pseudo with a VL and SEW operand defining one full virtual
// register, followed by a passthru and a register source.
static bool isPlainRVV(const TargetInstrInfo *TII, const MachineInstr &MI,
                       StringRef &Name) {
  Name = TII->getName(MI.getOpcode());
  if (!Name.starts_with("PseudoV") || Name.contains("MASK") ||
      Name.contains("TIED"))
    return false;
  const MCInstrDesc &D = MI.getDesc();
  return RISCVII::hasVLOp(D.TSFlags) && RISCVII::hasSEWOp(D.TSFlags) &&
         MI.getNumExplicitDefs() == 1 && MI.getNumExplicitOperands() >= 5 &&
         MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isVirtual() &&
         !MI.getOperand(0).getSubReg() && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isReg() && !MI.getOperand(2).getSubReg();
}

static bool hasUndefPassthru(const MachineInstr &MI) {
  return MI.getOperand(1).isUndef() || !MI.getOperand(1).getReg();
}

static const MachineOperand &vlOperand(const MachineInstr &MI) {
  return MI.getOperand(RISCVII::getVLOpNum(MI.getDesc()));
}

static unsigned log2SEWOf(const MachineInstr &MI) {
  return MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();
}

// Whether A and B run with the same VL operand and SEW.
static bool sameVLAndSEW(const MachineInstr &A, const MachineInstr &B) {
  const MachineOperand &VA = vlOperand(A), &VB = vlOperand(B);
  bool SameVL = VA.isImm() ? VB.isImm() && VA.getImm() == VB.getImm()
                           : VB.isReg() && VA.getReg() == VB.getReg();
  return SameVL && log2SEWOf(A) == log2SEWOf(B);
}

// "VI_" for an immediate amount, "VX_" for a register.
static const char *amountForm(const MachineOperand &MO) {
  return MO.isImm() ? "VI_" : "VX_";
}

// If R holds vlenb * 2^E (PseudoReadVLENB scaled by SLLI / SRLI), E. vlenb is
// a power of two of at least 16 bytes, so a right shift by up to 3 is exact.
static std::optional<int> vlenbScale(Register R, const MachineRegisterInfo &MRI) {
  int E = 0;
  while (R.isVirtual()) {
    const MachineInstr *D = MRI.getUniqueVRegDef(R);
    if (!D)
      return std::nullopt;
    if (D->getOpcode() == RISCV::PseudoReadVLENB)
      return E;
    if ((D->getOpcode() != RISCV::SLLI && D->getOpcode() != RISCV::SRLI) ||
        !D->getOperand(1).isReg() || !D->getOperand(2).isImm())
      return std::nullopt;
    int64_t Sh = D->getOperand(2).getImm();
    if (D->getOpcode() == RISCV::SRLI && Sh > 3)
      return std::nullopt;
    E += D->getOpcode() == RISCV::SLLI ? Sh : -Sh;
    R = D->getOperand(1).getReg();
  }
  return std::nullopt;
}

// Whether R is set by `ADDI Z, -Sub` with Z = vlenb * 2^E.
static bool isVlenbMinus(Register R, int E, int64_t Sub,
                         const MachineRegisterInfo &MRI) {
  if (!R.isVirtual())
    return false;
  const MachineInstr *D = MRI.getUniqueVRegDef(R);
  return D && D->getOpcode() == RISCV::ADDI && D->getOperand(1).isReg() &&
         D->getOperand(2).isImm() && D->getOperand(2).getImm() == -Sub &&
         vlenbScale(D->getOperand(1).getReg(), MRI) == E;
}

// Lane reverse (tl.flip, a <n-1,...,1,0> shuffle). LLVM lowers it as
//   idx = vrsub(vid.v, c)                         ; c = last lane index
//   T.sub_k = vrgather.vv(v.sub_(L-1-k), idx)     ; k = 0..L-1, VLMAX each
//   y = vslidedown.vx(T, VLMAX - n)               ; VL n
// which reverses each register, reverses the register order (all of
// VLMAX lanes) and slides the n lanes that hold v to the bottom: y[i] =
// v[n-1-i] for i < n. When n is a whole LMUL1 register at the minimum VLEN
// it is one gather with c = n - 1 and VL n, and no slide. Either way the
// group applied to y gives v back, reading only y[0, n): v = flip(y).
// Q is the first of the gathers, reading VIn (v or a copy of v).
static void matchFlipPlan(const TargetInstrInfo *TII,
                          const TargetRegisterInfo *TRI,
                          const MachineRegisterInfo &MRI,
                          const ReverseBlockModel &BM, MachineInstr &Q,
                          Register VIn, SmallVectorImpl<RevPlan> &Out) {
  StringRef Name = TII->getName(Q.getOpcode());
  if (!Name.starts_with("PseudoVRGATHER_VV_") || Name.contains("MASK") ||
      Q.getNumExplicitDefs() != 1 || !Q.getOperand(0).isReg() ||
      !Q.getOperand(2).isReg() || Q.getOperand(2).getReg() != VIn ||
      !Q.getOperand(3).isReg() || Q.getOperand(3).getSubReg() ||
      !(Q.getOperand(1).isUndef() || !Q.getOperand(1).getReg()))
    return;
  Register T = Q.getOperand(0).getReg(), Idx = Q.getOperand(3).getReg();
  if (!T.isVirtual() || !Idx.isVirtual())
    return;
  unsigned Log2SEW = log2SEWOf(Q);
  const MachineOperand &QVL = vlOperand(Q);
  auto TIt = BM.Regs.find(T);
  if (TIt == BM.Regs.end())
    return;
  const ReverseBlockModel::RegRefs &TR = TIt->second;

  // The index: vrsub(vid.v, c) with the same VL and SEW as each other.
  MachineInstr *Rsub = MRI.getUniqueVRegDef(Idx);
  StringRef RName;
  if (!Rsub || !isPlainRVV(TII, *Rsub, RName) || !hasUndefPassthru(*Rsub) ||
      !(RName.starts_with("PseudoVRSUB_VX_") ||
        RName.starts_with("PseudoVRSUB_VI_")) ||
      log2SEWOf(*Rsub) != Log2SEW)
    return;
  MachineInstr *Vid = MRI.getUniqueVRegDef(Rsub->getOperand(2).getReg());
  StringRef VName;
  if (!Vid || !TII->getName(Vid->getOpcode()).starts_with("PseudoVID_V_") ||
      !sameVLAndSEW(*Vid, *Rsub) ||
      !(Vid->getOperand(1).isUndef() || !Vid->getOperand(1).getReg()))
    return;
  const MachineOperand &C = Rsub->getOperand(3);
  if (C.isReg() && !MRI.getUniqueVRegDef(C.getReg()))
    return;

  // The gathers, in program order, and the slide, if any.
  SmallVector<MachineInstr *, 8> Gathers;
  for (unsigned D : TR.Defs)
    Gathers.push_back(BM.Instrs[D]);
  if (Gathers.empty() || Gathers.front() != &Q ||
      unsigned(std::distance(MRI.def_begin(T), MRI.def_end())) !=
          Gathers.size())
    return;
  unsigned QPos = BM.Pos.lookup(&Q);
  unsigned LastG = BM.Pos.lookup(Gathers.back());
  auto VIt = BM.Regs.find(VIn);
  if (VIt == BM.Regs.end() || VIt->second.defIn(QPos, LastG + 1))
    return;
  MachineInstr *Slide = nullptr;
  Register Y = T;
  std::optional<int64_t> N;

  if (constVL(QVL, MRI)) {
    // One gather, whole registers, index n-1-i over VL n.
    N = constVL(QVL, MRI);
    if (Gathers.size() != 1 || Q.getOperand(0).getSubReg() ||
        Q.getOperand(2).getSubReg() || !sameVLAndSEW(Q, *Rsub) ||
        constOperand(C, MRI) != *N - 1)
      return;
  } else {
    // VLMAX gathers on LMUL1 pieces of an L-register group, and a slide.
    if (!QVL.isImm() || !Name.contains("_M1_") ||
        !sameVLAndSEW(Q, *Rsub) || !C.isReg() ||
        !isVlenbMinus(C.getReg(), 3 - int(Log2SEW), 1, MRI))
      return;
    unsigned L = TRI->getRegClassWeight(MRI.getRegClass(T)).RegWeight;
    if (!isPowerOf2_32(L) || L > 8 || Gathers.size() != L ||
        MRI.getRegClass(VIn) != MRI.getRegClass(T))
      return;
    SmallBitVector Seen(L);
    for (MachineInstr *G : Gathers) {
      if (G->getOpcode() != Q.getOpcode() || G->getOperand(2).getReg() != VIn ||
          G->getOperand(3).getReg() != Idx || !sameVLAndSEW(*G, Q))
        return;
      unsigned DK = 0, SK = 0;
      if (L > 1) {
        auto K = [&](const MachineOperand &MO, unsigned &Out) {
          if (!MO.getSubReg())
            return false;
          StringRef SN = TRI->getSubRegIndexName(MO.getSubReg());
          return SN.consume_front("sub_vrm1_") && !SN.getAsInteger(10, Out) &&
                 Out < L;
        };
        if (!K(G->getOperand(0), DK) || !K(G->getOperand(2), SK) ||
            SK != L - 1 - DK || Seen.test(DK))
          return;
      } else if (G->getOperand(0).getSubReg() || G->getOperand(2).getSubReg()) {
        return;
      }
      Seen.set(DK);
    }
    // The only reader of T is the slide down by VLMAX - n.
    if (TR.LiveOut || TR.Reads.size() != 1)
      return;
    Slide = BM.Instrs[TR.Reads.front()];
    StringRef SName;
    if (!isPlainRVV(TII, *Slide, SName) || !hasUndefPassthru(*Slide) ||
        !SName.starts_with("PseudoVSLIDEDOWN_VX_") ||
        Slide->getOperand(2).getReg() != T || log2SEWOf(*Slide) != Log2SEW ||
        !Slide->getOperand(3).isReg())
      return;
    N = constVL(vlOperand(*Slide), MRI);
    if (!N || *N <= 0 ||
        !isVlenbMinus(Slide->getOperand(3).getReg(),
                      3 + int(Log2_32(L)) - int(Log2SEW), *N, MRI))
      return;
    Y = Slide->getOperand(0).getReg();
  }

  MachineInstr *Last = Slide ? Slide : &Q;
  RevPlan P;
  P.Kind = RevPlan::Flip;
  P.Fwd.push_back(&Q);
  for (MachineInstr *G : Gathers)
    if (G != &Q)
      P.Fwd.push_back(G);
  if (Slide)
    P.Fwd.push_back(Slide);
  // Their scalar operands (c, VL) must still be valid too, when they are here.
  for (MachineInstr *MI : {Rsub, Vid})
    if (MI->getParent() == Q.getParent())
      P.Fwd.push_back(MI);
  P.Clone.push_back(Vid);
  P.Clone.push_back(Rsub);
  P.Clone.append(Gathers.begin(), Gathers.end());
  if (Slide)
    P.Clone.push_back(Slide);
  P.FlipSrc = VIn;
  P.FlipIdx = Idx;
  P.Opcs = {Last->getOpcode()};
  unsigned YFrom = BM.Pos.lookup(Last) + 1;
  P.Ins.push_back({Y, YFrom, {{Slide ? Slide->getOpcode() : Q.getOpcode(), 0}}});
  P.Valid = makeRange(Slide ? vlOperand(*Slide) : QVL, Log2SEW, Last->getDesc(),
                      1u << Log2SEW);
  Out.push_back(P);
}

// The constant pool entry a register's address computation points at:
// ADDI (AUIPC %const.k), %pcrel_lo(sym), or ADDI (LUI %hi(%const.k)),
// %lo(%const.k).
static std::optional<unsigned> constantPoolIndexOf(Register Addr,
                                                   const MachineRegisterInfo &MRI) {
  const MachineInstr *A = Addr.isVirtual() ? MRI.getUniqueVRegDef(Addr) : nullptr;
  if (!A || A->getOpcode() != RISCV::ADDI || !A->getOperand(1).isReg())
    return std::nullopt;
  const MachineInstr *Hi = MRI.getUniqueVRegDef(A->getOperand(1).getReg());
  if (!Hi || (Hi->getOpcode() != RISCV::AUIPC && Hi->getOpcode() != RISCV::LUI) ||
      !Hi->getOperand(1).isCPI() || Hi->getOperand(1).getOffset())
    return std::nullopt;
  const MachineOperand &Lo = A->getOperand(2);
  if (Hi->getOpcode() == RISCV::AUIPC ? !Lo.isMCSymbol()
                                      : !Lo.isCPI() || Lo.getIndex() !=
                                                           Hi->getOperand(1).getIndex() ||
                                            Lo.getOffset())
    return std::nullopt;
  return Hi->getOperand(1).getIndex();
}

// Permutation gather (a constant shuffle with every lane of v used once):
//   idx = vle<eew>(%const.k) [; vsext/vzext.vf2]   ; a permutation p of 0..n-1
//   y   = vrgather.vv / vrgatherei16.vv(v, idx)    ; y[i] = v[p[i]], VL n
// Its inverse is another gather, by the inverse permutation q (q[p[i]] = i):
// v[j] = y[q[j]]. q is a compile-time constant; the rebuild loads it from a
// new constant pool entry with a clone of the index load (and extension).
static void matchPermPlan(const TargetInstrInfo *TII,
                          const TargetRegisterInfo *TRI,
                          const MachineRegisterInfo &MRI,
                          const MachineConstantPool &MCP,
                          const ReverseBlockModel &BM, MachineInstr &Q,
                          Register VIn, SmallVectorImpl<RevPlan> &Out) {
  StringRef Name = TII->getName(Q.getOpcode());
  if (!(Name.starts_with("PseudoVRGATHER_VV_") ||
        Name.starts_with("PseudoVRGATHEREI16_VV_")) ||
      Name.contains("MASK") || Q.getNumExplicitDefs() != 1 ||
      !Q.getOperand(0).isReg() || Q.getOperand(0).getSubReg() ||
      !Q.getOperand(2).isReg() || Q.getOperand(2).getReg() != VIn ||
      Q.getOperand(2).getSubReg() || !Q.getOperand(3).isReg() ||
      Q.getOperand(3).getSubReg() || !hasUndefPassthru(Q))
    return;
  std::optional<int64_t> N = constVL(vlOperand(Q), MRI);
  Register Idx = Q.getOperand(3).getReg();
  if (!N || *N <= 0 || !Idx.isVirtual())
    return;

  // idx = [vsext/vzext.vf2] (vle from the constant pool).
  MachineInstr *Ext = nullptr, *Load = MRI.getUniqueVRegDef(Idx);
  bool Signed = false;
  if (!Load)
    return;
  StringRef EName;
  if (isPlainRVV(TII, *Load, EName) &&
      (EName.starts_with("PseudoVSEXT_VF2_") ||
       EName.starts_with("PseudoVZEXT_VF2_"))) {
    if (!hasUndefPassthru(*Load))
      return;
    Ext = Load;
    Signed = EName.starts_with("PseudoVSEXT");
    Load = MRI.getUniqueVRegDef(Ext->getOperand(2).getReg());
    if (!Load)
      return;
  }
  StringRef LName = TII->getName(Load->getOpcode());
  if (!LName.starts_with("PseudoVLE") || !LName.contains("_V_") ||
      LName.contains("FF") || LName.contains("MASK") ||
      Load->getNumExplicitDefs() != 1 || !Load->getOperand(2).isReg() ||
      !(Load->getOperand(1).isUndef() || !Load->getOperand(1).getReg()) ||
      !Load->hasOneMemOperand())
    return;
  const PseudoSourceValue *PSV = (*Load->memoperands_begin())->getPseudoValue();
  if (!PSV || PSV->kind() != PseudoSourceValue::ConstantPool)
    return;
  std::optional<int64_t> LoadVL = constVL(vlOperand(*Load), MRI);
  std::optional<unsigned> CPI =
      constantPoolIndexOf(Load->getOperand(2).getReg(), MRI);
  if (!LoadVL || *LoadVL < *N || !CPI || *CPI >= MCP.getConstants().size())
    return;
  const MachineConstantPoolEntry &E = MCP.getConstants()[*CPI];
  if (E.isMachineConstantPoolEntry())
    return;
  const auto *CDS = dyn_cast<ConstantDataSequential>(E.Val.ConstVal);
  if (!CDS || !CDS->getElementType()->isIntegerTy() ||
      CDS->getNumElements() < uint64_t(*N))
    return;

  // The first n indices must be a permutation of 0..n-1.
  SmallVector<int64_t, 64> Inv(*N, -1), Fwd(*N);
  for (int64_t I = 0; I != *N; ++I) {
    APInt V = CDS->getElementAsAPInt(I);
    int64_t P = Signed ? V.getSExtValue() : int64_t(V.getZExtValue());
    if (P < 0 || P >= *N || Inv[P] >= 0)
      return;
    Inv[P] = I;
    Fwd[I] = P;
  }
  SmallVector<Constant *, 64> Elts;
  for (unsigned I = 0, NE = CDS->getNumElements(); I != NE; ++I)
    Elts.push_back(I < uint64_t(*N)
                       ? ConstantInt::get(CDS->getElementType(), Inv[I])
                       : CDS->getElementAsConstant(I));

  RevPlan P;
  P.Kind = RevPlan::PermGather;
  P.Fwd.push_back(&Q);
  P.Clone.push_back(Load);
  if (Ext)
    P.Clone.push_back(Ext);
  P.Clone.push_back(&Q);
  for (MachineInstr *MI : P.Clone)
    if (MI != &Q && MI->getParent() == Q.getParent())
      P.Fwd.push_back(MI);
  P.FlipSrc = VIn;
  P.FlipIdx = Idx;
  P.PermInv = ConstantVector::get(Elts);
  P.PermAlign = E.getAlign();
  P.Opcs = {Q.getOpcode()};
  unsigned QPos = BM.Pos.lookup(&Q);
  P.Ins.push_back({Q.getOperand(0).getReg(), QPos + 1, {{Q.getOpcode(), 0}}});
  P.Valid = makeRange(vlOperand(Q), log2SEWOf(Q), Q.getDesc(),
                      1u << log2SEWOf(Q));
  // The rebuilt index (and its unextended load) live briefly.
  P.TempW = TRI->getRegClassWeight(MRI.getRegClass(Idx)).RegWeight +
            (Ext ? TRI->getRegClassWeight(
                       MRI.getRegClass(Load->getOperand(0).getReg()))
                       .RegWeight
                 : 0);
  Out.push_back(P);

  // A self-inverse permutation (every butterfly step of a reduction: a swap
  // of two blocks) is undone by the same gather, so the rebuild can read Q's
  // own index instead of loading a new one: one instruction. The index is
  // then an input like y, and must be live where the rebuild goes.
  if (Fwd == Inv) {
    RevPlan R;
    R.Kind = RevPlan::PermGather;
    R.PermReuseIdx = true;
    R.Fwd.push_back(&Q);
    R.Clone.push_back(&Q);
    R.FlipSrc = VIn;
    R.FlipIdx = Idx;
    R.Opcs = {Q.getOpcode()};
    R.Ins.push_back({Q.getOperand(0).getReg(), QPos + 1, {{Q.getOpcode(), 0}}});
    R.Ins.push_back({Idx, QPos, {{Q.getOpcode(), 3}}});
    R.Valid = P.Valid;
    Out.push_back(R);
  }
}

// The rebuilds of v from a group of instructions starting at Q, a use of VIn
// (v itself or a copy of it), that inverts that group as a whole.
static void matchGroupPlans(const TargetInstrInfo *TII,
                            const MachineRegisterInfo &MRI,
                            const ReverseBlockModel &BM, MachineInstr &Q,
                            Register VIn, SmallVectorImpl<RevPlan> &Out) {
  StringRef Name;
  if (!isPlainRVV(TII, Q, Name))
    return;
  // Every group but the zip reads v as its vector source, operand 2.
  bool VIsSrc = Q.getOperand(2).getReg() == VIn;
  auto RegIt = BM.Regs.find(VIn);
  if (RegIt == BM.Regs.end())
    return;
  const ReverseBlockModel::RegRefs &VR = RegIt->second;
  unsigned QPos = BM.Pos.lookup(&Q);
  StringRef N = Name.drop_front(strlen("PseudoV"));
  StringRef L = lmulSuffix(Name);
  unsigned Log2SEW = log2SEWOf(Q);
  unsigned SEW = 1u << Log2SEW;
  Register QDst = Q.getOperand(0).getReg();
  auto DefIn = [&](Register R, unsigned From, unsigned To) {
    auto It = BM.Regs.find(R);
    return It != BM.Regs.end() && It->second.defIn(From, To);
  };
  // The first def of R after position Pos.
  auto NextDef = [&](Register R, unsigned Pos) -> MachineInstr * {
    auto It = BM.Regs.find(R);
    if (It == BM.Regs.end())
      return nullptr;
    for (unsigned D : It->second.Defs)
      if (D > Pos)
        return BM.Instrs[D];
    return nullptr;
  };
  auto Opcode = [&](const Twine &S) -> unsigned {
    return findPseudo(TII, S).value_or(0);
  };

  // Widen: v = narrow(Y). SEW of vsext/vzext is the wide destination's; of
  // vfwcvt and vwadd(u).vx it is the narrow source's.
  if (VIsSrc && hasUndefPassthru(Q)) {
    unsigned Rev = 0, VLog2SEW = Log2SEW;
    if (N.starts_with("SEXT_VF2_") || N.starts_with("ZEXT_VF2_")) {
      Rev = Opcode("PseudoVNSRL_WI_" + halfLMul(L));
      VLog2SEW = Log2SEW - 1;
    } else if ((N.starts_with("WADD_VX_") || N.starts_with("WADDU_VX_")) &&
               Q.getOperand(3).isReg() &&
               Q.getOperand(3).getReg() == RISCV::X0) {
      Rev = Opcode("PseudoVNSRL_WI_" + L); // vwcvt(u).x.x.v
    } else if (N.starts_with("FWCVT_F_F_V_")) {
      Rev = Opcode("PseudoVFNCVT_F_F_W_" + L);
    } else if (N.starts_with("FWCVT_F_X_V_") || N.starts_with("FWCVT_F_XU_V_")) {
      // Every SEW bit integer is exact in a 2*SEW bit float, and converts
      // back exactly.
      StringRef LMulOnly = L.split('_').first;
      Rev = Opcode(Twine("PseudoVFNCVT_RTZ_") +
                   (N.starts_with("FWCVT_F_XU") ? "XU" : "X") + "_F_W_" +
                   LMulOnly);
    }
    if (Rev) {
      RevPlan P;
      P.Kind = RevPlan::Narrow;
      P.Fwd = {&Q};
      P.Ins.push_back({QDst, QPos + 1, {{Rev, 2}}});
      P.Opcs = {Rev};
      P.Valid = makeRange(vlOperand(Q), VLog2SEW, TII->get(Rev),
                          1u << VLog2SEW);
      Out.push_back(P);
    }
  }

  // Interleave: Z = zip(A, B) is vwaddu.vv Z, A, B ; vwmaccu.vx Z, -1, B.
  // A = vnsrl.wi(Z, 0), B = vnsrl.w[ix](Z, SEW).
  if (N.starts_with("WADDU_VV_") && hasUndefPassthru(Q) &&
      Q.getOperand(3).isReg()) {
    Register B = Q.getOperand(3).getReg();
    MachineInstr *W = NextDef(QDst, QPos);
    StringRef WName;
    if (W && isPlainRVV(TII, *W, WName) &&
        WName == ("PseudoVWMACCU_VX_" + L).str() &&
        W->getOperand(1).getReg() == QDst && W->getOperand(3).isReg() &&
        W->getOperand(3).getReg() == B && !W->getOperand(3).getSubReg() &&
        !Q.getOperand(3).getSubReg() && sameVLAndSEW(Q, *W) &&
        !DefIn(B, QPos + 1, BM.Pos.lookup(W))) {
      std::optional<int64_t> C = constOperand(W->getOperand(2), MRI);
      uint64_t Ones = maskTrailingOnes<uint64_t>(SEW);
      if (C && (uint64_t(*C) & Ones) == Ones) {
        unsigned WPos = BM.Pos.lookup(W);
        for (unsigned Half = 0; Half != 2; ++Half) {
          if (Q.getOperand(2 + Half).getReg() != VIn ||
              Q.getOperand(2 + Half).getSubReg())
            continue;
          unsigned Rev = Opcode(Twine("PseudoVNSRL_") +
                                (Half && SEW > 31 ? "WX_" : "WI_") + L);
          if (!Rev)
            continue;
          RevPlan P;
          P.Kind = RevPlan::ZipHalf;
          P.Fwd = {&Q, W};
          P.Ins.push_back({QDst, WPos + 1, {{Rev, 2}}});
          P.Opcs = {Rev};
          P.Half = Half;
          P.Valid = makeRange(vlOperand(Q), Log2SEW, TII->get(Rev), SEW);
          Out.push_back(P);
        }
      }
    }
  }

  // Deinterleave: Lo = vnsrl(v, 0) and Hi = vnsrl(v, SEW) together hold all
  // of v: v = zip(Lo, Hi), whose elements are 2*SEW bits wide.
  if (VIsSrc && (N.starts_with("NSRL_WI_") || N.starts_with("NSRL_WX_")) &&
      hasUndefPassthru(Q) && constOperand(Q.getOperand(3), MRI) == 0) {
    for (unsigned R : VR.Reads) {
      MachineInstr &H = *BM.Instrs[R];
      StringRef HName;
      if (&H == &Q || !isPlainRVV(TII, H, HName) || !hasUndefPassthru(H) ||
          !HName.starts_with("PseudoVNSRL_W") || lmulSuffix(HName) != L ||
          H.getOperand(2).getReg() != VIn || !sameVLAndSEW(Q, H) ||
          constOperand(H.getOperand(3), MRI) != int64_t(SEW) ||
          DefIn(VIn, std::min(QPos, R) + 1, std::max(QPos, R) + 1))
        continue;
      unsigned Add = Opcode("PseudoVWADDU_VV_" + L);
      unsigned Mac = Opcode("PseudoVWMACCU_VX_" + L);
      if (!Add || !Mac)
        break;
      RevPlan P;
      P.Kind = RevPlan::Unzip;
      P.Fwd = {&Q, &H};
      P.Ins.push_back({QDst, QPos + 1, {{Add, 2}}});
      P.Ins.push_back({H.getOperand(0).getReg(), R + 1, {{Add, 3}, {Mac, 3}}});
      P.Opcs = {Add, Mac};
      P.Valid = makeRange(vlOperand(Q), Log2SEW, TII->get(Add), 2 * SEW);
      Out.push_back(P);
      break;
    }
  }

  // Rotation by k within n = VL lanes: vslidedown r, v, k ; vslideup r, v,
  // n-k. Its inverse is the rotation by n-k.
  if (VIsSrc && N.starts_with("SLIDEDOWN_V") && hasUndefPassthru(Q)) {
    std::optional<int64_t> NL = constVL(vlOperand(Q), MRI);
    std::optional<int64_t> K = constOperand(Q.getOperand(3), MRI);
    MachineInstr *W = NextDef(QDst, QPos);
    StringRef WName;
    if (NL && K && *K > 0 && *K < *NL && W && isPlainRVV(TII, *W, WName) &&
        WName.starts_with("PseudoVSLIDEUP_V") && lmulSuffix(WName) == L &&
        W->getOperand(1).getReg() == QDst &&
        W->getOperand(2).getReg() == VIn && sameVLAndSEW(Q, *W) &&
        constOperand(W->getOperand(3), MRI) == *NL - *K &&
        !DefIn(VIn, QPos + 1, BM.Pos.lookup(W) + 1)) {
      unsigned Down = Opcode(Twine("PseudoVSLIDEDOWN_") +
                             amountForm(W->getOperand(3)) + L);
      unsigned Up =
          Opcode(Twine("PseudoVSLIDEUP_") + amountForm(Q.getOperand(3)) + L);
      if (Down && Up) {
        RevPlan P;
        P.Kind = RevPlan::Rotate;
        P.Fwd = {&Q, W};
        P.Ins.push_back({QDst, BM.Pos.lookup(W) + 1, {{Down, 2}, {Up, 2}}});
        P.Opcs = {Down, Up};
        P.Valid = makeRange(vlOperand(Q), Log2SEW, TII->get(Up), SEW);
        Out.push_back(P);
      }
    }
  }

  // Rotation of the bits of each element: y = v >> s | v << (SEW - s). With
  // s = SEW/2 this swaps adjacent lanes of half the width, which is how
  // LLVM lowers a <1,0,3,2,...> shuffle. Inverse: v = y << s | y >> SEW-s.
  if (VIsSrc && (N.starts_with("SRL_VX_") || N.starts_with("SRL_VI_")) &&
      hasUndefPassthru(Q)) {
    std::optional<int64_t> S1 = constOperand(Q.getOperand(3), MRI);
    for (unsigned R : VR.Reads) {
      if (!S1 || *S1 <= 0 || *S1 >= int64_t(SEW))
        break;
      MachineInstr &H = *BM.Instrs[R];
      StringRef HName;
      if (&H == &Q || !isPlainRVV(TII, H, HName) || !hasUndefPassthru(H) ||
          !HName.starts_with("PseudoVSLL_V") || lmulSuffix(HName) != L ||
          H.getOperand(2).getReg() != VIn || !sameVLAndSEW(Q, H) ||
          constOperand(H.getOperand(3), MRI) != int64_t(SEW) - *S1)
        continue;
      Register A = QDst, Bv = H.getOperand(0).getReg();
      unsigned Last = std::max(QPos, R);
      if (DefIn(VIn, std::min(QPos, R) + 1, Last + 1))
        continue;
      // The OR (or XOR / ADD: the bits are disjoint) of the two shifts.
      MachineInstr *O = nullptr;
      auto AIt = BM.Regs.find(A);
      if (AIt == BM.Regs.end())
        continue;
      for (unsigned OP : AIt->second.Reads) {
        if (OP <= Last)
          continue;
        MachineInstr &C = *BM.Instrs[OP];
        StringRef CName;
        if (isPlainRVV(TII, C, CName) && hasUndefPassthru(C) &&
            (CName == ("PseudoVOR_VV_" + L).str() ||
             CName == ("PseudoVXOR_VV_" + L).str() ||
             CName == ("PseudoVADD_VV_" + L).str()) &&
            C.getOperand(3).isReg() && sameVLAndSEW(Q, C) &&
            ((C.getOperand(2).getReg() == A &&
              C.getOperand(3).getReg() == Bv) ||
             (C.getOperand(2).getReg() == Bv &&
              C.getOperand(3).getReg() == A)) &&
            !DefIn(A, QPos + 1, OP) && !DefIn(Bv, R + 1, OP)) {
          O = &C;
          break;
        }
      }
      if (!O)
        continue;
      unsigned Srl =
          Opcode(Twine("PseudoVSRL_") + amountForm(H.getOperand(3)) + L);
      unsigned Sll =
          Opcode(Twine("PseudoVSLL_") + amountForm(Q.getOperand(3)) + L);
      if (!Srl || !Sll)
        break;
      RevPlan P;
      P.Kind = RevPlan::BitRotate;
      P.Fwd = {&Q, &H, O};
      P.Ins.push_back(
          {O->getOperand(0).getReg(), BM.Pos.lookup(O) + 1, {{Srl, 2}, {Sll, 2}}});
      P.Opcs = {Srl, Sll, O->getOpcode()};
      P.Valid = makeRange(vlOperand(Q), Log2SEW, O->getDesc(), SEW);
      Out.push_back(P);
      break;
    }
  }
}

namespace {
// Rematerialization information attached to a live range, as in V8's
// integrated rematerialization (Vardanyan, Asryan, Buchatskiy, "Integrated
// Register Rematerialization in JavaScript V8 JIT Compiler"). Every operand
// of a value that tells how to recompute it is recorded, and all of them are
// propagated to the value's live interval, so each use position has every
// option to choose from:
//
//   3: c = a + b      OutputC3: direct,  c = a + b  (from a, b)
//   5: e = c - d      InputC5:  reverse, c = e + d  (from e, d)
//   6: f = c + b      InputC6:  reverse, c = f - b  (from f, b)
//
// An option is only valid at a point where the values it reads are still
// live, which is checked at the rematerialization site.
//
// LiveInterval has no room for such data, and the LiveIntervals analysis is
// not kept up to date by the transforms of this pass, so the interval is
// modelled per block on the positions of ReverseBlockModel.
struct RematOption {
  unsigned At;  // position of the operand (def or use) the option comes from
  RevPlan Plan; // Direct for the def, a reverse rebuild for a use
};
struct RematInterval {
  Register Reg;
  // Positions reading the value, sorted. A vreg defined several times in the
  // block (a masked load whose passthru is tied to the result) is modelled
  // from its last def on, the only value whose uses can all be rewritten.
  SmallVector<unsigned, 4> Uses;
  SmallVector<RematOption, 4> Options;
};
} // namespace

// Reverse rematerialization with the timing of -custom-remat: a value whose
// next use is at least RematGap instructions after the previous one is killed
// at the previous use and recomputed right before the distant one, so its
// register is free in between:
//   %Y = OP(%v, %K)    ; q: an option of v's interval, v = REV(%Y, %K)
//   use1 %v            ; v now dies here
//   ... >= RematGap instructions ...
//   %v2 = REV(%Y, %K)  ; new
//   other              ; one instruction between, as -custom-remat does
//   use2 %v2           ; this and every later use of v are rewritten
bool ExpandPseudos::ProcessReverseRematChain(MachineFunction &MF) {
  DenseMap<Register, unsigned> ClassOf;
  SmallVector<SmallVector<Register, 4>, 16> Classes;
  auto MembersOf = [&](Register R) -> ArrayRef<Register> {
    auto It = ClassOf.find(R);
    if (It == ClassOf.end())
      return ArrayRef<Register>();
    return Classes[It->second];
  };
  // Deps[C]: the classes some member of class C was recomputed from,
  // transitively. A value is never recomputed from something derived from
  // itself: y' = a' * b from a' = y / b would only keep two copies alive in
  // a cycle.
  SmallVector<DenseSet<unsigned>, 16> Deps;
  // Copies made by this pass: never direct-rematerialized again (only split
  // by reverse steps).
  DenseSet<Register> Created;
  auto ClassIdOf = [&](Register R) {
    auto [It, Inserted] = ClassOf.try_emplace(R, Classes.size());
    if (Inserted) {
      Classes.push_back({R});
      Deps.emplace_back();
    }
    return It->second;
  };
  auto DerivedFrom = [&](Register A, Register V) {
    auto IA = ClassOf.find(A), IV = ClassOf.find(V);
    return IA != ClassOf.end() && IV != ClassOf.end() &&
           (IA->second == IV->second || Deps[IA->second].count(IV->second));
  };
  auto SameValue = [&](Register A, Register B) {
    if (A == B)
      return true;
    auto IA = ClassOf.find(A), IB = ClassOf.find(B);
    return IA != ClassOf.end() && IB != ClassOf.end() &&
           IA->second == IB->second;
  };
  auto Weight = [&](Register X) {
    return isVectorReg(X, *MRI)
               ? TRI->getRegClassWeight(MRI->getRegClass(X)).RegWeight
               : 0u;
  };

  using Model = ReverseBlockModel;
  auto LiveOuts = computeVRegLiveOuts(MF);
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    Model BM;
    unsigned Rounds = 0, MaxRounds = 0;
    while (true) {
      BM.build(MBB, *MRI, *TRI, LiveOuts[&MBB]);
      if (!MaxRounds)
        MaxRounds = 2 * BM.Instrs.size() + 16;
      if (++Rounds > MaxRounds)
        break;

      // A register equal to R that is usable in this block: its def
      // position (-1 when live in) and the last position it is needed at.
      struct Alt {
        Register Reg;
        int Def;
        unsigned Last;
        unsigned W; // pressure weight if vector, else 0
      };
      // R itself, valid from position From on (R must not be redefined in
      // [From, To)), plus R's copies. A copy may take several instructions
      // to build (a zip, a rotation); it is complete after its last def.
      auto AltsOf = [&](Register R, unsigned From, unsigned To,
                        SmallVectorImpl<Alt> &Out) {
        auto It = BM.Regs.find(R);
        if (It != BM.Regs.end() && !It->second.HasSubReg &&
            !It->second.defIn(From, To))
          Out.push_back({R, -1, It->second.last(), Weight(R)});
        for (Register M : MembersOf(R)) {
          if (M == R)
            continue;
          auto MI = BM.Regs.find(M);
          if (MI == BM.Regs.end() || MI->second.HasSubReg)
            continue;
          const Model::RegRefs &RR = MI->second;
          int Def = RR.Defs.empty() ? -1 : (int)RR.Defs.back();
          Out.push_back({M, Def, RR.last(), Weight(M)});
        }
      };

      // Steps 1 and 2 of the paper: the options found on the operands of
      // each value, propagated to its live interval.
      auto BuildInterval = [&](Register V, RematInterval &LI) {
        const Model::RegRefs &VR = BM.Regs[V];
        LI.Reg = V;
        unsigned Start = VR.Defs.empty() ? 0 : VR.Defs.back() + 1;
        for (unsigned R : VR.Reads)
          if (R >= Start)
            LI.Uses.push_back(R);

        // OutputV: v's own def, recomputed from its inputs. The def may be
        // replayed together with the defs of its vector source, a chain of
        // up to MaxDirectSteps instructions back to a root (Figure 2(b) of
        // Bahi & Eisenbeis): v = OP1(OP2(root, K2), K1). There is one option
        // per chain length, each reading its own root, so the site can pick
        // the shortest chain whose root is still live there. A link's second
        // source, if a vector the chain does not compute, is an input too;
        // at most one such input is allowed.
        if (VR.Defs.size() == 1 && MRI->hasOneDef(V) && !Created.count(V)) {
          constexpr unsigned MaxDirectSteps = 8;
          SmallVector<MachineInstr *, 8> Links; // v's def first
          for (MachineInstr *L = BM.Instrs[VR.Defs[0]];
               L && Links.size() < MaxDirectSteps;) {
            std::optional<ReverseOp> Op = matchDirectOp(TII, *L, *MRI);
            if (!Op || !BM.Pos.count(L) || L->getOperand(0).getSubReg() ||
                L->getOperand(2).getSubReg() ||
                !MRI->hasOneDef(L->getOperand(0).getReg()))
              break;
            Links.push_back(L);

            // The option replaying Links, from the source of the last one.
            Register Root = L->getOperand(2).getReg();
            SmallSet<Register, 8> Computed; // by the chain itself
            for (MachineInstr *C : Links)
              Computed.insert(C->getOperand(0).getReg());
            RevPlan Plan;
            Plan.Kind = RevPlan::Direct;
            Plan.Fwd.assign(Links.begin(), Links.end());
            Plan.Clone.assign(Links.rbegin(), Links.rend());
            Plan.Op = *matchDirectOp(TII, *Links.front(), *MRI);
            Plan.Ins.push_back({Root, BM.Pos.lookup(L), {{L->getOpcode(), 2}}});
            bool Ok = !SameValue(Root, V) && !Computed.count(Root);
            for (MachineInstr *C : Links) {
              const MachineOperand &KOp = C->getOperand(3);
              if (!Ok || !KOp.isReg() || !KOp.getReg())
                continue;
              Register K = KOp.getReg();
              if (!K.isVirtual() || SameValue(K, V)) {
                Ok = false;
              } else if (!isVectorReg(K, *MRI) || Computed.count(K)) {
                // A scalar is checked like VL; a chain value is rebuilt.
              } else if (K == Root) {
                Plan.Ins[0].Slots.push_back({C->getOpcode(), 3});
              } else if (Plan.Ins.size() == 1) {
                Plan.Ins.push_back({K, BM.Pos.lookup(C), {{C->getOpcode(), 3}}});
              } else if (Plan.Ins[1].Reg == K) {
                Plan.Ins[1].From = std::min(Plan.Ins[1].From, BM.Pos.lookup(C));
                Plan.Ins[1].Slots.push_back({C->getOpcode(), 3});
              } else {
                Ok = false; // a second vector input
              }
            }
            if (!Ok)
              break;
            LI.Options.push_back({VR.Defs[0], std::move(Plan)});

            // Extend the chain through the root's own def.
            L = Root.isVirtual() && MRI->hasOneDef(Root)
                    ? MRI->getUniqueVRegDef(Root)
                    : nullptr;
          }
        }

        // InputV: every use q of v, or of a copy of v, that can be inverted.
        SmallVector<std::pair<unsigned, Register>, 8> Qs;
        for (unsigned R : LI.Uses)
          Qs.push_back({R, V});
        for (Register M : MembersOf(V)) {
          if (M == V)
            continue;
          auto MIt = BM.Regs.find(M);
          if (MIt != BM.Regs.end())
            for (unsigned R : MIt->second.Reads)
              Qs.push_back({R, M});
        }
        for (auto [QPos, VIn] : Qs) {
          MachineInstr *Q = BM.Instrs[QPos];
          if (Q->getNumExplicitOperands() >= 6) {
            for (unsigned Vs : {2u, 3u}) {
              const MachineOperand &VOp = Q->getOperand(Vs);
              if (!VOp.isReg() || VOp.getReg() != VIn || VOp.getSubReg())
                continue;
              std::optional<ReverseOp> Op = matchReverseOp(TII, *Q, Vs);
              if (!Op)
                continue;
              Register Y = Q->getOperand(0).getReg();
              unsigned QK = Vs == 2 ? 3 : 2;
              const MachineOperand &KOp = Q->getOperand(QK);
              if (!Y.isVirtual() || SameValue(Y, V) ||
                  (KOp.isReg() && SameValue(KOp.getReg(), V)))
                continue;
              bool KIsReg = KOp.isReg() && KOp.getReg();
              if (KIsReg && (!KOp.getReg().isVirtual() ||
                             (Vs == 3 && !isVectorReg(KOp.getReg(), *MRI))))
                continue;
              RematOption O{QPos, {}};
              RevPlan &Plan = O.Plan;
              Plan.Kind = RevPlan::Elementwise;
              Plan.Fwd = {Q};
              Plan.Op = *Op;
              Plan.QK = QK;
              Plan.KIsReg = KIsReg;
              unsigned YIdx = Op->KFirst ? 3 : 2, KIdx = Op->KFirst ? 2 : 3;
              Plan.Ins.push_back({Y, QPos + 1, {{Op->Opcode, YIdx}}});
              if (KIsReg)
                Plan.Ins.push_back({KOp.getReg(), QPos, {{Op->Opcode, KIdx}}});
              // Integer ADD/SUB/RSUB/XOR invert exactly; the float ops round.
              Plan.Exact = !TII->getName(Op->Opcode).starts_with("PseudoVF");
              Plan.Valid = makeRange(vlOperand(*Q), log2SEWOf(*Q),
                                     TII->get(Op->Opcode),
                                     1u << log2SEWOf(*Q));
              LI.Options.push_back(std::move(O));
            }
          }
          // Non-elementwise: invert a group of instructions reading v.
          SmallVector<RevPlan, 2> Plans;
          matchGroupPlans(TII, *MRI, BM, *Q, VIn, Plans);
          matchFlipPlan(TII, TRI, *MRI, BM, *Q, VIn, Plans);
          matchPermPlan(TII, TRI, *MRI, *MF.getConstantPool(), BM, *Q, VIn,
                        Plans);
          for (RevPlan &Plan : Plans)
            if (llvm::none_of(Plan.Ins, [&](const RevPlan::Input &In) {
                  return SameValue(In.Reg, V);
                }))
              LI.Options.push_back({QPos, std::move(Plan)});
        }
      };

      struct Site {
        Register V;
        RevPlan Plan;
        Alt In[2]{};
        unsigned At = 0, P = 0, U1 = 0, U2 = 0, NextDef = 0;
        int Benefit = 0;
      } Best;
      bool HaveBest = false;

      // Step 4 of the paper: whether option O of v's interval can recompute
      // v for the gap (U1, U2), and where. Updates Best if it is better than
      // the options seen so far for this gap.
      auto TryOption = [&](const RematInterval &LI, const RematOption &O,
                           unsigned U1, unsigned U2,
                           ArrayRef<MachineInstr *> Moved) {
        Register V = LI.Reg;
        const RevPlan &Plan = O.Plan;
        bool IsDirect = Plan.Kind == RevPlan::Direct;
        const MCInstrDesc &DstDesc = TII->get(Plan.dstOpcode());
        if (const TargetRegisterClass *RC = TII->getRegClass(DstDesc, 0))
          if (!TRI->getCommonSubClass(MRI->getRegClass(V), RC))
            return;
        // A direct clone computes exactly what v's def did, tail included;
        // any other rebuild only the elements in Plan.Valid.
        if (!IsDirect && !llvm::all_of(Moved, [&](MachineInstr *U) {
              return readsOnlyPrefix(TII, *MRI, *U, V, Plan.Valid);
            }))
          return;

        // The latest site the stall gap to U2 allows (U2 itself at m4 / m8):
        // the rebuild writes v right before Instrs[PGap].
        MachineInstr &Consumer = *BM.Instrs[U2];
        unsigned DstLMul = pseudoLMul(DstDesc);
        StallGap Need = minStallGap(
            DstLMul ? DstLMul : Weight(V), DstDesc.mayLoad(),
            Consumer.mayStore(),
            isIntMulName(TII->getName(Plan.dstOpcode())) ||
                isIntMulName(TII->getName(Consumer.getOpcode())));
        LLVM_DEBUG(dbgs() << "  Stall gap: LMUL "
                          << (DstLMul ? DstLMul : Weight(V)) << ", "
                          << (DstDesc.mayLoad() ? "L" : "C")
                          << (Consumer.mayStore() ? "S" : "C") << ", need "
                          << Need.Vec << " vector / " << Need.Scalar
                          << " scalar, rebuild "
                          << TII->getName(Plan.dstOpcode()) << "\n");
        unsigned PGap = U2, GapVec = 0, GapScalar = 0;
        while (!coversStallGap(Need, GapVec, GapScalar)) {
          if (PGap <= U1 + 1)
            return;
          --PGap;
          ++(isVectorInstr(*BM.Instrs[PGap], *MRI) ? GapVec : GapScalar);
        }

        SmallVector<Alt, 4> Alts[2];
        assert(Plan.Ins.size() <= 2);
        for (unsigned I = 0, E = Plan.Ins.size(); I != E; ++I) {
          const RevPlan::Input &In = Plan.Ins[I];
          SmallVector<Alt, 4> All;
          AltsOf(In.Reg, In.From, U2, All);
          for (Alt &A : All) {
            if (DerivedFrom(A.Reg, V))
              continue;
            if (llvm::any_of(In.Slots, [&](auto S) {
                  const TargetRegisterClass *RC =
                      TII->getRegClass(TII->get(S.first), S.second);
                  return RC &&
                         !TRI->getCommonSubClass(MRI->getRegClass(A.Reg), RC);
                }))
              continue;
            // A scalar may be kept alive a bit longer; it does not count
            // against the vector registers.
            if (!A.W)
              A.Last = Model::Inf;
            Alts[I].push_back(A);
          }
          if (Alts[I].empty())
            return;
        }

        unsigned MaxFwd = 0;
        for (MachineInstr *F : Plan.Fwd)
          MaxFwd = std::max(MaxFwd, BM.Pos.lookup(F));
        unsigned From = 0;
        for (const RevPlan::Input &In : Plan.Ins)
          From = std::max(From, In.From);
        // The scalar operands the rebuild reuses (VL, amounts, FRM) must
        // still hold their values at P.
        auto ScalarsValid = [&](unsigned P) {
          for (MachineInstr *F : Plan.Fwd) {
            unsigned FPos = BM.Pos.lookup(F);
            for (const MachineOperand &MO : F->explicit_uses()) {
              if (!MO.isReg() || !MO.getReg() || MO.getReg() == RISCV::X0 ||
                  MO.isUndef() || isVectorReg(MO.getReg(), *MRI))
                continue;
              if (MO.getReg().isVirtual()) {
                auto It = BM.Regs.find(MO.getReg());
                if (It == BM.Regs.end() || It->second.defIn(FPos, P))
                  return false;
              } else if (physRegModifiedIn(BM, MO.getReg(), FPos, P, TRI)) {
                return false;
              }
            }
          }
          return llvm::none_of(Plan.Fwd, [&](MachineInstr *F) {
            return F->readsRegister(RISCV::FRM, TRI) &&
                   physRegModifiedIn(BM, RISCV::FRM, BM.Pos.lookup(F), P, TRI);
          });
        };

        Alt None{Register(), -1, Model::Inf, 0};
        ArrayRef<Alt> Second =
            Plan.Ins.size() > 1 ? ArrayRef<Alt>(Alts[1]) : ArrayRef<Alt>(None);
        for (const Alt &A0 : Alts[0]) {
          for (const Alt &A1 : Second) {
            // As close to U2 as the stall gap allows, as -custom-remat, or
            // right after the last use of a vector input if that is earlier.
            unsigned P = PGap;
            for (const Alt *A : {&A0, &A1})
              if (A->Last != Model::Inf)
                P = std::min(P, A->Last + 1);
            // The hole left in v's live range must be as long as the one
            // -custom-remat leaves.
            if (P < U1 + RematGap || (int)P <= A0.Def || (int)P <= A1.Def ||
                P <= MaxFwd || P < From || !ScalarsValid(P))
              continue;
            int Benefit = int(P - U1 - 1) * int(Weight(V));
            // Prefer the latest site, then an exact rebuild, then the
            // fewest new instructions, then the option nearest the site.
            auto Key = [](const RevPlan &Pl, unsigned P, unsigned At) {
              return std::make_tuple(P, Pl.Exact, -int(Pl.numNew()), At);
            };
            if (HaveBest &&
                Key(Plan, P, O.At) <= Key(Best.Plan, Best.P, Best.At))
              continue;
            Best = {V,  Plan, {A0, A1}, O.At, P, U1, U2,
                    (unsigned)BM.Instrs.size(), Benefit};
            HaveBest = true;
          }
        }
      };

      // In program order: the first value with a distant use that one of its
      // options can serve is transformed, then the block is re-analyzed.
      for (Register V : BM.Order) {
        const Model::RegRefs &VR = BM.Regs[V];
        // A value read in another block keeps its register past this
        // block's uses anyway.
        if (!isVectorReg(V, *MRI) || VR.HasSubRegDef || VR.Reads.empty() ||
            VR.LiveOut)
          continue;
        // The register allocator already recomputes a value whose def reads
        // no vector register (e.g. vid.v, vmv.v.x: only VL and scalars)
        // wherever it is needed; replacing it with copies computed from other
        // vectors only takes that away.
        if (MachineInstr *Def = MRI->getUniqueVRegDef(V))
          if (TII->isReMaterializable(*Def) &&
              llvm::none_of(Def->all_uses(), [&](const MachineOperand &MO) {
                return !MO.isUndef() && isVectorReg(MO.getReg(), *MRI);
              }))
            continue;

        RematInterval LI;
        BuildInterval(V, LI);
        if (LI.Options.empty())
          continue;
        bool HasGap = false;
        for (unsigned I = 1, E = LI.Uses.size(); I < E; ++I)
          HasGap |= LI.Uses[I] - LI.Uses[I - 1] - 1 >= RematGap;
        LLVM_DEBUG(if (HasGap && Rounds == 1) {
          dbgs() << "RematInterval " << printReg(V, TRI) << " uses";
          for (unsigned U : LI.Uses)
            dbgs() << " " << U;
          dbgs() << "\n";
          for (const RematOption &O : LI.Options)
            dbgs() << "  @" << O.At << " " << planName(O.Plan.Kind) << ": "
                   << *BM.Instrs[O.At];
        });

        // Step 4: the timing of -custom-remat. Every gap of at least
        // RematGap instructions between two uses is a site; the options are
        // checked there, the earliest gap any of them can serve is taken.
        for (unsigned I = 1, E = LI.Uses.size(); I < E && !HaveBest; ++I) {
          unsigned U1 = LI.Uses[I - 1], U2 = LI.Uses[I];
          if (U2 - U1 - 1 < RematGap)
            continue;
          SmallVector<MachineInstr *, 4> Moved;
          for (unsigned J = I; J != E; ++J)
            Moved.push_back(BM.Instrs[LI.Uses[J]]);
          // Only an operand before the distant use can describe v there
          // (a use of a copy of v may lie inside the gap).
          for (const RematOption &O : LI.Options)
            if (O.At < U2)
              TryOption(LI, O, U1, U2, Moved);
        }
        if (HaveBest)
          break;
      }
      if (!HaveBest)
        break;

      // Graph transformation: insert v' = REV(Y', K') at P and move every
      // use of v from U2 up to the next def of v to it.
      Register V = Best.V;
      const RevPlan &Plan = Best.Plan;
      MachineInstr *Q = Plan.Fwd[0];
      MachineInstr *InsertBefore = BM.Instrs[Best.P];
      MachineBasicBlock::iterator IP = InsertBefore->getIterator();
      const DebugLoc &DL = InsertBefore->getDebugLoc();
      Register NewV = MRI->createVirtualRegister(MRI->getRegClass(V));
      if (const TargetRegisterClass *RC =
              TII->getRegClass(TII->get(Plan.dstOpcode()), 0))
        MRI->constrainRegClass(NewV, RC);
      Register In0 = Best.In[0].Reg, In1 = Best.In[1].Reg;
      for (unsigned I = 0, E = Plan.Ins.size(); I != E; ++I)
        for (auto [Opc, Idx] : Plan.Ins[I].Slots)
          if (const TargetRegisterClass *RC =
                  TII->getRegClass(TII->get(Opc), Idx))
            MRI->constrainRegClass(Best.In[I].Reg, RC);

      LLVM_DEBUG(dbgs() << "ReverseRemat[" << planName(Plan.Kind)
                        << (Plan.PermReuseIdx ? ", self-inverse" : "")
                        << (Plan.Exact ? "" : ", rounded")
                        << "]: recompute " << printReg(V, TRI) << " from "
                        << printReg(In0, TRI);
                 if (In1) dbgs() << ", " << printReg(In1, TRI);
                 dbgs() << " (inverting";
                 for (MachineInstr *F : Plan.Fwd) dbgs() << "\n    " << *F;
                 dbgs() << ")  killed after " << *BM.Instrs[Best.U1]
                        << "  rematerialized before " << *InsertBefore
                        << "  freeing " << Best.Benefit
                        << " register-instruction(s)\n");

      SmallVector<MachineInstr *, 4> NewMIs;
      auto Build = [&](unsigned Opc, Register Dst) {
        MachineInstrBuilder MIB = BuildMI(MBB, IP, DL, TII->get(Opc));
        MIB.addReg(Dst, RegState::Define);
        NewMIs.push_back(MIB);
        return MIB;
      };
      // VL and SEW of Like, and a fully agnostic policy if Opc has one.
      auto AddVLSEW = [&](MachineInstrBuilder &MIB, const MachineInstr &Like,
                          unsigned Log2SEW) {
        MIB.add(copyUse(vlOperand(Like)));
        MIB.addImm(Log2SEW);
        if (RISCVII::hasVecPolicyOp(MIB->getDesc().TSFlags))
          MIB.addImm(RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC);
      };
      // The policy of From, which runs the same opcode family.
      auto AddVLSEWPolicy = [&](MachineInstrBuilder &MIB,
                                const MachineInstr &From) {
        MIB.add(copyUse(vlOperand(From)));
        MIB.addImm(log2SEWOf(From));
        if (RISCVII::hasVecPolicyOp(MIB->getDesc().TSFlags))
          MIB.addImm(RISCVII::hasVecPolicyOp(From.getDesc().TSFlags)
                         ? From.getOperand(
                                   RISCVII::getVecPolicyOpNum(From.getDesc()))
                               .getImm()
                         : RISCVVType::TAIL_AGNOSTIC |
                               RISCVVType::MASK_AGNOSTIC);
      };
      auto NewGPR = [&](int64_t Imm) {
        Register R = MRI->createVirtualRegister(&RISCV::GPRRegClass);
        NewMIs.push_back(
            BuildMI(MBB, IP, DL, TII->get(RISCV::ADDI), R)
                .addReg(RISCV::X0)
                .addImm(Imm));
        return R;
      };

      switch (Plan.Kind) {
      case RevPlan::Direct: {
        // Replay the chain from its root (In0) to v, each link writing a
        // fresh register and the last one NewV. The second vector input,
        // if any, is In1; scalar operands and chain values are kept or
        // remapped as they are.
        DenseMap<Register, Register> Map;
        Map[Plan.Ins[0].Reg] = In0;
        if (In1)
          Map[Plan.Ins[1].Reg] = In1;
        for (MachineInstr *L : Plan.Clone) {
          Register Old = L->getOperand(0).getReg();
          Register Dst = L == Q ? NewV
                                : MRI->createVirtualRegister(MRI->getRegClass(Old));
          MachineInstr *C = MF.CloneMachineInstr(L);
          for (MachineOperand &MO : C->operands()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            if (MO.getReg() == Old) {
              MO.setReg(Dst); // the def and its tied undef passthru
            } else {
              auto It = Map.find(MO.getReg());
              if (It != Map.end())
                MO.setReg(It->second);
            }
            if (MO.isDef())
              MO.setIsDead(false);
          }
          Map[Old] = Dst;
          C->setDebugLoc(DL);
          MBB.insert(IP, C);
          NewMIs.push_back(C);
        }
        break;
      }
      case RevPlan::Elementwise: {
        const MachineOperand &QK = Q->getOperand(Plan.QK);
        MachineOperand KOp =
            Plan.KIsReg ? MachineOperand::CreateReg(In1, false)
                        : MachineOperand::CreateImm(
                              Plan.Op.NegateImm ? -QK.getImm() : QK.getImm());
        MachineInstrBuilder MIB = Build(Plan.Op.Opcode, NewV);
        MIB.addReg(NewV, RegState::Undef);
        if (Plan.Op.KFirst) {
          MIB.add(KOp);
          MIB.addReg(In0);
        } else {
          MIB.addReg(In0);
          MIB.add(KOp);
        }
        for (unsigned I = 4, E = Q->getNumExplicitOperands(); I != E; ++I)
          MIB.add(Q->getOperand(I));
        // Implicit operands added after selection (e.g. the FRM read of a
        // dynamic rounding mode) are not in the descriptor.
        for (const MachineOperand &MO : Q->implicit_operands())
          if (MO.isReg() &&
              !(MO.isDef() ? MIB->definesRegister(MO.getReg(), TRI)
                           : MIB->readsRegister(MO.getReg(), TRI)))
            MIB.add(MO);
        MIB->setFlags(Q->getFlags());
        break;
      }
      case RevPlan::Narrow: {
        unsigned Opc = Plan.Opcs[0];
        StringRef Name = TII->getName(Opc);
        MachineInstrBuilder MIB = Build(Opc, NewV);
        MIB.addReg(NewV, RegState::Undef).addReg(In0);
        if (Name.starts_with("PseudoVNSRL_WI"))
          MIB.addImm(0);
        bool ReadsFRM = Name.starts_with("PseudoVFNCVT_F_F");
        if (ReadsFRM) // exact, so any rounding mode will do
          MIB.addImm(RISCVFPRndMode::DYN);
        AddVLSEW(MIB, *Q, Plan.Valid.Log2SEW);
        if (ReadsFRM)
          MIB.addReg(RISCV::FRM, RegState::Implicit);
        MIB->setFlag(MachineInstr::NoFPExcept);
        break;
      }
      case RevPlan::ZipHalf: {
        unsigned Opc = Plan.Opcs[0];
        unsigned SEW = 1u << log2SEWOf(*Q);
        MachineOperand Shift = MachineOperand::CreateImm(Plan.Half ? SEW : 0);
        if (TII->getName(Opc).starts_with("PseudoVNSRL_WX"))
          Shift = MachineOperand::CreateReg(NewGPR(SEW), false);
        MachineInstrBuilder MIB = Build(Opc, NewV);
        MIB.addReg(NewV, RegState::Undef).addReg(In0).add(Shift);
        AddVLSEW(MIB, *Q, log2SEWOf(*Q));
        break;
      }
      case RevPlan::Unzip: {
        // NewV = Lo + Hi + Hi * (2^SEW - 1) = Lo | Hi << SEW, in 2*SEW bits.
        Register Ones = NewGPR(-1);
        MachineInstrBuilder Add = Build(Plan.Opcs[0], NewV);
        Add.addReg(NewV, RegState::Undef).addReg(In0).addReg(In1);
        AddVLSEW(Add, *Q, log2SEWOf(*Q));
        MachineInstrBuilder Mac = Build(Plan.Opcs[1], NewV);
        Mac.addReg(NewV).addReg(Ones).addReg(In1);
        AddVLSEW(Mac, *Q, log2SEWOf(*Q));
        break;
      }
      case RevPlan::Rotate: {
        // Down by n-k, then the k lanes that fell off the end back up.
        MachineInstr *W = Plan.Fwd[1];
        MachineInstrBuilder Down = Build(Plan.Opcs[0], NewV);
        Down.addReg(NewV, RegState::Undef)
            .addReg(In0)
            .add(copyUse(W->getOperand(3)));
        AddVLSEWPolicy(Down, *Q);
        MachineInstrBuilder Up = Build(Plan.Opcs[1], NewV);
        Up.addReg(NewV).addReg(In0).add(copyUse(Q->getOperand(3)));
        AddVLSEWPolicy(Up, *W);
        break;
      }
      case RevPlan::Flip: {
        // Replay the flip on y: clone vid, vrsub, the gathers and the slide
        // with v replaced by y, the slide (or gather) writing NewV, and
        // fresh registers for the index and the gathered group. The index
        // is reused instead when it is still live here anyway.
        DenseMap<Register, Register> Map;
        Map[Plan.FlipSrc] = In0;
        Map[Plan.Clone.back()->getOperand(0).getReg()] = NewV;
        auto IdxIt = BM.Regs.find(Plan.FlipIdx);
        bool ReuseIdx = IdxIt != BM.Regs.end() && !IdxIt->second.HasSubReg &&
                        MRI->hasOneDef(Plan.FlipIdx) &&
                        IdxIt->second.last() >= Best.P;
        for (MachineInstr *F : Plan.Clone) {
          if (ReuseIdx && (F == Plan.Clone[0] || F == Plan.Clone[1]))
            continue;
          MachineInstr *C = MF.CloneMachineInstr(F);
          for (MachineOperand &MO : C->operands()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register R = MO.getReg();
            if (MO.isDef() && !Map.count(R))
              Map[R] = MRI->createVirtualRegister(MRI->getRegClass(R));
            auto It = Map.find(R);
            if (It != Map.end())
              MO.setReg(It->second);
            if (MO.isDef())
              MO.setIsDead(false);
          }
          C->setDebugLoc(DL);
          MBB.insert(IP, C);
          NewMIs.push_back(C);
        }
        if (ReuseIdx)
          RegsToClearKillFlags.insert(Plan.FlipIdx);
        break;
      }
      case RevPlan::PermGather: {
        if (Plan.PermReuseIdx) {
          // The gather again, on y with Q's own index.
          MachineInstr *C = MF.CloneMachineInstr(Q);
          for (MachineOperand &MO : C->operands()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            if (MO.getReg() == Q->getOperand(0).getReg())
              MO.setReg(NewV);
            else if (MO.getReg() == Plan.FlipSrc)
              MO.setReg(In0);
            else if (MO.getReg() == Plan.FlipIdx)
              MO.setReg(In1);
            if (MO.isDef())
              MO.setIsDead(false);
          }
          C->setDebugLoc(DL);
          MBB.insert(IP, C);
          NewMIs.push_back(C);
          break;
        }
        // Address the inverse permutation (a new constant pool entry) the
        // way PseudoLLA expands, then clone the index load, its extension
        // and the gather, reading y and the new index.
        unsigned CPI = MF.getConstantPool()->getConstantPoolIndex(
            Plan.PermInv, Plan.PermAlign);
        Register Hi = MRI->createVirtualRegister(&RISCV::GPRRegClass);
        Register Addr = MRI->createVirtualRegister(&RISCV::GPRRegClass);
        MCSymbol *Sym = MF.getContext().createNamedTempSymbol("pcrel_hi");
        MachineInstr *Auipc =
            BuildMI(MBB, IP, DL, TII->get(RISCV::AUIPC), Hi)
                .addConstantPoolIndex(CPI, 0, RISCVII::MO_PCREL_HI);
        Auipc->setPreInstrSymbol(MF, Sym);
        NewMIs.push_back(Auipc);
        NewMIs.push_back(BuildMI(MBB, IP, DL, TII->get(RISCV::ADDI), Addr)
                             .addReg(Hi)
                             .addSym(Sym, RISCVII::MO_PCREL_LO));
        DenseMap<Register, Register> Map;
        Map[Plan.FlipSrc] = In0;
        Map[Plan.Clone.back()->getOperand(0).getReg()] = NewV;
        Map[Plan.Clone.front()->getOperand(2).getReg()] = Addr;
        for (MachineInstr *F : Plan.Clone) {
          MachineInstr *C = MF.CloneMachineInstr(F);
          for (MachineOperand &MO : C->operands()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register R = MO.getReg();
            if (MO.isDef() && !Map.count(R))
              Map[R] = MRI->createVirtualRegister(MRI->getRegClass(R));
            auto It = Map.find(R);
            if (It != Map.end())
              MO.setReg(It->second);
            if (MO.isDef())
              MO.setIsDead(false);
          }
          C->setDebugLoc(DL);
          MBB.insert(IP, C);
          NewMIs.push_back(C);
        }
        break;
      }
      case RevPlan::BitRotate: {
        // v = y >> (SEW - s) | y << s, with the shift amounts of the
        // original swapped.
        MachineInstr *H = Plan.Fwd[1], *O = Plan.Fwd[2];
        const TargetRegisterClass *RC = MRI->getRegClass(V);
        Register Lo = MRI->createVirtualRegister(RC);
        Register Hi = MRI->createVirtualRegister(RC);
        MachineInstrBuilder Srl = Build(Plan.Opcs[0], Lo);
        Srl.addReg(Lo, RegState::Undef)
            .addReg(In0)
            .add(copyUse(H->getOperand(3)));
        AddVLSEWPolicy(Srl, *Q);
        MachineInstrBuilder Sll = Build(Plan.Opcs[1], Hi);
        Sll.addReg(Hi, RegState::Undef)
            .addReg(In0)
            .add(copyUse(Q->getOperand(3)));
        AddVLSEWPolicy(Sll, *H);
        MachineInstrBuilder Or = Build(Plan.Opcs[2], NewV);
        Or.addReg(NewV, RegState::Undef).addReg(Lo).addReg(Hi);
        AddVLSEWPolicy(Or, *O);
        break;
      }
      }
      for (MachineInstr *MI : NewMIs)
        for (MachineOperand &MO : MI->all_uses())
          if (MO.getReg()) {
            MO.setIsKill(false);
            if (MO.getReg().isVirtual())
              RegsToClearKillFlags.insert(MO.getReg());
          }

      MachineBasicBlock::iterator End =
          Best.NextDef < BM.Instrs.size()
              ? BM.Instrs[Best.NextDef]->getIterator()
              : MBB.end();
      for (MachineInstr &I :
           make_range(MachineBasicBlock::iterator(InsertBefore), End))
        for (MachineOperand &MO : I.operands())
          if (MO.isReg() && MO.isUse() && MO.getReg() == V) {
            MO.setReg(NewV);
            MO.setIsKill(false);
          }

      Created.insert(NewV);
      // NewV holds the same value as V, so later steps may use either. It
      // was computed from its inputs, and from whatever they came from.
      unsigned NewC;
      if (MRI->hasOneDef(V)) {
        NewC = ClassIdOf(V);
        Classes[NewC].push_back(NewV);
        ClassOf[NewV] = NewC;
      } else {
        NewC = ClassIdOf(NewV);
      }
      SmallVector<Register, 2> Srcs = {In0};
      if (In1)
        Srcs.push_back(In1);
      for (Register R : Srcs) {
        unsigned C = ClassIdOf(R);
        DenseSet<unsigned> Add = Deps[C];
        Add.insert(C);
        Deps[NewC].insert(Add.begin(), Add.end());
      }

      // A direct chain that took over every use leaves v's def dead, and
      // with it any link whose result only fed the next one.
      if (Plan.Kind == RevPlan::Direct)
        for (MachineInstr *L : Plan.Fwd) {
          Register R = L->getOperand(0).getReg();
          // L's own undef passthru reads R too.
          if (llvm::any_of(MRI->use_nodbg_operands(R),
                           [&](MachineOperand &MO) { return MO.getParent() != L; }))
            break;
          for (MachineOperand &MO : make_early_inc_range(MRI->reg_operands(R)))
            if (MO.isDebug())
              MO.setReg(Register());
          L->eraseFromParent();
        }

      RegsToClearKillFlags.insert(V);
      LISValid = false;
      Changed = true;
    }
  }
  return Changed;
}

// Vector register pressure right after each instruction of MBB, tracked with
// RegPressureTracker as RISCVRegisterPressure does. Needs valid LiveIntervals.
void ExpandPseudos::computeBlockPressure(
    MachineBasicBlock &MBB, DenseMap<const MachineInstr *, unsigned> &Out) {
  if (VRSet == ~0u)
    return;
  IntervalPressure Pressure;
  RegPressureTracker RPTracker(Pressure);
  RPTracker.init(MBB.getParent(), &RegClassInfo, LIS, &MBB, MBB.begin(), false,
                 false);
  while (RPTracker.getPos() != MBB.end()) {
    const MachineInstr &MI = *RPTracker.getPos();
    RPTracker.advance();
    Out[&MI] = RPTracker.getRegSetPressureAtPos()[VRSet];
  }
}

// Positions, pressure events, register ranges and live interval lengths of
// the vector registers in MBB. The last use of a register does not count
// towards pressure, so a register is live after its first reference up to but
// not including its last one: +LMUL at the first, -LMUL at the last.
void ExpandPseudos::computeBlockUsage(MachineBasicBlock &MBB) {
  BlockUsage &BU = Usage[&MBB];
  for (auto &KV : BU.LIL)
    VRegLIL[KV.first] -= KV.second;
  BU = BlockUsage();

  DenseMap<Register, std::pair<unsigned, bool>> FirstIsUse; // first pos, is use
  DenseSet<Register> HasDef;
  DenseMap<Register, unsigned> OpenPhys; // physical vector reg -> last ref
  unsigned Pos = 0;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    BU.Pos[&MI] = Pos;
    BU.VecPrefix.push_back(BU.VecPrefix.empty() ? 0 : BU.VecPrefix.back());
    bool Touches = false;
    // Uses first, then defs, so a physical register redefined by the
    // instruction that also reads it is closed after the read.
    for (int Pass = 0; Pass < 2; ++Pass)
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || !isVectorReg(MO.getReg(), *MRI))
          continue;
        if (MO.isDef() != (Pass == 1))
          continue;
        // An undef use (the passthru tied to a def) reads nothing.
        if (MO.isUse() && MO.isUndef())
          continue;
        Touches = true;
        Register R = MO.getReg();
        if (R.isVirtual()) {
          auto It = BU.Range.try_emplace(R, Pos, Pos).first;
          It->second.second = Pos;
          FirstIsUse.try_emplace(R, Pos, MO.isUse());
          if (MO.isDef())
            HasDef.insert(R);
        } else if (MO.isDef()) {
          auto It = OpenPhys.find(R);
          if (It != OpenPhys.end())
            BU.Events.push_back({It->second, R, 1, false, true});
          BU.Events.push_back({Pos, R, 1, true, true});
          OpenPhys[R] = Pos;
        } else {
          auto It = OpenPhys.find(R);
          if (It != OpenPhys.end())
            It->second = Pos;
        }
      }
    if (Touches)
      ++BU.VecPrefix.back();
    ++Pos;
  }
  for (auto &KV : OpenPhys)
    BU.Events.push_back({KV.second, KV.first, 1, false, true});
  BU.VecPrefix.push_back(BU.VecPrefix.empty() ? 0 : BU.VecPrefix.back());

  for (auto &KV : BU.Range) {
    Register R = KV.first;
    unsigned W = TRI->getRegClassWeight(MRI->getRegClass(R)).RegWeight;
    // Live out of the block if referenced in another block, or carried around
    // a loop (first reference is a use and there is a def).
    bool LiveOut = FirstIsUse[R].second && HasDef.count(R);
    for (MachineInstr &U : MRI->reg_instructions(R))
      if (U.getParent() != &MBB) {
        LiveOut = true;
        break;
      }
    BU.Events.push_back({KV.second.first, R, (int)W, true, true});
    BU.Events.push_back({KV.second.second, R, (int)W, false, !LiveOut});
    BU.LIL[R] = (BU.VecPrefix[KV.second.second] - BU.VecPrefix[KV.second.first]) * W;
  }
  llvm::stable_sort(BU.Events, [](const BlockUsage::RefEvent &A,
                                  const BlockUsage::RefEvent &B) {
    return A.Pos < B.Pos;
  });
  BU.NextEvent.assign(Pos + 1, 0);
  for (unsigned I = 0, E = 0; I <= Pos; ++I) {
    while (E < BU.Events.size() && BU.Events[E].Pos < I)
      ++E;
    BU.NextEvent[I] = E;
  }
  for (auto &KV : BU.LIL)
    VRegLIL[KV.first] += KV.second;
}

// Compute the vector register pressure after each instruction and the live
// interval length of each vector vreg for the code as it is now.
void ExpandPseudos::computeVectorRegUsage(MachineFunction &MF,
                                          LiveIntervals &LI) {
  LIS = &LI;
  LISValid = true;
  VRPressure.clear();
  VRegLIL.clear();
  Usage.clear();

  VRSet = ~0u;
  for (unsigned i = 0, e = TRI->getNumRegPressureSets(); i != e; ++i)
    if (StringRef(TRI->getRegPressureSetName(i)) == "VR")
      VRSet = i;
  RegClassInfo.runOnMachineFunction(MF);

  for (MachineBasicBlock &MBB : MF) {
    computeBlockUsage(MBB);
    computeBlockPressure(MBB, VRPressure);
  }
}

// Peak vector-register pressure across the whole function and the sum of
// live-interval lengths (SLIL), computed purely from Usage/VRegLIL (a plain
// operand scan, kept up to date by every transform via computeBlockUsage).
// Unlike computeVectorRegUsage's own MaxRP (via RegPressureTracker), this
// does not need LiveIntervals, so it is safe to call after transforms that
// leave LISValid false, to see whether pressure and SLIL actually went down.
std::pair<unsigned, unsigned> ExpandPseudos::computeCurrentUsage() const {
  unsigned MaxPressure = 0;
  for (auto &BlockKV : Usage) {
    const BlockUsage &BU = BlockKV.second;
    unsigned Running = 0;
    for (unsigned I = 0, E = BU.Events.size(); I != E;) {
      unsigned Pos = BU.Events[I].Pos;
      unsigned J = I;
      while (J != E && BU.Events[J].Pos == Pos)
        ++J;
      // Apply this position's whole net delta (its DEFs and its own
      // operands' LAST USEs) atomically before checking the max: they
      // belong to the same instruction, whose destination can reuse a
      // dying source's register, so they are never simultaneously live.
      for (unsigned K = I; K != J; ++K)
        if (BU.Events[K].Pressure)
          Running += BU.Events[K].IsDef ? BU.Events[K].W : -BU.Events[K].W;
      MaxPressure = std::max(MaxPressure, Running);
      I = J;
    }
  }
  unsigned SLIL = 0;
  for (auto &KV : VRegLIL)
    SLIL += KV.second;
  return {MaxPressure, SLIL};
}

bool ExpandPseudos::runProcess(MachineFunction &MF) {

  bool EverMadeChange = false;

  // -custom-reverse includes -custom-remat: its direct options and the
  // load / cheap ALU remat are both direct rematerialization.
  if (Remat || Reverse) {
    bool Changed = ProcessRedundantReload(MF);
    Changed |= ProcessRematLoads(MF);
    EverMadeChange |= Changed;
    if (Changed) {
      // Instructions were created without updating LiveIntervals.
      LISValid = false;
      for (MachineBasicBlock &MBB : MF)
        computeBlockUsage(MBB);
    }
  }
  if (Reverse) {
    bool Changed = ProcessReverseRematChain(MF);
    EverMadeChange |= Changed;
    if (Changed) {
      // Instructions were created without updating LiveIntervals.
      LISValid = false;
      for (MachineBasicBlock &MBB : MF)
        computeBlockUsage(MBB);
    }
  }
  if (Sink)
    ProcessInSameBlock(MF);
  if (a) {
    ProcessInSameAffine(MF);
  }
  if (Thresh)
    EverMadeChange |= ProcessThreshold(MF);

  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();
  RematRegs.clear();

  return EverMadeChange;
}

bool ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "******** Expand Pseudos ********\n");

  STI = &MF.getSubtarget();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool AnyFlag = Sink || a || Thresh || Remat || Reverse || Copy;
  if (!AnyFlag) {
    dbgs()<<"Custom Sink disabled.\n";
  } else if (Sink) {
    std::string option = "";
    if (Sink) option += " Sink";
    if (Copy) option += " Copy";
    if (Remat || Reverse) option += " Remat";
    if (a) option += " Affine";
    if (Thresh) option += " Thresh";
    if (Reverse) option += " Reverse";
    dbgs()<<"Custom"<< option <<" enabled.\n";
  }

  computeVectorRegUsage(MF, getAnalysis<LiveIntervalsWrapperPass>().getLIS());

  // Without a -custom-* flag the pass changes nothing: a true baseline. It
  // only reports the vector register usage when debugging.
  if (!AnyFlag) {
    LLVM_DEBUG({
      auto [MaxRP, SLIL] = computeCurrentUsage();
      dbgs() << "Vector register usage: max VP " << MaxRP
             << ", SLIL " << SLIL << "\n";
    });
    return false;
  }

  LLVM_DEBUG({
    auto [MaxRP, SLIL] = computeCurrentUsage();
    dbgs() << "Before transforms: max VP " << MaxRP
           << ", SLIL " << SLIL << "\n";
  });

  bool MadeChange = false;

  // Pseudo expansions run whenever a -custom-* flag is set; the COPY-to-VMV
  // lowering only with -custom-copy.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      // Only expand pseudos.
      if (!MI.isPseudo())
        continue;

      // Give targets a chance to expand even standard pseudos.
      if (TII->expandPostRAPseudo(MI)) {
        LLVM_DEBUG(dbgs() << "expandPseudo: "<<MI);
        MadeChange = true;
        continue;
      }

      // Expand standard pseudos.
      switch (MI.getOpcode()) {
      //case TargetOpcode::SUBREG_TO_REG:
      //  MadeChange |= LowerSubregToReg(&MI);
      //  break;
      case TargetOpcode::COPY:
        if (Copy && LowerCopy(MBB, MI)) {
          MadeChange = true;
          LLVM_DEBUG(dbgs()<<"lowerCopy success.\n");
        }
        break;
      case TargetOpcode::DBG_VALUE:
        continue;
      case TargetOpcode::INSERT_SUBREG:
      case TargetOpcode::EXTRACT_SUBREG:
        llvm_unreachable("Sub-register pseudos should have been eliminated.");
      }
    }
  }
  // The usage the transforms start from includes the lowered copies.
  if (MadeChange)
    for (MachineBasicBlock &MBB : MF)
      computeBlockUsage(MBB);

  MadeChange |= runProcess(MF);

  LLVM_DEBUG({
    auto [MaxRP, SLIL] = computeCurrentUsage();
    dbgs() << "After transforms: max VP " << MaxRP
           << ", SLIL " << SLIL << "\n";
  });

  return MadeChange;
}
